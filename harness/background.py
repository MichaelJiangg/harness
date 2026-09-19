"""会话级后台任务管理：排队、状态、并发限制、超时和跨轮查询。"""

from collections import deque
from threading import Event, Lock, Thread
from time import monotonic


PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"


def _public_task(task, *, now=None):
    now = monotonic() if now is None else now
    snapshot = {key: value for key, value in task.items() if key != "_stop_event"}
    end = snapshot.get("finished_at") or now
    snapshot["elapsed"] = max(0.0, end - snapshot["created_at"])
    return snapshot


class BackgroundManager:
    """管理同进程后台任务；任务只读写入状态字典，结果由会话退出后丢弃。"""

    def __init__(self, *, max_concurrent, default_timeout, on_event=None):
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError("后台并发上限必须是正整数。")
        if type(default_timeout) is not int or not 1 <= default_timeout <= 300:
            raise ValueError("后台任务超时必须是 1～300 秒的整数。")
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.on_event = on_event if on_event is not None else lambda task_id, event: None
        self._lock = Lock()
        self._next_id = 1
        self._tasks = {}
        self._queue = deque()
        self._active = set()
        self._stop_events = {}
        self._abort = Event()

    def submit(self, func, description, *, timeout=None, on_event=None):
        if not callable(func):
            raise ValueError("后台任务必须是可调用对象。")
        if not isinstance(description, str) or not description.strip() or "\x00" in description:
            raise ValueError("后台任务描述必须是非空文本，且不能含空字符。")
        if timeout is None:
            timeout = self.default_timeout
        if type(timeout) is not int or not 1 <= timeout <= 300:
            raise ValueError("后台任务超时必须是 1～300 秒的整数。")
        event_callback = on_event if on_event is not None else self.on_event

        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            self._tasks[task_id] = {
                "task_id": task_id,
                "description": description,
                "status": PENDING,
                "created_at": monotonic(),
                "started_at": None,
                "finished_at": None,
                "result": None,
            }
            self._queue.append((task_id, func, timeout, event_callback))
        event_callback(task_id, {"type": "background_submitted",
                                 "task": self.check(task_id)})
        self._pump()
        return self.check(task_id)

    def check(self, task_id):
        with self._lock:
            task = self._tasks.get(task_id)
            return None if task is None else _public_task(task)

    def list_tasks(self):
        with self._lock:
            return [_public_task(task) for task in self._tasks.values()]

    def shutdown(self):
        self._abort.set()
        with self._lock:
            stop_events = list(self._stop_events.values())
        for event in stop_events:
            event.set()

    def _pump(self):
        while True:
            with self._lock:
                if len(self._active) >= self.max_concurrent or not self._queue:
                    return
                task_id, func, timeout, event_callback = self._queue.popleft()
                task = self._tasks[task_id]
                task["status"] = RUNNING
                task["started_at"] = monotonic()
                stop_event = Event()
                if self._abort.is_set():
                    stop_event.set()
                self._stop_events[task_id] = stop_event
                self._active.add(task_id)
            event_callback(task_id, {"type": "background_started",
                                     "task": self.check(task_id)})
            Thread(
                target=self._run,
                args=(task_id, func, timeout, stop_event, event_callback),
                daemon=True,
            ).start()

    def _run(self, task_id, func, timeout, stop_event, event_callback):
        timer = Thread(target=self._timeout, args=(task_id, timeout, stop_event, event_callback),
                       daemon=True)
        timer.start()
        try:
            result = func(stop_event)
            terminal = COMPLETED if isinstance(result, dict) and result.get("status") == "success" else FAILED
            if terminal == FAILED and not isinstance(result, dict):
                result = {"status": "error", "code": "background_failed",
                          "message": "后台任务未返回有效结果。"}
        except Exception:
            terminal = FAILED
            result = {"status": "error", "code": "background_failed",
                      "message": "后台任务执行失败。"}
        finally:
            updated = False
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None and task["status"] == RUNNING:
                    task["status"] = terminal
                    task["finished_at"] = monotonic()
                    task["result"] = result
                    updated = True
                self._active.discard(task_id)
                self._stop_events.pop(task_id, None)
            if updated:
                task = self.check(task_id)
                event_callback(task_id, {"type": "background_finished", "task": task})
            self._pump()

    def _timeout(self, task_id, timeout, stop_event, event_callback):
        if stop_event.wait(timeout):
            return
        stop_event.set()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and task["status"] == RUNNING:
                task["status"] = FAILED
                task["finished_at"] = monotonic()
                task["result"] = {
                    "status": "error", "code": "background_timeout",
                    "message": f"后台任务超过 {timeout} 秒，已标记失败并停止继续执行。",
                }
        task = self.check(task_id)
        if task is not None:
            event_callback(task_id, {"type": "background_finished", "task": task})
