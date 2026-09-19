from threading import Event
from time import monotonic, sleep
import unittest

from harness.background import BackgroundManager, COMPLETED, FAILED, PENDING, RUNNING


class BackgroundManagerTests(unittest.TestCase):
    def manager(self, **options):
        settings = {"max_concurrent": 2, "default_timeout": 2, **options}
        manager = BackgroundManager(**settings)
        self.addCleanup(manager.shutdown)
        return manager

    def wait_for(self, predicate, timeout=1):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            value = predicate()
            if value:
                return value
            sleep(0.005)
        return predicate()

    def test_task_lifecycle_reaches_completed_with_result(self):
        events = []
        manager = BackgroundManager(
            max_concurrent=1, default_timeout=2,
            on_event=lambda task_id, event: events.append((task_id, event)),
        )
        self.addCleanup(manager.shutdown)
        submitted = manager.submit(
            lambda stop: {"status": "success", "content": "完成"},
            "运行测试",
        )
        self.assertEqual(events[0][1]["task"]["status"], PENDING)
        task = self.wait_for(
            lambda: (lambda task: task if task["status"] == COMPLETED else False)(
                manager.check(submitted["task_id"]),
            ),
        )
        self.assertEqual(task["status"], COMPLETED)
        self.assertEqual(task["result"]["content"], "完成")
        self.assertEqual([event["type"] for _, event in events], [
            "background_submitted", "background_started", "background_finished",
        ])
        self.assertTrue(all(task_id == submitted["task_id"] for task_id, _ in events))

    def test_concurrency_limit_keeps_extra_tasks_pending(self):
        manager = self.manager(max_concurrent=1)
        first_done = Event()
        second_started = Event()

        def first(stop):
            first_done.wait(1)
            return {"status": "success", "content": "first"}

        def second(stop):
            second_started.set()
            return {"status": "success", "content": "second"}

        first_id = manager.submit(first, "first")["task_id"]
        second_id = manager.submit(second, "second")["task_id"]
        self.assertEqual(manager.check(first_id)["status"], RUNNING)
        self.assertEqual(manager.check(second_id)["status"], PENDING)
        self.assertFalse(second_started.is_set())
        first_done.set()
        task = self.wait_for(
            lambda: (lambda task: task if task["status"] == COMPLETED else False)(
                manager.check(second_id),
            ),
        )
        self.assertEqual(task["status"], COMPLETED)

    def test_exception_and_timeout_become_failed(self):
        manager = self.manager(default_timeout=1)
        failed_id = manager.submit(
            lambda stop: (_ for _ in ()).throw(RuntimeError("private")),
            "失败任务",
        )["task_id"]
        failed = self.wait_for(
            lambda: (lambda task: task if task["status"] == FAILED else False)(
                manager.check(failed_id),
            ),
        )
        self.assertEqual(failed["result"]["code"], "background_failed")
        self.assertNotIn("private", str(failed["result"]))

        timeout_id = manager.submit(
            lambda stop: (stop.wait(2), {"status": "error", "content": "stopped"})[1],
            "超时任务",
            timeout=1,
        )["task_id"]
        timed_out = self.wait_for(
            lambda: (lambda task: task if task["status"] == FAILED else False)(
                manager.check(timeout_id),
            ),
            timeout=1.5,
        )
        self.assertEqual(timed_out["result"]["code"], "background_timeout")

    def test_shutdown_signals_running_tasks(self):
        manager = self.manager()
        stopped = Event()
        task_id = manager.submit(
            lambda stop: (stop.wait(1), stopped.set(), {"status": "error", "content": "stopped"})[2],
            "等待停止",
        )["task_id"]
        self.wait_for(
            lambda: (lambda task: task if task["status"] == RUNNING else False)(
                manager.check(task_id),
            ),
        )
        manager.shutdown()
        self.assertTrue(stopped.wait(1))
        task = self.wait_for(
            lambda: (lambda task: task if task["status"] == FAILED else False)(
                manager.check(task_id),
            ),
        )
        self.assertEqual(task["result"]["content"], "stopped")

    def test_unknown_task_returns_none(self):
        self.assertIsNone(self.manager().check(99))


if __name__ == "__main__":
    unittest.main()
