"""经本地确认执行有限时的 bash 命令，分别限量收集两路输出。"""

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
from time import monotonic

from .definition import ToolDefinition
from .executor import ToolError


DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_OUTPUT_BYTES = 64 * 1024
_TRUNCATION_MARKER = b"\n... [output truncated] ...\n"
_CLEANUP_TIMEOUT = 0.4


DEFINITION = ToolDefinition(
    name="bash",
    description=(
        "从会话启动目录执行 bash 命令，支持管道和重定向。"
        "每次执行前必须由用户在本地确认完整命令，模型不能代替用户批准。"
        "用户拒绝后不得自行重试，需等待用户新的明确要求。"
        "命令以当前用户权限运行，工作目录不是文件访问沙箱。"
        "仅支持 macOS／Linux，标准输入关闭，不支持交互式程序或持久后台作业。"
        "timeout 默认 30 秒，最多 120 秒，超时会终止同组进程。"
        "返回 stdout、stderr、exit_code、timed_out 和 cancelled；"
        "每路输出最多保留 64 KiB，超限保留首尾并标记。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1, "description": "待执行的完整 bash 命令。"},
            "timeout": {
                "type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT,
                "default": DEFAULT_TIMEOUT, "description": "命令超时秒数，范围 1～120。",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    requires_confirmation=True,
    supports_cancellation=True,
)


def execute(arguments, workspace, *, abort=None):
    command = arguments["command"]
    timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
    if not command.strip() or "\x00" in command:
        raise ToolError("invalid_arguments", "command 必须是非空命令，且不能含空字符。")
    if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT:
        raise ToolError("invalid_arguments", "timeout 必须是 1～120 之间的整数。")
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise ToolError("unsupported_platform", "bash 工具当前仅支持 macOS／Linux。")
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "命令已取消，未执行。")

    environment = os.environ.copy()
    for name in ("DEEPSEEK_API_KEY", "BASH_ENV", "ENV"):
        environment.pop(name, None)
    try:
        process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-c", command],
            cwd=Path(workspace), env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, bufsize=0,
        )
    except (OSError, ValueError):
        raise ToolError("launch_error", "无法启动命令，请检查工作目录与 bash。") from None

    stdout, stderr = _OutputBuffer(), _OutputBuffer()
    timed_out = cancelled = output_error = False
    deadline = monotonic() + timeout
    cleanup_deadline = None
    try:
        with selectors.DefaultSelector() as selector:
            for stream, buffer in ((process.stdout, stdout), (process.stderr, stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, buffer)
            while selector.get_map() or process.poll() is None:
                now = monotonic()
                cancelled = cancelled or (abort is not None and abort.is_set())
                if cleanup_deadline is None and (cancelled or now >= deadline):
                    timed_out = not cancelled
                    _kill_group(process.pid)
                    cleanup_deadline = now + _CLEANUP_TIMEOUT
                limit = cleanup_deadline if cleanup_deadline is not None else deadline
                if now >= limit:
                    break
                for key, _ in selector.select(min(0.05, limit - now)):
                    try:
                        chunk = os.read(key.fd, 16384)
                    except BlockingIOError:
                        continue
                    if chunk:
                        key.data.append(chunk)
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
    except (OSError, ValueError):
        output_error = True
    finally:
        _kill_group(process.pid)
        process.stdout.close()
        process.stderr.close()
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass

    exit_code = process.returncode
    success = exit_code == 0 and not timed_out and not cancelled and not output_error
    if output_error:
        message = "读取命令输出失败，进程已停止。"
    elif cancelled:
        message = "命令已取消，已终止同组进程。"
    elif timed_out:
        message = f"命令超过 {timeout} 秒，已终止同组进程。"
    else:
        message = f"命令已结束，退出码为 {exit_code}。"
    result = {
        "status": "success" if success else "error",
        "stdout": stdout.text(), "stderr": stderr.text(),
        "exit_code": exit_code, "timed_out": timed_out, "cancelled": cancelled,
        "stdout_truncated": stdout.truncated, "stderr_truncated": stderr.truncated,
        "stdout_bytes": stdout.total, "stderr_bytes": stderr.total,
        "message": message,
    }
    if output_error:
        result["code"] = "command_error"
    return result


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _OutputBuffer:
    def __init__(self):
        self.total = 0
        self.truncated = False
        self.head = b""
        self.tail = b""

    def append(self, chunk):
        self.total += len(chunk)
        if not self.truncated and self.total <= MAX_OUTPUT_BYTES:
            self.head += chunk
            return
        tail_limit = MAX_OUTPUT_BYTES - MAX_OUTPUT_BYTES // 2 - len(_TRUNCATION_MARKER)
        if not self.truncated:
            combined = self.head + chunk
            self.head = combined[:MAX_OUTPUT_BYTES // 2]
            self.tail = combined[-tail_limit:]
            self.truncated = True
        else:
            self.tail = (self.tail + chunk)[-tail_limit:]

    def text(self):
        content = self.head
        if self.truncated:
            content += _TRUNCATION_MARKER + self.tail
        return content.decode("utf-8", errors="replace")
