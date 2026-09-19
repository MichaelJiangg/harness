"""只保存权限决策所需元信息的本地 JSONL 审计。"""

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from uuid import uuid4


class AuditError(Exception):
    """日志不能安全落盘时，执行器必须停止工具调用。"""


def sanitize(params):
    """按白名单保留元信息，不复制未知字段及敏感正文。"""
    if not isinstance(params, dict):
        return {}
    result = {}
    if isinstance(params.get("path"), str):
        result["path"] = params["path"][:512]
    for key in ("offset", "column", "limit", "timeout", "max_results"):
        value = params.get(key)
        if type(value) is int or type(value) is float and math.isfinite(value):
            result[key] = value
    for key in ("content", "command", "keyword"):
        value = params.get(key)
        if isinstance(value, str):
            result[f"{key}_chars"] = len(value)
    command = params.get("command")
    if isinstance(command, str):
        result["command_sha256"] = hashlib.sha256(
            command.encode("utf-8", errors="replace")
        ).hexdigest()
    return result


def _require_regular_file(info):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AuditError("无法安全写入权限审计日志。")


class PermissionAuditLog:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.session_id = str(uuid4())

    def record(self, tool, params, decision, matched_rule, *, risk,
               confirmation=None, event="decision", call_id=None):
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                "call_id": call_id,
                "source": "tool_executor",
                "tool": tool,
                "params": sanitize(params),
                "decision": decision,
                "matched_rule": matched_rule,
                "risk": risk,
                "confirmation": confirmation,
                "event": event,
            }
            payload = (json.dumps(entry, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
            self._append(payload)
        except (OSError, ValueError, TypeError, OverflowError, AuditError):
            raise AuditError("无法安全写入权限审计日志。") from None

    def _append(self, payload):
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        with ExitStack() as handles:
            root_fd = os.open(self.workspace, directory_flags)
            handles.callback(os.close, root_fd)
            try:
                os.mkdir(".harness", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            directory_fd = os.open(".harness", directory_flags, dir_fd=root_fd)
            handles.callback(os.close, directory_fd)
            os.fchmod(directory_fd, 0o700)
            # 打开前排除设备等特殊文件，打开后再次验证实际文件句柄。
            try:
                info = os.stat("permission.log", dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                _require_regular_file(info)
            file_flags = (os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
                          | os.O_NONBLOCK | os.O_CLOEXEC)
            try:
                # 排他创建区分首次创建竞争；冲突后只打开已有文件，不循环重试。
                file_fd = os.open("permission.log", file_flags | os.O_CREAT | os.O_EXCL,
                                  mode=0o600, dir_fd=directory_fd)
            except FileExistsError:
                file_fd = os.open("permission.log", file_flags, dir_fd=directory_fd)
            handles.callback(os.close, file_fd)
            _require_regular_file(os.fstat(file_fd))
            os.fchmod(file_fd, 0o600)
            # 每次独立打开；文件锁同时保护线程和不同会话的追加记录。
            fcntl.flock(file_fd, fcntl.LOCK_EX)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(file_fd, remaining)
                if written <= 0:
                    raise AuditError("无法安全写入权限审计日志。")
                remaining = remaining[written:]
            os.fsync(file_fd)
