"""按依赖顺序检查模块并生成启动状态报告。"""

from dataclasses import dataclass
import sys

from .config import get_settings
from .hooks import HookManager
from .mcp import MCPManager, load_server_configs
from .memory import MemoryStore
from .tools import get_tool_definitions


@dataclass(frozen=True)
class StartupItem:
    name: str
    status: str
    detail: str = ""
    error: str = ""
    required: bool = False


class StartupFailure(RuntimeError):
    pass


def check_startup(*, client, check_mcp=False):
    items = []
    warnings = 0

    def add(name, detail="", *, required=False):
        items.append(StartupItem(name=name, status="OK", detail=detail, required=required))

    def warn(name, error, *, required=False):
        nonlocal warnings
        warnings += 1
        items.append(StartupItem(name=name, status="WARN", error=error, required=required))

    add("Config", "loaded and validated", required=True)
    add(
        "Query engine",
        f"provider={getattr(client, 'provider', 'unknown')}",
        required=True,
    )
    try:
        tools = get_tool_definitions()
    except Exception as error:
        raise StartupFailure(f"内置工具初始化失败：{error}") from None
    add("Built-in tools", f"{len(tools)} tools", required=True)

    mcp_configs, source = load_server_configs()
    add("MCP servers", f"{len(mcp_configs)} configured ({source})")
    if check_mcp and mcp_configs:
        manager = MCPManager(mcp_configs, reconnect=False, connect=False)
        try:
            manager.connect_all()
            failed = [
                status for status in manager.status()
                if status["state"] not in {"connected", "connecting"}
            ]
            connected = len(manager.status()) - len(failed)
            if failed:
                warnings += 1
                items[-1] = StartupItem(
                    name="MCP servers",
                    status="WARN",
                    detail=f"{connected}/{len(mcp_configs)} connected",
                    error="; ".join(
                        f"{status['name']}: {status.get('last_error') or status['state']}"
                        for status in failed
                    ),
                )
            else:
                items[-1] = StartupItem(
                    name="MCP servers",
                    status="OK",
                    detail=f"{connected}/{len(mcp_configs)} connected",
                )
        finally:
            manager.close()

    permissions = get_settings()["permissions"]
    rule_count = len(permissions["rules"])
    list_count = len(permissions["allow"]) + len(permissions["ask"]) + len(permissions["deny"])
    add("Permissions", f"{rule_count} rules, {list_count} tool entries", required=True)

    hook_manager = HookManager()
    if hook_manager.load_error:
        warn("Hooks", hook_manager.load_error)
    else:
        add("Hooks", f"{len(hook_manager.hooks)} hooks")

    memory_store = MemoryStore()
    try:
        memory_count = len(memory_store.records())
    except Exception as error:
        warn("Memory", str(error))
    else:
        add("Memory", f"{memory_count} entries")

    if sys.stdin.isatty() and sys.stdout.isatty():
        add("Terminal", "interactive")
    else:
        add("Terminal", "non-interactive")
    return items, warnings


def format_startup_report(items, warnings, *, check_mode=False):
    lines = []
    if check_mode:
        lines.append("Checking configuration...")
    for item in items:
        marker = {
            "OK": "✓" if check_mode else "OK",
            "WARN": "✗" if check_mode else "WARN",
        }.get(item.status, item.status)
        padding = "." * max(1, 20 - len(item.name))
        detail = f" {item.detail}" if item.detail else ""
        lines.append(f"[init] {item.name} {padding} {marker}{detail}")
        if item.error:
            lines.append(f"      └─ {item.error}")
    lines.append(
        f"{'Result: ' if check_mode else 'Harness ready: '}"
        f"{len(items) - warnings}/{len(items)} checks passed, {warnings} warning(s)"
    )
    return "\n".join(lines)


__all__ = [
    "StartupFailure",
    "StartupItem",
    "check_startup",
    "format_startup_report",
]
