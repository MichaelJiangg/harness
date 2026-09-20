"""本地斜杠命令的注册表和运行时上下文。"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LocalCommand:
    name: str
    help: str
    handler: Callable[["CommandContext", list[str]], Any]
    available_while_busy: bool = False
    busy_message: str | None = None


class CommandContext:
    """命令处理器可以访问的当前会话快照与回调。"""

    def __init__(
        self,
        *,
        write: Callable[[str], None],
        messages: list[dict],
        active_model: str,
        active_label: str,
        available_models: dict[str, dict],
        switch_model: Callable[[str], bool],
        reset_conversation: Callable[[], None],
        format_history: Callable[[], str],
        format_cost: Callable[[], str],
        format_tools: Callable[[], str],
        compact: Callable[[], None],
        help_text: str,
        busy: bool,
    ):
        self.write = write
        self.messages = messages
        self.active_model = active_model
        self.active_label = active_label
        self.available_models = available_models
        self.switch_model = switch_model
        self.reset_conversation = reset_conversation
        self.format_history = format_history
        self.format_cost = format_cost
        self.format_tools = format_tools
        self.compact = compact
        self.help_text = help_text
        self.busy = busy


_COMMANDS: dict[str, LocalCommand] = {}


def register(name: str, help_text: str, *, available_while_busy=False, busy_message=None):
    def decorator(handler):
        if name in _COMMANDS:
            raise ValueError(f"命令已注册：{name}")
        _COMMANDS[name] = LocalCommand(
            name=name,
            help=help_text,
            handler=handler,
            available_while_busy=available_while_busy,
            busy_message=busy_message,
        )
        return handler

    return decorator


def get_commands():
    return dict(_COMMANDS)
