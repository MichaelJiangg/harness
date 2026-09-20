"""自动发现 harness/commands/ 下的本地命令。"""

import importlib
from pathlib import Path

from .base import CommandContext, LocalCommand, get_commands


_DISCOVERED = False


def _discover():
    global _DISCOVERED
    if _DISCOVERED:
        return
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.stem.startswith("_") or path.stem in {"base"}:
            continue
        importlib.import_module(f"{__name__}.{path.stem}")
    _DISCOVERED = True


def command_entries():
    _discover()
    return list(get_commands().values())


def resolve_command(name):
    _discover()
    return get_commands().get(name)


__all__ = ["CommandContext", "LocalCommand", "command_entries", "resolve_command"]
