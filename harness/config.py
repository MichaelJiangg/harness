"""读取并验证项目配置；密钥仍由环境变量或 .env 单独提供。"""

from copy import deepcopy
from functools import lru_cache
import math
import os
from pathlib import Path
import shlex
import tomllib
from urllib.parse import urlsplit

from .permissions import parse_rules

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
PROJECT_FILE = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(minimum):
    return lambda value: type(value) is int and value >= minimum


def _number(minimum, *, exclusive=False):
    def check(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value) and (value > minimum if exclusive else value >= minimum)
        except OverflowError:
            return False
    return check


def _endpoint(value):
    if not _text(value) or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                               for character in value):
        return False
    try:
        parts = urlsplit(value)
        return (parts.scheme == "https" and bool(parts.hostname)
                and parts.username is None and parts.password is None
                and (parts.port is None or parts.port > 0))
    except ValueError:
        return False


def _tool_names(value):
    return isinstance(value, list) and all(
        _text(name) and not any(character.isspace() for character in name)
        for name in value
    )


def _auto_directories(value):
    return isinstance(value, list) and all(
        isinstance(path, str) and path and "\x00" not in path and not Path(path).is_absolute()
        and ".." not in Path(path).parts
        for path in value
    )


def _permission_rules(value):
    if not isinstance(value, list):
        return False
    try:
        parse_rules(value)
    except ValueError:
        return False
    return True


def _weekdays(value):
    return isinstance(value, list) and all(type(day) is int and 0 <= day <= 6 for day in value)


def _hours(value):
    return isinstance(value, list) and all(
        isinstance(period, list) and len(period) == 2
        and all(type(hour) is int for hour in period)
        and 0 <= period[0] < period[1] <= 24
        for period in value
    )


_RATES = {
    "input_hit_per_million": _number(0),
    "input_miss_per_million": _number(0),
    "output_per_million": _number(0),
}
_SCHEMA = {
    "model": {"name": _text, "endpoint": _endpoint, "request_timeout": _number(0, exclusive=True)},
    "display": {"character_delay": _number(0)},
    "engine": {
        "max_requests": _integer(1), "max_retries": _integer(0),
        "retry_initial_delay": _number(0), "retry_backoff": _number(1),
    },
    "background": {
        "max_concurrent": _integer(1), "default_timeout": _integer(1),
    },
    "security": {
        "mode": lambda value: value in {"ask", "auto"},
        "auto_directories": _auto_directories,
    },
    "swarm": {
        "max_requests": _integer(1), "max_role_requests": _integer(1),
    },
    "context": {
        "max_chars": _integer(1), "summary_chars": _integer(1),
        "keep_recent_turns": _integer(0), "max_compactions": _integer(0),
        "tool_result_chars": _integer(1),
    },
    "tools": {
        "file_max_bytes": _integer(1),
        "read_file": {"page_lines": _integer(1)},
        "bash": {
            "default_timeout": _integer(1), "max_timeout": _integer(1),
            "max_output_bytes": _integer(128),
        },
        "grep": {
            "default_max_results": _integer(1), "max_results": _integer(1),
            "max_line_chars": _integer(1), "max_result_chars": _integer(1000),
        },
    },
    "pricing": {
        **_RATES, "peak": _RATES, "currency": _text, "source": _text,
        "checked_at": _text, "peak_weekdays": _weekdays, "peak_hours_utc": _hours,
    },
    "permissions": {
        "allow": _tool_names, "ask": _tool_names, "deny": _tool_names,
        "rules": _permission_rules,
    },
}


def _validate_table(value, schema, location="tool.harness"):
    if not isinstance(value, dict) or value.keys() != schema.keys():
        raise ValueError(f"pyproject.toml 的 {location} 必须完整填写规定字段，不能包含未知字段。")
    for name, rule in schema.items():
        field = f"{location}.{name}"
        if isinstance(rule, dict):
            _validate_table(value[name], rule, field)
        elif not rule(value[name]):
            raise ValueError(f"pyproject.toml 的 {field} 类型或取值无效。")


def load_settings(path=None):
    """读取指定 TOML 的 tool.harness；默认路径不随工作目录改变。"""
    path = PROJECT_FILE if path is None else Path(path)
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except FileNotFoundError:
        raise ValueError("未找到项目 pyproject.toml，请恢复配置文件。") from None
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError("无法读取 pyproject.toml，请检查文件权限、UTF-8 编码和 TOML 格式。") from None
    tool = document.get("tool")
    settings = tool.get("harness") if isinstance(tool, dict) else None
    _validate_table(settings, _SCHEMA)
    context, background, security, swarm, bash, grep = (
        settings["context"], settings["background"], settings["security"], settings["swarm"],
        settings["tools"]["bash"], settings["tools"]["grep"],
    )
    if security["mode"] == "auto" and not security["auto_directories"]:
        # 空列表表示信任会话启动目录；启动时允许，运行时按当前工作区解析。
        pass
    if background["default_timeout"] > 300:
        raise ValueError("pyproject.toml 的后台任务默认超时不能超过 300 秒。")
    if swarm["max_role_requests"] > swarm["max_requests"]:
        raise ValueError("pyproject.toml 的 Swarm 单角色请求上限不能超过团队总请求上限。")
    if bash["default_timeout"] > bash["max_timeout"]:
        raise ValueError("pyproject.toml 的 Bash 默认超时不能超过最大超时。")
    if grep["default_max_results"] > grep["max_results"]:
        raise ValueError("pyproject.toml 的搜索默认条数不能超过最大条数。")
    if context["summary_chars"] >= context["max_chars"]:
        raise ValueError("pyproject.toml 的摘要长度必须小于上下文长度。")
    if grep["max_result_chars"] > context["tool_result_chars"] - 500:
        raise ValueError("pyproject.toml 的搜索结果长度须比工具结果长度至少少 500 字符。")
    if 6 * grep["max_line_chars"] + 400 > grep["max_result_chars"]:
        raise ValueError("pyproject.toml 的搜索结果预算须至少为单行片段长度的 6 倍加 400 字符。")
    return settings


@lru_cache(maxsize=1)
def _cached_settings():
    return load_settings()


def get_settings():
    """返回启动配置快照的副本；当前进程不热加载磁盘上的修改。"""
    return deepcopy(_cached_settings())


def load_api_key(*, env_file=None, environ=None):
    environ = os.environ if environ is None else environ
    if "DEEPSEEK_API_KEY" in environ:
        return environ["DEEPSEEK_API_KEY"]

    path = ENV_FILE if env_file is None else Path(env_file)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ValueError("无法读取项目 .env，请检查文件权限和 UTF-8 编码。") from None

    api_key = None
    for number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if name.strip() != "DEEPSEEK_API_KEY":
            continue
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
            if not separator or len(parts) > 1:
                raise ValueError
        except ValueError:
            raise ValueError(f".env 第 {number} 行的 DEEPSEEK_API_KEY 格式无效，请检查引号与赋值格式。") from None
        api_key = parts[0] if parts else ""
    return api_key
