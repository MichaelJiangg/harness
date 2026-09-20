"""读取并验证项目配置；密钥仍由环境变量或 .env 单独提供。"""

from copy import deepcopy
from functools import lru_cache
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import tomllib
from urllib.parse import urlsplit

from .permissions import parse_rules

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
CONFIG_FILE = Path.cwd() / ".harness" / "config.toml"
LEGACY_PROJECT_FILE = Path(__file__).resolve().parent.parent / "pyproject.toml"


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


def _mcp_servers(value):
    if not isinstance(value, list) or len(value) > 50:
        return False
    names = set()
    for item in value:
        if not isinstance(item, dict):
            return False
        name = item.get("name")
        command = item.get("command")
        args = item.get("args", [])
        env = item.get("env", [])
        if (not _text(name) or any(character.isspace() for character in name)
                or name in names or not _text(command)):
            return False
        if (not isinstance(args, list)
                or not all(isinstance(arg, str) for arg in args)):
            return False
        if not isinstance(env, list) or not all(
            isinstance(entry, str) and entry.partition("=")[0] and "=" in entry
            and "\x00" not in entry
            and not any(character.isspace() or ord(character) < 32
                        for character in entry.partition("=")[0])
            for entry in env
        ):
            return False
        names.add(name)
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
    "provider": lambda value: value in {"auto", "deepseek", "glm"},
    "model": {"name": _text, "endpoint": _endpoint, "request_timeout": _number(0, exclusive=True)},
    "glm": {
        "name": _text,
        "endpoint": _endpoint,
        "request_timeout": _number(0, exclusive=True),
        "reasoning_effort": lambda value: value in {"low", "high", "max"},
        "pricing": {
            **_RATES, "currency": _text, "source": _text, "checked_at": _text,
        },
    },
    "display": {"character_delay": _number(0)},
    "engine": {
        "max_turns": _integer(1), "max_requests": _integer(1),
        "max_retries": _integer(0),
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
    "presets": {
        "project_conventions": lambda value: type(value) is bool,
        "auto_format": lambda value: type(value) is bool,
        "session_memory": lambda value: type(value) is bool,
    },
    "mcp": {
        "enabled": lambda value: type(value) is bool,
        "servers": _mcp_servers,
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

_DEFAULT_SETTINGS = {
    "provider": "auto",
    "model": {
        "name": "deepseek-flash",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "request_timeout": 120,
    },
    "glm": {
        "name": "glm-5.3-flash",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "request_timeout": 120,
        "reasoning_effort": "low",
        "pricing": {
            "input_hit_per_million": 0.0,
            "input_miss_per_million": 0.0,
            "output_per_million": 0.0,
            "currency": "CNY",
            "source": "https://docs.bigmodel.cn/llms-full.txt",
            "checked_at": "2026-09-20",
        },
    },
    "display": {"character_delay": 0.02},
    "engine": {
        "max_turns": 100,
        "max_requests": 20,
        "max_retries": 3,
        "retry_initial_delay": 1.0,
        "retry_backoff": 2.0,
    },
    "background": {"max_concurrent": 5, "default_timeout": 300},
    "security": {"mode": "ask", "auto_directories": []},
    "swarm": {"max_requests": 120, "max_role_requests": 40},
    "presets": {
        "project_conventions": True,
        "auto_format": True,
        "session_memory": True,
    },
    "mcp": {
        "enabled": True,
        "servers": [{
            "name": "tavily",
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.2.22"],
            "env": ["TAVILY_API_KEY=${TAVILY_API_KEY}"],
        }],
    },
    "context": {
        "max_chars": 64000,
        "summary_chars": 2000,
        "keep_recent_turns": 4,
        "max_compactions": 6,
        "tool_result_chars": 12000,
    },
    "tools": {
        "file_max_bytes": 1048576,
        "read_file": {"page_lines": 200},
        "bash": {
            "default_timeout": 30,
            "max_timeout": 120,
            "max_output_bytes": 65536,
        },
        "grep": {
            "default_max_results": 100,
            "max_results": 500,
            "max_line_chars": 500,
            "max_result_chars": 5500,
        },
    },
    "pricing": {
        "input_hit_per_million": 0.003,
        "input_miss_per_million": 0.15,
        "output_per_million": 0.6,
        "currency": "USD",
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
        "checked_at": "2026-09-18",
        "peak_weekdays": [0, 1, 2, 3, 4],
        "peak_hours_utc": [[1, 4], [6, 10]],
        "peak": {
            "input_hit_per_million": 0.006,
            "input_miss_per_million": 0.30,
            "output_per_million": 1.2,
        },
    },
    "permissions": {
        "allow": ["read_file", "grep", "delegate", "notes_append"],
        "ask": ["write_file", "notes_replace", "web_fetch", "web_search"],
        "deny": [],
        "rules": [],
    },
}

_PROCESS_SETTINGS = None


def _validate_table(value, schema, location="config"):
    if not isinstance(value, dict) or value.keys() != schema.keys():
        raise ValueError(f"{location} 必须完整填写规定字段，不能包含未知字段。")
    for name, rule in schema.items():
        field = f"{location}.{name}"
        if isinstance(rule, dict):
            _validate_table(value[name], rule, field)
        elif not rule(value[name]):
            raise ValueError(f"{location}.{field} 类型或取值无效。")


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_aliases(settings):
    engine = settings.setdefault("engine", {})
    if "context_window" in engine:
        settings.setdefault("context", {})["max_chars"] = engine.pop("context_window")
    tools = settings.setdefault("tools", {})
    if "timeout" in tools:
        tools.setdefault("bash", {})["default_timeout"] = tools.pop("timeout")
    return settings


def _parse_int(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数。") from None
    return number


def _parse_servers(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("HARNESS_MCP_SERVERS 必须是 JSON 数组。") from None
    return value


def load_settings(path=None):
    """读取部分 .harness/config.toml；文件不存在时返回空覆盖。"""
    path = CONFIG_FILE if path is None else Path(path)
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError(
            "无法读取 .harness/config.toml，请检查文件权限、UTF-8 编码和 TOML 格式。"
        ) from None
    return _apply_aliases(document)


def resolve_settings(*, config_file=None, environ=None, cli_overrides=None):
    """按 CLI > 环境变量 > 配置文件 > 默认值解析最终配置。"""
    settings = deepcopy(_DEFAULT_SETTINGS)
    file_overrides = load_settings(config_file)
    settings = _deep_merge(settings, file_overrides)
    environ = os.environ if environ is None else environ
    env_overrides = {}
    if environ.get("HARNESS_PROVIDER"):
        env_overrides["provider"] = environ["HARNESS_PROVIDER"]
    if environ.get("HARNESS_MODEL"):
        env_overrides["model"] = {"name": environ["HARNESS_MODEL"]}
    if environ.get("HARNESS_MAX_TURNS") is not None:
        env_overrides["engine"] = {
            "max_turns": _parse_int(environ["HARNESS_MAX_TURNS"], "HARNESS_MAX_TURNS"),
        }
    if environ.get("HARNESS_CONTEXT_WINDOW_CHARS") is not None:
        env_overrides["context"] = {
            "max_chars": _parse_int(
                environ["HARNESS_CONTEXT_WINDOW_CHARS"],
                "HARNESS_CONTEXT_WINDOW_CHARS",
            ),
        }
    if environ.get("HARNESS_TOOL_TIMEOUT") is not None:
        env_overrides["tools"] = {
            "bash": {
                "default_timeout": _parse_int(
                    environ["HARNESS_TOOL_TIMEOUT"], "HARNESS_TOOL_TIMEOUT",
                ),
            },
        }
    if environ.get("HARNESS_MCP_SERVERS") is not None:
        env_overrides["mcp"] = {
            "servers": _parse_servers(environ["HARNESS_MCP_SERVERS"]),
        }
    settings = _deep_merge(settings, env_overrides)
    settings = _deep_merge(settings, _apply_aliases(cli_overrides or {}))
    _validate_table(settings, _SCHEMA, "配置")
    context, background, security, swarm, bash, grep = (
        settings["context"], settings["background"], settings["security"], settings["swarm"],
        settings["tools"]["bash"], settings["tools"]["grep"],
    )
    if security["mode"] == "auto" and not security["auto_directories"]:
        # 空列表表示信任会话启动目录；启动时允许，运行时按当前工作区解析。
        pass
    if background["default_timeout"] > 300:
        raise ValueError("配置的后台任务默认超时不能超过 300 秒。")
    if swarm["max_role_requests"] > swarm["max_requests"]:
        raise ValueError("配置的 Swarm 单角色请求上限不能超过团队总请求上限。")
    if bash["default_timeout"] > bash["max_timeout"]:
        raise ValueError("配置的 Bash 默认超时不能超过最大超时。")
    if grep["default_max_results"] > grep["max_results"]:
        raise ValueError("配置的搜索默认条数不能超过最大条数。")
    if context["summary_chars"] >= context["max_chars"]:
        raise ValueError("配置的摘要长度必须小于上下文长度。")
    if grep["max_result_chars"] > context["tool_result_chars"] - 500:
        raise ValueError("配置的搜索结果长度须比工具结果长度至少少 500 字符。")
    if 6 * grep["max_line_chars"] + 400 > grep["max_result_chars"]:
        raise ValueError("配置的搜索结果预算须至少为单行片段长度的 6 倍加 400 字符。")
    return settings


@lru_cache(maxsize=1)
def _cached_settings():
    return resolve_settings()


def get_settings():
    """返回启动配置快照的副本；当前进程不热加载磁盘上的修改。"""
    if _PROCESS_SETTINGS is not None:
        return deepcopy(_PROCESS_SETTINGS)
    return deepcopy(_cached_settings())


def set_process_settings(settings):
    global _PROCESS_SETTINGS
    _PROCESS_SETTINGS = deepcopy(settings)


def clear_process_settings():
    global _PROCESS_SETTINGS
    _PROCESS_SETTINGS = None
    _cached_settings.cache_clear()


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--provider", choices=["auto", "deepseek", "glm"])
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--tool-timeout", type=int)
    parser.add_argument("--mcp-server", action="append", default=[])
    parser.add_argument("--show-config", action="store_true")
    return parser.parse_args(argv)


def cli_overrides_from_args(args):
    overrides = {}
    if args.model:
        overrides["model"] = {"name": args.model}
    if args.max_turns is not None:
        overrides["engine"] = {"max_turns": args.max_turns}
    if args.context_window is not None:
        overrides["context"] = {"max_chars": args.context_window}
    if args.tool_timeout is not None:
        overrides["tools"] = {"bash": {"default_timeout": args.tool_timeout}}
    if args.mcp_server:
        servers = []
        for item in args.mcp_server:
            name, separator, command = item.partition("=")
            if not separator or not name.strip() or not command.strip():
                raise ValueError("--mcp-server 必须使用 name=command 格式。")
            servers.append({
                "name": name.strip(),
                "command": command.strip(),
                "args": [],
                "env": [],
            })
        overrides["mcp"] = {"servers": servers}
    return overrides


def load_api_key(*, env_file=None, environ=None):
    return load_env_key("DEEPSEEK_API_KEY", env_file=env_file, environ=environ)


def load_tavily_api_key(*, env_file=None, environ=None):
    return load_env_key("TAVILY_API_KEY", env_file=env_file, environ=environ)


def load_glm_api_key(*, env_file=None, environ=None):
    return load_env_key("GLM_API_KEY", env_file=env_file, environ=environ)


def select_model_provider(*, env_file=None, environ=None):
    """按显式配置或 key 可用性选择模型提供商；auto 模式优先 GLM。"""
    requested = load_env_key(
        "HARNESS_PROVIDER", env_file=env_file, environ=environ
    )
    if requested is not None:
        requested = requested.strip().lower()
        if requested not in {"auto", "deepseek", "glm"}:
            raise ValueError("HARNESS_PROVIDER 只支持 auto、deepseek 或 glm。")
    deepseek_key = load_api_key(env_file=env_file, environ=environ)
    glm_key = load_glm_api_key(env_file=env_file, environ=environ)
    if requested in {"deepseek", "glm"}:
        provider = requested
    else:
        provider = "glm" if glm_key and glm_key.strip() else "deepseek"
    key = glm_key if provider == "glm" else deepseek_key
    if not isinstance(key, str) or not key.strip():
        raise ValueError(
            "请在项目 .env 或当前进程中设置 DEEPSEEK_API_KEY 或 GLM_API_KEY。"
        )
    return provider, key


def load_env_key(key_name, *, env_file=None, environ=None):
    environ = os.environ if environ is None else environ
    if key_name in environ:
        return environ[key_name]

    path = ENV_FILE if env_file is None else Path(env_file)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ValueError("无法读取项目 .env，请检查文件权限和 UTF-8 编码。") from None

    secret = None
    for number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if name.strip() != key_name:
            continue
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
            if not separator or len(parts) > 1:
                raise ValueError
        except ValueError:
            raise ValueError(f".env 第 {number} 行的 {key_name} 格式无效，请检查引号与赋值格式。") from None
        secret = parts[0] if parts else ""
    return secret
