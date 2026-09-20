"""MCP stdio 客户端：加载配置、维护连接、发现工具并转发调用。"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from threading import Event, Lock, RLock, Thread
from time import monotonic

from .config import get_settings, load_env_key
from .tools.definition import ToolDefinition


PROTOCOL_VERSION = "2024-11-05"
REQUEST_TIMEOUT = 120
MAX_TOOLS_PER_SERVER = 200
MAX_SERVERS = 50
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
_SECRET_ENV_NAMES = {"DEEPSEEK_API_KEY", "TAVILY_API_KEY"}
_UNSAFE_ENV_NAMES = {"BASH_ENV", "ENV"}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]")


class MCPError(RuntimeError):
    pass


class MCPConfigError(ValueError):
    pass


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...]
    env: tuple[str, ...] = ()


@dataclass
class MCPServerStatus:
    name: str
    state: str = "pending"
    tool_count: int = 0
    started_at: float | None = None
    last_error: str | None = None
    attempts: int = 0

    def as_dict(self):
        uptime = (
            max(0.0, monotonic() - self.started_at)
            if self.started_at is not None else None
        )
        return {
            "name": self.name,
            "state": self.state,
            "tool_count": self.tool_count,
            "uptime_seconds": uptime,
            "last_error": self.last_error,
            "attempts": self.attempts,
        }


class MCPServer:
    def __init__(self, config, *, stop_event=None):
        self.config = config
        self.process = None
        self._next_id = 0
        self._responses = {}
        self._lock = Lock()
        self._reader = None
        self._stderr_reader = None
        self._stopped = False
        self._unmatched = []
        self._stop_event = stop_event if stop_event is not None else Event()
        self._exit_event = Event()

    @property
    def alive(self):
        return (
            self.process is not None
            and self.process.poll() is None
            and not self._stopped
        )

    def wait_for_exit(self, timeout=None):
        return self._exit_event.wait(timeout)

    def start(self):
        environment = {
            key: value for key, value in os.environ.items()
            if key not in _SECRET_ENV_NAMES and key not in _UNSAFE_ENV_NAMES
            and not key.startswith(("BASH_FUNC_", "LD_", "DYLD_"))
        }
        for item in self.config.env:
            name, separator, raw_value = item.partition("=")
            if separator:
                environment[name] = _resolve_env_value(self.config, raw_value)
        try:
            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
        except OSError as error:
            raise MCPError(f"无法启动 MCP 服务器 {self.config.name}：{error}") from None
        self._reader = Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()

    def initialize(self):
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "harness", "version": "0.9.1"},
        })
        self._notify("notifications/initialized", {})
        return result

    def list_tools(self):
        return self._request("tools/list", {})

    def call_tool(self, name, arguments):
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        self._stopped = True
        if self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                process_pid = getattr(self.process, "pid", None)
                if os.name == "posix" and isinstance(process_pid, int):
                    os.killpg(process_pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=2)
            except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    if os.name == "posix" and isinstance(process_pid, int):
                        os.killpg(process_pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if self._reader is not None:
            self._reader.join(timeout=1)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=1)

    def _request(self, method, params):
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            for index, payload in enumerate(self._unmatched):
                if payload.get("id") == request_id:
                    self._unmatched.pop(index)
                    return payload.get("result", {}) if isinstance(payload, dict) else {}
            event = Event()
            self._responses[request_id] = (event, None)
            self._write({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })
        deadline = monotonic() + REQUEST_TIMEOUT
        while not event.is_set():
            if self._stop_event.is_set():
                with self._lock:
                    self._responses.pop(request_id, None)
                raise MCPError("MCP 管理器已停止。")
            remaining = deadline - monotonic()
            if remaining <= 0:
                with self._lock:
                    self._responses.pop(request_id, None)
                raise MCPError(f"MCP 请求超时：{method}")
            event.wait(min(0.1, remaining))
        with self._lock:
            _, payload = self._responses.pop(request_id)
        if payload is None:
            raise MCPError("MCP 响应无效。")
        if isinstance(payload, dict) and payload.get("error"):
            raise MCPError(str(payload["error"]))
        return payload.get("result", {}) if isinstance(payload, dict) else {}

    def _notify(self, method, params):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, payload):
        if self.process is None or self.process.stdin is None:
            raise MCPError("MCP 服务器尚未启动。")
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except OSError:
            raise MCPError("无法向 MCP 服务器发送消息。") from None

    def _read_loop(self):
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                if self._stopped:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                request_id = payload.get("id")
                if request_id is not None:
                    with self._lock:
                        pending = self._responses.get(request_id)
                        if pending is not None:
                            event, _ = pending
                            self._responses[request_id] = (event, payload)
                            event.set()
                        else:
                            self._unmatched.append(payload)
        except (OSError, ValueError):
            pass
        finally:
            if not self._stopped:
                self._exit_event.set()

    def _drain_stderr(self):
        process = self.process
        if process is None or process.stderr is None:
            return
        try:
            for _ in iter(lambda: process.stderr.read(4096), ""):
                if self._stopped:
                    break
        except (OSError, ValueError):
            pass


def _resolve_env_value(config, raw_value):
    reference = _ENV_REFERENCE.fullmatch(raw_value)
    if reference is None:
        return raw_value
    variable = reference.group(1)
    try:
        value = load_env_key(variable)
    except ValueError:
        raise MCPError(
            f"MCP 服务器 {config.name} 无法读取环境变量 {variable}，"
            "请检查 .env 文件权限和 UTF-8 编码。"
        ) from None
    if value is None or not value.strip():
        raise MCPError(
            f"MCP 服务器 {config.name} 需要的环境变量 {variable} 未配置或为空，"
            "请在项目 .env 或当前进程中设置。"
        )
    return value


class MCPManager:
    def __init__(self, configs, *, reconnect=True, on_change=None, connect=True):
        self._configs = tuple(configs)
        self._config_by_name = {config.name: config for config in self._configs}
        if len(self._config_by_name) != len(self._configs):
            raise ValueError("MCP 服务器名称不能重复。")
        self._servers = {}
        self._discovered = {}
        self._tool_map = {}
        self._connect_locks = {
            config.name: Lock() for config in self._configs
        }
        self._tools_lock = RLock()
        self._status_lock = RLock()
        self._stop_event = Event()
        self._monitors = []
        self.tools = []
        self.load_errors = []
        self.statuses = {
            config.name: MCPServerStatus(config.name) for config in self._configs
        }
        self._on_change = on_change
        if connect:
            self.connect_all()
        if reconnect:
            for config in self._configs:
                monitor = Thread(
                    target=self._monitor, args=(config,),
                    daemon=True, name=f"harness-mcp-{config.name}",
                )
                self._monitors.append(monitor)
                monitor.start()

    def connect_all(self):
        for config in self._configs:
            if self._stop_event.is_set():
                return
            self._connect(config)

    @property
    def servers(self):
        with self._tools_lock:
            return list(self._servers.values())

    def definitions(self):
        return [definition.to_deepseek() for definition in self.tool_definitions()]

    def tool_definitions(self):
        with self._tools_lock:
            return list(self.tools)

    def tool_names(self, name=None):
        with self._tools_lock:
            if name is None:
                return [
                    tool_name
                    for discovered in self._discovered.values()
                    for _, tool_name in discovered
                ]
            return [
                tool_name
                for _, tool_name in self._discovered.get(name, [])
            ]

    def status(self):
        with self._status_lock:
            return [self.statuses[config.name].as_dict() for config in self._configs]

    def execute(self, arguments, workspace, *, internal_name=None):
        with self._tools_lock:
            mapping = self._tool_map.get(internal_name)
        if mapping is None:
            raise MCPError(f"未知 MCP 工具：{internal_name}")
        name, tool_name = mapping
        with self._connect_locks[name]:
            server = self._servers.get(name)
            if server is None or not server.alive:
                if self._stop_event.is_set():
                    raise MCPError(f"MCP 服务器 {name} 已停止。")
                self._connect_locked(self._config_by_name[name], reconnecting=True)
                server = self._servers.get(name)
            if server is None:
                raise MCPError(f"MCP 服务器 {name} 当前不可用。")
        response = server.call_tool(tool_name, arguments)
        if not isinstance(response, dict):
            raise MCPError("MCP 工具返回无效响应。")
        content = response.get("content", [])
        text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
        structured = response.get("structuredContent")
        result = {
            "status": "error" if response.get("isError") else "success",
            "content": text,
            "message": "MCP 工具调用失败。" if response.get("isError") else "MCP 工具调用完成。",
        }
        if isinstance(structured, (dict, list)):
            result["structuredContent"] = structured
        return result

    def close(self):
        self._stop_event.set()
        for monitor in self._monitors:
            monitor.join(timeout=1)
        with self._tools_lock:
            servers = list(self._servers.values())
            self._servers.clear()
        for server in servers:
            server.close()

    def _connect(self, config, *, reconnecting=False):
        with self._connect_locks[config.name]:
            return self._connect_locked(config, reconnecting=reconnecting)

    def _connect_locked(self, config, *, reconnecting=False):
        if self._stop_event.is_set():
            return False
        name = config.name
        with self._status_lock:
            status = self.statuses[name]
            status.attempts += 1
            status.last_error = None
            status.state = "reconnecting" if reconnecting else "connecting"
            status.started_at = None
            self._notify_change(name)
        server = MCPServer(config, stop_event=self._stop_event)
        discovered = []
        try:
            server.start()
            server.initialize()
            response = server.list_tools()
            tools = response.get("tools", []) if isinstance(response, dict) else []
            if not isinstance(tools, list) or len(tools) > MAX_TOOLS_PER_SERVER:
                raise MCPError(f"{name} 返回了无效工具列表。")
            used_names = {
                definition.name
                for other_name, discovered in self._discovered.items()
                if other_name != name
                for definition, _ in discovered
            }
            for index, item in enumerate(tools):
                tool_name = self._tool_name(item, index)
                internal_name = _unique_mcp_name(name, tool_name, used_names)
                definition, _ = self._build_definition(
                    name, item, index, internal_name=internal_name,
                )
                discovered.append((definition, tool_name))
            self._install_server(name, server, discovered)
            with self._status_lock:
                status = self.statuses[name]
                status.state = "connected"
                status.tool_count = len(discovered)
                status.started_at = monotonic()
                status.last_error = None
                self._notify_change(name)
            return True
        except Exception as error:
            server.close()
            with self._status_lock:
                status = self.statuses[name]
                status.state = "reconnecting" if reconnecting else "failed"
                status.started_at = None
                status.last_error = _safe_error(error)
                with self._tools_lock:
                    status.tool_count = len(self._discovered.get(name, []))
                self._notify_change(name)
            if not reconnecting:
                self.load_errors.append(f"{name}: {status.last_error}")
            return False

    def _install_server(self, name, server, discovered):
        with self._tools_lock:
            old = self._servers.get(name)
            self._servers[name] = server
            self._discovered[name] = list(discovered)
            self._rebuild_tools_locked()
        if old is not None and old is not server:
            old.close()

    def _rebuild_tools_locked(self):
        self.tools = []
        self._tool_map = {}
        for config in self._configs:
            server = self._servers.get(config.name)
            if server is None:
                continue
            for definition, tool_name in self._discovered.get(config.name, []):
                self.tools.append(definition)
                self._tool_map[definition.name] = (config.name, tool_name)

    def _monitor(self, config):
        name = config.name
        while not self._stop_event.is_set():
            with self._tools_lock:
                server = self._servers.get(name)
            if server is not None and server.alive:
                if server.wait_for_exit(timeout=0.5):
                    if self._stop_event.wait(self._reconnect_delay(name)):
                        break
                    self._connect(config, reconnecting=True)
                continue
            if self._stop_event.wait(self._reconnect_delay(name)):
                break
            self._connect(config, reconnecting=True)

    def _reconnect_delay(self, name):
        with self._status_lock:
            attempts = self.statuses[name].attempts
        return min(
            RECONNECT_MAX_DELAY,
            RECONNECT_BASE_DELAY * (2 ** min(max(attempts - 1, 0), 5)),
        )

    def _notify_change(self, name):
        if self._on_change is not None:
            with self._status_lock:
                status = self.statuses[name].as_dict()
            self._on_change(name, status)

    @staticmethod
    def _tool_name(item, index):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise MCPError(f"工具 {index} 缺少有效名称。")
        tool_name = item["name"].strip()
        if not tool_name or any(character.isspace() for character in tool_name):
            raise MCPError(f"工具 {index} 名称无效。")
        return tool_name

    @staticmethod
    def _build_definition(server_name, item, index, *, internal_name):
        tool_name = MCPManager._tool_name(item, index)
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            description = f"外部 MCP 工具 {tool_name}"
        schema = item.get("inputSchema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        return ToolDefinition(
            name=internal_name,
            description=f"[MCP {server_name}] {description.strip()}",
            input_schema=schema,
            validate_arguments=False,
        ), tool_name


def load_server_configs(workspace=None):
    """优先读取工作区 .harness/mcp.json，否则回退到 pyproject.toml。"""
    root = Path.cwd() if workspace is None else Path(workspace)
    path = root / ".harness" / "mcp.json"
    if path.exists():
        return _load_json_configs(path), ".harness/mcp.json"
    settings = get_settings()["mcp"]
    configs = [_config_from_toml(item) for item in settings["servers"]]
    return configs, "pyproject.toml"


def _load_json_configs(path):
    try:
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
    except FileNotFoundError:
        raise MCPConfigError("未找到 MCP 配置文件 .harness/mcp.json。") from None
    except (OSError, UnicodeError):
        raise MCPConfigError("无法读取 .harness/mcp.json，请检查文件权限和 UTF-8 编码。") from None
    except json.JSONDecodeError:
        raise MCPConfigError("无法解析 .harness/mcp.json，请检查 JSON 格式。") from None
    if not isinstance(document, dict) or set(document) != {"servers"}:
        raise MCPConfigError(".harness/mcp.json 顶层只能包含 servers 字段。")
    servers = document["servers"]
    if not isinstance(servers, dict) or len(servers) > MAX_SERVERS:
        raise MCPConfigError(
            f".harness/mcp.json 的 servers 必须是对象且最多 {MAX_SERVERS} 台服务器。"
        )
    configs = []
    for name, item in servers.items():
        configs.append(_config_from_json(name, item))
    return configs


def _config_from_json(name, item):
    if (not isinstance(name, str) or not name.strip()
            or "\x00" in name
            or any(character.isspace() or ord(character) < 32 for character in name)):
        raise MCPConfigError("MCP 服务器名称必须是非空且不含空白的字符串。")
    if not isinstance(item, dict) or set(item) - {"command", "args", "env"}:
        raise MCPConfigError(f"MCP 服务器 {name} 包含未知字段。")
    command = item.get("command")
    args = item.get("args", [])
    env = item.get("env", {})
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise MCPConfigError(f"MCP 服务器 {name} 的 command 无效。")
    if not isinstance(args, list) or not all(
        isinstance(argument, str) and "\x00" not in argument for argument in args
    ):
        raise MCPConfigError(f"MCP 服务器 {name} 的 args 必须是字符串数组。")
    if not isinstance(env, dict):
        raise MCPConfigError(f"MCP 服务器 {name} 的 env 必须是对象。")
    env_items = []
    for variable, value in env.items():
        if (not isinstance(variable, str)
                or _ENV_NAME.fullmatch(variable) is None
                or not isinstance(value, str)
                or "\x00" in value):
            raise MCPConfigError(f"MCP 服务器 {name} 的环境变量配置无效。")
        env_items.append(f"{variable}={value}")
    return MCPServerConfig(
        name=name,
        command=command,
        args=tuple(args),
        env=tuple(env_items),
    )


def _config_from_toml(item):
    return MCPServerConfig(
        name=item["name"],
        command=item["command"],
        args=tuple(item.get("args", [])),
        env=tuple(item.get("env", [])),
    )


def _safe_error(error):
    message = str(error).replace("\x00", "")
    return message if len(message) <= 500 else message[:497] + "..."


def _unique_mcp_name(server_name, tool_name, used_names):
    safe_server = _UNSAFE_TOOL_NAME.sub("_", server_name)
    safe_tool = _UNSAFE_TOOL_NAME.sub("_", tool_name)
    base = f"mcp_{safe_server}_{safe_tool}"
    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate
