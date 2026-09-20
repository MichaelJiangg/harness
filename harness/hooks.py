"""从 .harness/hooks.json 加载用户自定义生命周期钩子。"""

from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import subprocess


HOOKS_FILE = ".harness/hooks.json"
HOOK_EVENTS = {
    "session_start", "session_end", "before_send_message",
    "after_reply", "before_tool", "after_tool",
}
HOOK_TYPES = {"shell", "prompt", "python"}
MAX_HOOKS = 100
MAX_COMMAND_CHARS = 10000
MAX_PROMPT_CHARS = 12000
HOOK_TIMEOUT = 30


@dataclass(frozen=True)
class HookDefinition:
    event: str
    type: str
    name: str
    command: str | None = None
    prompt: str | None = None
    module: str | None = None
    function: str | None = None
    timeout: int = HOOK_TIMEOUT


@dataclass
class HookResult:
    prompts: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class HookManager:
    """加载本地配置并在事件发生时执行钩子。"""

    def __init__(self, workspace=None):
        self.workspace = (Path.cwd() if workspace is None else Path(workspace)).resolve()
        self.path = self.workspace / HOOKS_FILE
        self.hooks = []
        self.load_error = None
        self._load()

    def _load(self):
        try:
            content = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except (OSError, UnicodeError):
            self.load_error = "无法读取 hooks.json。"
            return
        try:
            document = json.loads(content)
            values = document.get("hooks")
        except json.JSONDecodeError:
            self.load_error = "hooks.json 不是有效 JSON。"
            return
        if not isinstance(values, list):
            self.load_error = "hooks.json 的 hooks 必须是数组。"
            return
        if len(values) > MAX_HOOKS:
            self.load_error = f"hooks.json 最多允许 {MAX_HOOKS} 个钩子。"
            return
        hooks = []
        for index, value in enumerate(values):
            try:
                hooks.append(self._parse_hook(value, index))
            except ValueError as error:
                self.load_error = str(error)
                return
        self.hooks = tuple(hooks)

    def _parse_hook(self, value, index):
        if not isinstance(value, dict):
            raise ValueError(f"hooks[{index}] 必须是对象。")
        event = value.get("event")
        kind = value.get("type")
        if event not in HOOK_EVENTS or kind not in HOOK_TYPES:
            raise ValueError(f"hooks[{index}] 的 event 或 type 无效。")
        name = value.get("name") or f"hooks[{index}]"
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError(f"hooks[{index}] 的 name 无效。")
        hook = HookDefinition(
            event=event, type=kind, name=name,
            command=value.get("command"), prompt=value.get("prompt"),
            module=value.get("module"), function=value.get("function"),
            timeout=value.get("timeout", HOOK_TIMEOUT),
        )
        if kind == "shell":
            if (not isinstance(hook.command, str) or not hook.command.strip()
                    or len(hook.command) > MAX_COMMAND_CHARS):
                raise ValueError(f"hooks[{index}] 的 shell command 无效。")
        elif kind == "prompt":
            if (not isinstance(hook.prompt, str) or not hook.prompt.strip()
                    or len(hook.prompt) > MAX_PROMPT_CHARS):
                raise ValueError(f"hooks[{index}] 的 prompt 无效。")
        else:
            if (not isinstance(hook.module, str) or not hook.module.strip()
                    or not isinstance(hook.function, str) or not hook.function.strip()
                    or "\x00" in hook.module or "\x00" in hook.function):
                raise ValueError(f"hooks[{index}] 的 python module/function 无效。")
        if type(hook.timeout) is not int or not 1 <= hook.timeout <= 120:
            raise ValueError(f"hooks[{index}] 的 timeout 必须是 1～120 秒。")
        return hook

    def run(self, event, context=None):
        result = HookResult()
        for hook in self.hooks:
            if hook.event != event:
                continue
            try:
                if hook.type == "shell":
                    result.outputs.append(self._run_shell(hook, context))
                elif hook.type == "prompt":
                    result.prompts.append(hook.prompt)
                else:
                    self._run_python(hook, context, result)
            except Exception as error:
                result.errors.append(f"{hook.name}: {error}")
        return result

    def prompts(self, event, context=None):
        return self.run(event, context).prompts

    def _run_shell(self, hook, context):
        environment = _hook_environment(os.environ, self.workspace, hook.event, context)
        completed = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", hook.command],
            cwd=self.workspace, env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=hook.timeout,
        )
        output = completed.stdout.strip()
        if completed.stderr.strip():
            output = f"{output}\n{completed.stderr.strip()}".strip()
        return f"{hook.name}: {output}" if output else hook.name

    def _run_python(self, hook, context, result):
        path = self.workspace / ".harness" / "hook_functions.py"
        if not path.is_file():
            raise RuntimeError("缺少 .harness/hook_functions.py")
        spec = importlib.util.spec_from_file_location("harness_hook_functions", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 hook_functions.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, hook.function, None)
        if not callable(function):
            raise RuntimeError(f"函数 {hook.function} 不存在")
        value = function(deepcopy(context or {}))
        if isinstance(value, str):
            result.prompts.append(value)
        elif isinstance(value, dict):
            prompt = value.get("prompt")
            prompts = value.get("prompts")
            if isinstance(prompt, str):
                result.prompts.append(prompt)
            if isinstance(prompts, list):
                result.prompts.extend(item for item in prompts if isinstance(item, str))
            message = value.get("message")
            if isinstance(message, str):
                result.outputs.append(f"{hook.name}: {message}")


def _hook_environment(environment, workspace, event, context):
    cleaned = dict(environment)
    for name in (
        "DEEPSEEK_API_KEY", "GLM_API_KEY", "TAVILY_API_KEY", "BASH_ENV", "ENV", "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ):
        cleaned.pop(name, None)
    for name in list(cleaned):
        if name.startswith("BASH_FUNC_"):
            cleaned.pop(name, None)
    cleaned.update({
        "HARNESS_EVENT": event,
        "HARNESS_WORKSPACE": str(workspace),
        "HARNESS_TOOL": str((context or {}).get("tool", "")),
    })
    return cleaned
