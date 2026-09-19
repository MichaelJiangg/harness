"""运行工作区内明确指定的 Node／Python 验证脚本，不接收任意 shell 代码。"""

import shlex
from pathlib import Path

from ..config import get_settings
from .definition import ToolDefinition
from .executor import ToolError


_SETTINGS = get_settings()["tools"]["bash"]
DEFAULT_TIMEOUT = _SETTINGS["default_timeout"]
MAX_TIMEOUT = _SETTINGS["max_timeout"]
_RUNNERS = {"node", "python3", "unittest", "node-test", "npm-test"}
_NODE_SUFFIXES = {".js", ".mjs", ".cjs"}


DEFINITION = ToolDefinition(
    name="run_verify",
    description=(
        "运行工作区内明确存在的 Node 或 Python 验证脚本。target 是相对路径，"
        "runner 可选 node、python3、unittest、node-test 或 npm-test；"
        "不能传入内联代码或任意 shell 参数。"
        "使用本工具代替 node -e、cat > /tmp/... 等临时脚本，确认时只展示目标路径和运行方式。"
        f"timeout 默认 {DEFAULT_TIMEOUT} 秒，最多 {MAX_TIMEOUT} 秒。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "target": {"type": "string", "minLength": 1,
                       "description": "工作区内待验证的文件或目录相对路径。"},
            "runner": {
                "type": "string", "minLength": 1,
                "description": "运行方式：node、python3、unittest、node-test 或 npm-test。",
            },
            "timeout": {
                "type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT,
                "default": DEFAULT_TIMEOUT,
                "description": f"验证超时秒数，范围 1～{MAX_TIMEOUT}。",
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    },
    supports_cancellation=True,
)


def execute(arguments, workspace, *, abort=None):
    from .bash import execute as execute_bash

    target = arguments["target"]
    runner = arguments.get("runner", "node")
    timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
    if not isinstance(target, str) or not target.strip() or "\x00" in target:
        raise ToolError("invalid_arguments", "target 必须是非空文本，且不能含空字符。")
    if runner not in _RUNNERS:
        raise ToolError("invalid_arguments",
                        "runner 必须是 node、python3、unittest、node-test 或 npm-test。")
    if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT:
        raise ToolError("invalid_arguments", f"timeout 必须是 1～{MAX_TIMEOUT} 之间的整数。")

    root = Path(workspace).resolve()
    candidate = (root / target).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError("access_denied", "验证目标必须位于会话工作区内。")
    relative_parts = candidate.relative_to(root).parts
    if not relative_parts or any(
        part in {".git", ".harness"} or part.startswith(".env")
        for part in relative_parts
    ):
        raise ToolError("access_denied", "验证目标包含受保护路径。")

    if runner in {"unittest", "node-test", "npm-test"}:
        if not candidate.is_dir():
            raise ToolError("invalid_arguments", f"{runner} 模式要求 target 指向目录。")
        target = shlex.quote(str(candidate.relative_to(root)))
        if runner == "unittest":
            command = f"python3 -m unittest discover -s {target} -q"
        elif runner == "node-test":
            command = f"node --test {target}/*.test.js"
        else:
            command = f"cd {target} && npm test"
    else:
        if not candidate.is_file():
            raise ToolError("invalid_arguments", "node/python3 模式要求 target 指向文件。")
        suffix = candidate.suffix.lower()
        if runner == "node" and suffix not in _NODE_SUFFIXES:
            raise ToolError("invalid_arguments", "node 模式仅支持 .js、.mjs、.cjs 文件。")
        if runner == "python3" and suffix != ".py":
            raise ToolError("invalid_arguments", "python3 模式仅支持 .py 文件。")
        command = runner + " " + shlex.quote(str(candidate.relative_to(root)))

    return execute_bash({"command": command, "timeout": timeout}, root, abort=abort)
