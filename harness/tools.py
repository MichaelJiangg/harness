"""Tool definitions and side-effect-free placeholders."""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件的占位工具；当前不会读取文件，只返回尚未实现的状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径。"}
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行命令的占位工具；当前不会运行命令，只返回尚未实现的状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "待执行的命令。"}
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]

_REQUIRED_PARAMETERS = {"read_file": "path", "run_command": "command"}


def execute_tool(name, arguments):
    """Validate parsed arguments and return a placeholder without running tools."""
    def error(message):
        return {
            "status": "error",
            "executed": False,
            "tool": name if isinstance(name, str) else None,
            "message": message,
        }

    if not isinstance(name, str) or name not in _REQUIRED_PARAMETERS:
        return error("未知工具。")

    parameter = _REQUIRED_PARAMETERS[name]
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {parameter}
        or not isinstance(arguments[parameter], str)
    ):
        return error(f"工具参数必须仅包含字符串字段 {parameter}。")

    return {
        "status": "not_implemented",
        "executed": False,
        "tool": name,
        "arguments": arguments,
        "message": "工具尚未实现，未执行任何操作。",
    }
