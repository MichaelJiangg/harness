"""把命令或独立分析任务提交到会话级后台管理器。"""

from ..config import get_settings
from .definition import ToolDefinition
from .executor import ToolError


_SETTINGS = get_settings()["background"]
DEFAULT_TIMEOUT = _SETTINGS["default_timeout"]


DEFINITION = ToolDefinition(
    name="background_submit",
    description=(
        "把耗时的测试命令或独立代码分析提交到后台线程，立即返回任务编号，"
        "主 AI 可以继续回答其他问题。description 是短标题；command 用于跑测试或 shell 任务，"
        "task 用于后台分析大量代码，两者必须且只能提供一个。后台任务可跨多轮用户输入保留结果，"
        f"默认 {DEFAULT_TIMEOUT} 秒超时，其中后台 Bash 命令受前台 Bash 最多 120 秒限制；"
        "超时或异常后状态为 FAILED。"
        "提交命令前仍按 Bash 风险检查并由用户确认；提交的任务在后台执行，不再二次确认。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {"type": "string", "minLength": 1, "description": "任务的简短标题。"},
            "command": {
                "type": "string", "minLength": 1,
                "description": "待后台执行的完整 bash 命令，例如运行测试套件。",
            },
            "task": {
                "type": "string", "minLength": 1,
                "description": "交给后台独立分析助手的完整任务说明。",
            },
            "timeout": {
                "type": "integer", "minimum": 1, "maximum": 300, "default": DEFAULT_TIMEOUT,
                "description": f"后台任务超时秒数，范围 1～300，默认 {DEFAULT_TIMEOUT}。",
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace, *, runner=None, abort=None):
    description = arguments["description"]
    command = arguments.get("command")
    task = arguments.get("task")
    timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
    if not isinstance(description, str) or not description.strip() or "\x00" in description:
        raise ToolError("invalid_arguments", "description 必须是非空文本，且不能含空字符。")
    if (command is None) == (task is None):
        raise ToolError("invalid_arguments", "command 与 task 必须且只能提供一个。")
    for name, value in (("command", command), ("task", task)):
        if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
            raise ToolError("invalid_arguments", f"{name} 必须是非空文本，且不能含空字符。")
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ToolError("invalid_arguments", "timeout 必须是 1～300 之间的整数。")
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "查询已停止，未提交后台任务。")
    if runner is None:
        raise ToolError("background_unavailable", "当前会话没有后台任务管理器，无法提交任务。")
    task = runner(description=description, command=command, task=task, timeout=timeout)
    return {
        "status": "success",
        "task": task,
        "message": f"后台任务 #{task['task_id']} 已提交，当前状态为 {task['status']}。",
    }
