"""查询会话级后台任务的状态和结果。"""

from .definition import ToolDefinition
from .executor import ToolError


DEFINITION = ToolDefinition(
    name="background_check",
    description=(
        "按任务编号查询后台任务。返回 PENDING、RUNNING、COMPLETED 或 FAILED 状态；"
        "完成或失败后返回结果，任务尚未完成时不要重复空等，可以先处理其他工作。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "minimum": 1, "description": "由 background_submit 返回的任务编号。"},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace, *, runner=None, abort=None):
    task_id = arguments["task_id"]
    if type(task_id) is not int or task_id < 1:
        raise ToolError("invalid_arguments", "task_id 必须是正整数。")
    if runner is None:
        raise ToolError("background_unavailable", "当前会话没有后台任务管理器，无法查询任务。")
    task = runner(task_id)
    if task is None:
        raise ToolError("background_task_not_found", f"未找到后台任务 #{task_id}。")
    return {
        "status": "success",
        "task": task,
        "message": f"后台任务 #{task_id} 当前状态为 {task['status']}。",
    }
