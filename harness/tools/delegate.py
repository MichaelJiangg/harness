"""委托独立子查询，运行入口仅由执行器在权限检查后注入。"""

from .definition import ToolDefinition
from .executor import ToolError


DEFINITION = ToolDefinition(
    name="delegate",
    description=(
        "将一个明确的子任务交给独立 AI 查询，等待其完成并返回报告。"
        "用于目录级、多文件、代码质量、跨文件对比等可能读取大量文件的分析任务；"
        "主 AI 应在这些任务开始时直接调用，不要先执行目录清点或行数统计；"
        "把完整背景和期望结果写入 task。"
        "description 是短标题，task 必须包含子任务所需背景和具体要求；"
        "子 AI 不会自动看到当前对话历史。"
        "子 AI 只继承当前开放的工具，并沿用工作目录、权限和本地确认规则，不能继续委托。"
        "不要用委托绕过权限限制；子查询的用量计入当前会话。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {"type": "string", "minLength": 1, "description": "子任务的简短标题。"},
            "task": {"type": "string", "minLength": 1, "description": "完整任务说明，包含必要背景与期望结果。"},
        },
        "required": ["description", "task"],
        "additionalProperties": False,
    },
    supports_cancellation=True,
)


def execute(arguments, workspace, *, runner=None, abort=None):
    for name in ("description", "task"):
        value = arguments.get(name)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ToolError("invalid_arguments", f"{name} 必须是非空文本，且不能含空字符。")
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "子任务已取消，未启动。")
    if runner is None:
        raise ToolError("delegation_unavailable", "当前没有可用的查询上下文，无法启动子任务。")
    return runner(description=arguments["description"], task=arguments["task"])
