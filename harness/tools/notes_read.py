"""读取项目根目录的 HARNESS.md 长期笔记。"""

from ..notes import NotesError, NotesStore
from .definition import ToolDefinition
from .executor import ToolError


DEFINITION = ToolDefinition(
    name="notes_read",
    description=(
        "读取会话工作区根目录的 HARNESS.md 项目长期笔记。"
        "无需参数，返回完整 UTF-8 Markdown 文本；文件不存在时返回空内容。"
        "笔记用于了解技术栈、编码约定、架构决定、已知问题和当前进展。"
        "需要查看被系统提示截断的完整笔记时调用此工具。"
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)


def execute(arguments, workspace):
    try:
        content = NotesStore(workspace).read()
    except NotesError as error:
        raise ToolError("notes_error", str(error)) from None
    return {
        "path": "HARNESS.md",
        "content": content,
        "message": f"已读取项目笔记（{len(content)} 字符）。",
    }
