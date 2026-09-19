"""向项目根目录的 HARNESS.md 追加长期知识。"""

from ..notes import NotesError, NotesStore
from .definition import ToolDefinition
from .executor import ToolError


DEFINITION = ToolDefinition(
    name="notes_append",
    description=(
        "向会话工作区根目录的 HARNESS.md 追加 Markdown 项目长期笔记。"
        "当用户明确说「记住」或对话中产生技术栈、编码约定、架构决定、已知问题、"
        "当前进展等长期有效信息时使用；不要记录普通寒暄或临时任务过程。"
        "content 是需要追加的 Markdown 片段，通常使用标题或无序列表。"
        "文件超过 65536 字节时追加会失败；不能修改项目根目录以外的文件。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "description": "要追加到 HARNESS.md 的 Markdown 内容。",
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace):
    try:
        appended = NotesStore(workspace).append(arguments["content"])
    except NotesError as error:
        raise ToolError("notes_error", str(error)) from None
    return {
        "path": "HARNESS.md",
        "bytes_appended": appended,
        "message": f"已追加项目笔记（{appended} 字节）。",
    }
