"""替换项目根目录的 HARNESS.md 全部内容。"""

from ..notes import NotesError, NotesStore
from .definition import ToolDefinition
from .executor import ToolError


DEFINITION = ToolDefinition(
    name="notes_replace",
    description=(
        "用完整 content 替换会话工作区根目录的 HARNESS.md 全部内容。"
        "仅在需要重写冲突、删除过时信息或用户明确要求整理笔记时使用；"
        "普通新增信息应优先使用 notes_append。允许空字符串清空文件。"
        "默认需要本地用户确认；不能修改项目根目录以外的文件。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "替换后的完整 UTF-8 Markdown 内容；空字符串表示清空。",
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace):
    try:
        written = NotesStore(workspace).replace(arguments["content"])
    except NotesError as error:
        raise ToolError("notes_error", str(error)) from None
    return {
        "path": "HARNESS.md",
        "bytes_written": written,
        "message": f"已替换项目笔记（{written} 字节）。",
    }
