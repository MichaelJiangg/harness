"""读取工作目录内的 UTF-8 文本，支持按行选取并保留原始换行。"""

from pathlib import Path
from stat import S_ISREG

from ..config import get_settings
from .definition import ToolDefinition
from .executor import ToolError


MAX_FILE_BYTES = get_settings()["tools"]["file_max_bytes"]


DEFINITION = ToolDefinition(
    name="read_file",
    description=(
        "读取会话启动目录内的 UTF-8 普通文本文件，返回 path 和 content。"
        "可通过 offset（从 0 开始）和 limit 按行读取，省略 limit 则读取剩余行。"
        "相对路径以启动目录为基准，绝对路径也必须位于该目录内。"
        f"不允许越界、.env*、.git 或 .harness 路径，整文件最多 {MAX_FILE_BYTES} 字节；长结果可能被截断。"
        "失败时返回错误代码和说明，可据此修正参数后重试。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "待读取的 UTF-8 文本文件的绝对或相对路径。"},
            "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "起始行号，从 0 开始。"},
            "limit": {"type": "integer", "minimum": 1, "description": "最多读取的行数；省略时读取剩余行。"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace):
    if not arguments["path"].strip() or "\x00" in arguments["path"]:
        raise ToolError("invalid_arguments", "path 必须是非空文件路径，且不能含空字符。")

    try:
        root = Path(workspace).resolve()
        candidate = root / arguments["path"]
        if _protected(candidate):
            raise ToolError("access_denied", "不允许读取 .env*、.git 或 .harness 路径。")
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ToolError("access_denied", "只能读取启动目录及其子目录内的文件。")
        if _protected(path):
            raise ToolError("access_denied", "不允许读取 .env*、.git 或 .harness 路径。")

        metadata = path.stat()
        if not S_ISREG(metadata.st_mode):
            raise ToolError("not_a_file", "路径不是普通文件，请提供文本文件路径。")
        if metadata.st_size > MAX_FILE_BYTES:
            raise ToolError("file_too_large", f"文件超过 {MAX_FILE_BYTES} 字节读取上限，请先缩小文件。")
        with path.open("rb") as source:
            data = source.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ToolError("file_too_large", f"文件超过 {MAX_FILE_BYTES} 字节读取上限，请先缩小文件。")
        content = data.decode("utf-8")
        if "\x00" in content:
            raise ToolError("invalid_encoding", "文件包含二进制内容，仅支持 UTF-8 文本。")
    except FileNotFoundError:
        raise ToolError("not_found", "文件不存在，请检查 path 后重试。") from None
    except (IsADirectoryError, NotADirectoryError):
        raise ToolError("not_a_file", "路径不是普通文件，请提供文本文件路径。") from None
    except PermissionError:
        raise ToolError("permission_denied", "没有读取该文件的权限。") from None
    except UnicodeDecodeError:
        raise ToolError("invalid_encoding", "文件不是有效的 UTF-8 文本。") from None
    except (OSError, RuntimeError, ValueError):
        raise ToolError("read_error", "无法读取文件，请检查路径和文件状态。") from None

    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    end = None if limit is None else offset + limit
    content = "".join(content.splitlines(keepends=True)[offset:end])
    return {
        "path": str(path.relative_to(root)),
        "content": content,
        "message": f"已读取文件（{len(content)} 字符）。",
    }


def _protected(path):
    return any(part.casefold() in {".git", ".harness"} or part.casefold().startswith(".env") for part in path.parts)
