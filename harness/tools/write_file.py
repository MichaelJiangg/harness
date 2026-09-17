"""本地确认后，在工作目录内创建或覆盖 UTF-8 文本文件。"""

from pathlib import Path
from stat import S_IMODE, S_ISREG
from tempfile import NamedTemporaryFile

from .definition import ToolDefinition
from .executor import ToolError


MAX_FILE_BYTES = 1024 * 1024


DEFINITION = ToolDefinition(
    name="write_file",
    description=(
        "在会话启动目录内创建或覆盖 UTF-8 普通文本文件。"
        "提供 path 和完整 content，已有文件将覆盖全文，缺失的父目录会自动创建。"
        "每次调用都必须先向用户展示完整内容并取得本地确认，模型不能代替用户批准。"
        "用户拒绝时不得自行重试写入，需等待用户新的明确要求。"
        "相对路径以启动目录为基准，绝对路径也必须位于该目录内。"
        "不允许越界、.env* 或 .git 路径，内容最多 1 MiB。"
        "失败时返回错误代码和说明。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "待写入的 UTF-8 文本文件的绝对或相对路径。"},
            "content": {"type": "string", "description": "待写入的完整文本；允许空字符串，已有文件将被全文覆盖。"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    requires_confirmation=True,
)


def execute(arguments, workspace):
    if not arguments["path"].strip() or "\x00" in arguments["path"]:
        raise ToolError("invalid_arguments", "path 必须是非空文件路径，且不能含空字符。")
    if "\x00" in arguments["content"]:
        raise ToolError("invalid_encoding", "内容包含空字符，仅支持 UTF-8 文本。")

    temporary = None
    try:
        data = arguments["content"].encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise ToolError("file_too_large", "内容超过 1 MiB 写入上限，请先缩小内容。")

        root = Path(workspace).resolve()
        candidate = root / arguments["path"]
        if _protected(candidate):
            raise ToolError("access_denied", "不允许写入 .env* 或 .git 路径。")
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ToolError("access_denied", "只能写入启动目录及其子目录内的文件。")
        if _protected(path):
            raise ToolError("access_denied", "不允许写入 .env* 或 .git 路径。")

        try:
            metadata = path.stat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and not S_ISREG(metadata.st_mode):
            raise ToolError("not_a_file", "路径不是普通文件，请提供文本文件路径。")

        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".harness-write-", delete=False) as target:
            temporary = Path(target.name)
            target.write(data)
        if metadata is not None:
            temporary.chmod(S_IMODE(metadata.st_mode))
        temporary.replace(path)
    except (IsADirectoryError, NotADirectoryError, FileExistsError):
        raise ToolError("not_a_file", "路径或父目录不是预期的文件类型，请检查 path。") from None
    except PermissionError:
        raise ToolError("permission_denied", "没有写入该文件或创建父目录的权限。") from None
    except UnicodeEncodeError:
        raise ToolError("invalid_encoding", "内容无法编码为有效的 UTF-8 文本。") from None
    except (OSError, RuntimeError, ValueError):
        raise ToolError("write_error", "无法写入文件，请检查路径和文件状态。") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    return {
        "path": str(path.relative_to(root)),
        "bytes_written": len(data),
        "message": f"已写入文件（{len(data)} 字节）。",
    }


def _protected(path):
    return any(part.casefold() == ".git" or part.casefold().startswith(".env") for part in path.parts)
