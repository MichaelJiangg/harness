"""按行或行内游标读取有预算上限的 UTF-8 页面，保留原始内容。"""

import json
from pathlib import Path
from stat import S_ISREG

from ..config import get_settings
from .definition import ToolDefinition
from .executor import ToolError


MAX_FILE_BYTES = get_settings()["tools"]["file_max_bytes"]
DEFAULT_PAGE_LINES = get_settings()["tools"]["read_file"]["page_lines"]


DEFINITION = ToolDefinition(
    name="read_file",
    description=(
        "分页读取会话启动目录内的 UTF-8 普通文本文件，返回 path、content 和下一页游标。"
        f"offset 为从 0 开始的行号，column 为该行从 0 开始的字符位置，默认均为 0；limit 默认每页最多 {DEFAULT_PAGE_LINES} 行。"
        "页面还受工具结果预算限制，超长单行通过 column 续读，不省略中间内容。"
        "需要全文时，在 eof 为 false 时按 next_offset 和 next_column 继续，直到 eof 为 true；"
        "只需要指定行段时读取目标范围即可，不必读完整个文件。"
        "相对路径以启动目录为基准，绝对路径也必须位于该目录内。"
        f"不允许越界、.env*、.git 或 .harness 路径，整文件最多 {MAX_FILE_BYTES} 字节。"
        "失败时返回错误代码和说明，可据此修正参数后重试。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "待读取的 UTF-8 文本文件的绝对或相对路径。"},
            "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "起始行号，从 0 开始。"},
            "column": {"type": "integer", "minimum": 0, "default": 0,
                       "description": "起始行内的 Unicode 字符位置，从 0 开始，包含原始换行字符；续读时使用 next_column。"},
            "limit": {"type": "integer", "minimum": 1, "default": DEFAULT_PAGE_LINES,
                      "description": f"本页最多读取的原始行数，默认 {DEFAULT_PAGE_LINES}；实际页长同时受结果预算限制。"},
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

    return _page(content, str(path.relative_to(root)), arguments)


def _page(content, path, arguments):
    # 查询上下文在工具执行时才存在；独立调用工具则使用配置中的默认预算。
    from ..orchestration import get_current_query

    state = get_current_query()
    budget = state.tool_result_limit if state is not None else get_settings()["context"]["tool_result_chars"]
    if type(budget) is not int or budget <= 0:
        raise ToolError("page_budget_too_small", "工具结果预算无效，无法返回文件页。")
    offset, column = arguments.get("offset", 0), arguments.get("column", 0)
    limit = arguments.get("limit", DEFAULT_PAGE_LINES)
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    if column and (offset >= total_lines or column >= len(lines[offset])):
        raise ToolError("invalid_arguments", "column 超出起始行范围；到达文件末尾时 column 必须为 0。")

    def result(text, next_offset, next_column):
        return {
            "path": path, "content": text,
            "message": f"已读取文件（{len(text)} 字符）。",
            "offset": offset, "column": column,
            "next_offset": next_offset, "next_column": next_column,
            "eof": next_offset >= total_lines, "total_lines": total_lines,
        }

    def fits(page):
        # 引擎序列化使用相同 JSON 选项，需提前算入执行器稍后补上的字段。
        complete = {**page, "status": "success", "executed": True, "tool": "read_file"}
        return len(json.dumps(complete, ensure_ascii=False)) <= budget

    if offset >= total_lines:
        page = result("", offset, 0)
        if fits(page):
            return page
        raise ToolError("page_budget_too_small", "工具结果预算不足，无法容纳文件页信息。")

    first = lines[offset][column:]

    def whole_lines(count):
        return result(first + "".join(lines[offset + 1:offset + count]), offset + count, 0)

    def longest_prefix(length, build):
        low, high = 0, length
        while low < high:
            middle = (low + high + 1) // 2
            if fits(build(middle)):
                low = middle
            else:
                high = middle - 1
        return low

    # 优先返回完整行；后续行装不下时留到下一页，不在普通页尾切断它。
    count = longest_prefix(min(limit, total_lines - offset), whole_lines)
    if count:
        return whole_lines(count)

    # 连第一行的剩余内容也装不下，才在该行内部按字符推进游标。
    def characters(count):
        return result(first[:count], offset, column + count)

    count = longest_prefix(len(first) - 1, characters)
    if not count:
        raise ToolError("page_budget_too_small", "工具结果预算不足，无法返回内容及前进游标；请提高预算。")
    return characters(count)


def _protected(path):
    return any(part.casefold() in {".git", ".harness"} or part.casefold().startswith(".env") for part in path.parts)
