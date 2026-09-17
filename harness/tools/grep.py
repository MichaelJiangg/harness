"""在工作目录内按字面关键词搜索文本，返回有限数量的完整匹配条目。"""

from fnmatch import fnmatchcase
from io import StringIO
import json
import os
from pathlib import Path
from stat import S_ISDIR, S_ISREG

from .definition import ToolDefinition
from .executor import ToolError


DEFAULT_MAX_RESULTS = 100
MAX_RESULTS = 500
MAX_FILE_BYTES = 1024 * 1024
MAX_LINE_CHARS = 500
# 为跳过统计和最终提示留出余量，避免引擎再把匹配条目拆成 JSON 首尾片段。
MAX_RESULT_CHARS = 5500


DEFINITION = ToolDefinition(
    name="grep",
    description=(
        "在会话启动目录内搜索 UTF-8 文本，keyword 是区分大小写的字面关键词，不是正则表达式。"
        "path 可指定文件或递归搜索的目录，默认当前工作目录；glob 按文件名筛选，如 *.py。"
        "返回 matches，每项包含相对 path、从 1 开始的 line_number 和该行 content。"
        "max_results 默认 100，最多 500；总结果和超长行也会截断并标记，可缩小 path 或 glob 后重搜。"
        "关键词本身过长无法在结果中完整保留时，请缩短关键词。"
        "不读取 .env*、.git、越界路径，不跟随目录软链接。"
        "跳过不可读、非 UTF-8、二进制、超过 1 MiB 的文件，并报告跳过数量。"
        "这是只读工具，不需要用户确认。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "minLength": 1, "description": "要在单行中查找的字面关键词，区分大小写。"},
            "path": {"type": "string", "minLength": 1, "default": ".", "description": "搜索目录或单个文件，必须在启动目录内；默认 .。"},
            "glob": {"type": "string", "minLength": 1, "default": "*", "description": "不含目录的文件名模式，例如 *.py、*.js，默认 *。"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS,
                            "default": DEFAULT_MAX_RESULTS, "description": "最多返回多少条匹配，默认 100，最大 500。"},
        },
        "required": ["keyword"],
        "additionalProperties": False,
    },
    supports_cancellation=True,
)


def execute(arguments, workspace, *, abort=None):
    keyword = arguments["keyword"]
    requested = arguments.get("path", ".")
    pattern = arguments.get("glob", "*")
    limit = arguments.get("max_results", DEFAULT_MAX_RESULTS)
    if not keyword or any(character in keyword for character in "\r\n\x00"):
        raise ToolError("invalid_arguments", "keyword 必须是非空的单行关键词，不能含空字符。")
    if len(json.dumps(keyword, ensure_ascii=False)) > MAX_RESULT_CHARS // 2:
        raise ToolError("invalid_arguments", "关键词过长，无法在搜索结果中完整保留，请缩短关键词。")
    if not requested.strip() or "\x00" in requested:
        raise ToolError("invalid_arguments", "path 必须是非空路径，且不能含空字符。")
    if not pattern or any(character in pattern for character in "/\r\n\x00"):
        raise ToolError("invalid_arguments", "glob 必须是不含目录的文件名模式，例如 *.py。")

    try:
        root = Path(workspace).resolve()
        start = _checked_path(root / requested, root)
        mode = start.stat().st_mode
        if not (S_ISDIR(mode) or S_ISREG(mode)):
            raise ToolError("not_a_file", "搜索路径必须是目录或普通文本文件。")
    except FileNotFoundError:
        raise ToolError("not_found", "搜索路径不存在，请检查 path。") from None
    except PermissionError:
        raise ToolError("permission_denied", "没有访问搜索路径的权限。") from None
    except (OSError, RuntimeError, ValueError):
        raise ToolError("search_error", "无法访问搜索路径，请检查 path。") from None

    matches = []
    skipped = {"skipped_files": 0, "skipped_directories": 0}
    truncated = cancelled = False

    def result():
        message = f"返回 {len(matches)} 条匹配。"
        if truncated:
            message += "结果已截断，请缩小搜索范围。"
        if skipped["skipped_files"] or skipped["skipped_directories"]:
            message += f'已跳过 {skipped["skipped_files"]} 个文件、{skipped["skipped_directories"]} 个目录。'
        if cancelled:
            message = "搜索已取消，返回已找到的部分结果。"
        return {
            "status": "error" if cancelled else "success",
            "matches": matches, "returned_count": len(matches), "truncated": truncated,
            **skipped, "cancelled": cancelled, "message": message,
        }

    for candidate in _files(start, S_ISDIR(mode), skipped, abort):
        if abort is not None and abort.is_set():
            break
        if not fnmatchcase(candidate.name, pattern):
            continue
        try:
            path = _checked_path(candidate, root)
            metadata = path.stat()
            if not S_ISREG(metadata.st_mode) or metadata.st_size > MAX_FILE_BYTES:
                skipped["skipped_files"] += 1
                continue
            with path.open("rb") as source:
                data = source.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES or b"\x00" in data:
                skipped["skipped_files"] += 1
                continue
            content = data.decode("utf-8")
            relative_path = str(candidate.relative_to(root))
            relative_path.encode("utf-8")
        except (ToolError, OSError, RuntimeError, ValueError):
            skipped["skipped_files"] += 1
            continue
        for line_number, raw_line in enumerate(StringIO(content, newline=None), 1):
            line = raw_line.removesuffix("\n")
            if abort is not None and abort.is_set():
                cancelled = truncated = True
                return result()
            if keyword not in line:
                continue
            if len(matches) >= limit:
                truncated = True
                return result()
            excerpt, line_truncated = _excerpt(line, keyword)
            matches.append({"path": relative_path, "line_number": line_number,
                            "content": excerpt, "line_truncated": line_truncated})
            # 使用执行层最终封装计算预算，JSON 转义字符也计入长度。
            envelope = {**result(), "executed": True, "tool": "grep"}
            if len(json.dumps(envelope, ensure_ascii=False)) > MAX_RESULT_CHARS:
                matches.pop()
                truncated = True
                return result()
    if abort is not None and abort.is_set():
        cancelled = truncated = True
    return result()


def _files(start, is_directory, skipped, abort):
    if not is_directory:
        yield start
        return

    def on_error(error):
        skipped["skipped_directories"] += 1

    for directory, names, filenames in os.walk(start, onerror=on_error, followlinks=False):
        if abort is not None and abort.is_set():
            return
        allowed = []
        for name in sorted(names):
            child = Path(directory) / name
            try:
                if _protected(child) or child.is_symlink():
                    skipped["skipped_directories"] += 1
                    continue
            except OSError:
                skipped["skipped_directories"] += 1
                continue
            allowed.append(name)
        names[:] = allowed
        for name in sorted(filenames):
            if abort is not None and abort.is_set():
                return
            yield Path(directory) / name


def _checked_path(candidate, root):
    if _protected(candidate):
        raise ToolError("access_denied", "不允许搜索 .env* 或 .git 路径。")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ToolError("access_denied", "只能搜索启动目录及其子目录内的文件。")
    if _protected(resolved):
        raise ToolError("access_denied", "不允许搜索 .env* 或 .git 路径。")
    return resolved


def _protected(path):
    return any(part.casefold() == ".git" or part.casefold().startswith(".env") for part in path.parts)


def _excerpt(line, keyword):
    keep = max(MAX_LINE_CHARS, len(keyword))
    if len(line) <= keep:
        return line, False
    before = min(100, keep - len(keyword))
    start = max(0, line.index(keyword) - before)
    end = min(len(line), start + keep)
    return ("…" if start else "") + line[start:end] + ("…" if end < len(line) else ""), True
