"""把最近会话摘要注入主查询的系统提示。"""

from .session import (
    DEFAULT_RECENT_COUNT, MAX_TOPIC_CHARS, _clean_text, _truncate,
)


MAX_CONTEXT_SUMMARY_CHARS = 600
MAX_MEMORY_CHARS = 6400


def memory_system_prompt(base_prompt, records, *, cold_records=(), count=DEFAULT_RECENT_COUNT,
                         max_chars=MAX_MEMORY_CHARS):
    recent_lines, recent_records = _recent_section(records, count)
    cold_lines, cold_records = _cold_section(cold_records)
    if not recent_lines and not cold_lines:
        return base_prompt
    context = _fit_memory_sections(
        recent_lines, recent_records, cold_lines, cold_records, max_chars,
    )
    return base_prompt + "\n\n" + context


def _recent_section(records, count):
    lines = []
    selected = list(records[-count:])
    for record in reversed(selected):
        line = _record_line(record)
        if line:
            lines.append(line)
    return lines, selected


def _cold_section(cold_records):
    lines = []
    selected = []
    for item in cold_records:
        if isinstance(item, tuple) and len(item) == 2:
            record, score = item
        else:
            record, score = item, None
        line = _record_line(record, score=score)
        if line:
            lines.append(line)
            selected.append((record, score))
    return lines, selected


def _record_line(record, *, score=None):
    date = _clean_text(str(record.get("date", ""))).replace("T", " ")[:10]
    summary = _truncate(
        _clean_text(record.get("summary", "")).replace("\n", " "),
        MAX_CONTEXT_SUMMARY_CHARS,
    )
    if not summary:
        return ""
    topics = [
        _clean_text(topic)[:MAX_TOPIC_CHARS]
        for topic in record.get("topics", [])
        if isinstance(topic, str) and _clean_text(topic).strip()
    ]
    suffix = f"（话题：{'、'.join(topics)}）" if topics else ""
    prefix = f"[{score:.2f}] " if score is not None else ""
    return f"- {prefix}{date}：{summary}{suffix}"


def _fit_memory_sections(recent_lines, recent_records, cold_lines, cold_records, max_chars):
    prefix = "## 最近对话记忆\n"
    suffix = "\n\n## 相关历史记忆\n"
    header_chars = len(prefix) + len(suffix)
    available = max(0, max_chars - header_chars)

    # 冷记忆优先级最低，从末尾逐步裁剪；最近会话其次，从最旧记录开始裁剪。
    while cold_lines and sum(map(len, recent_lines)) + sum(map(len, cold_lines)) > available:
        cold_lines.pop()
        cold_records.pop()
    while recent_lines and sum(map(len, recent_lines)) + sum(map(len, cold_lines)) > available:
        recent_lines.pop()
        recent_records.pop(0)

    sections = []
    if recent_lines:
        sections.append(prefix + "\n".join(recent_lines))
    if cold_lines:
        sections.append(suffix.strip() + "\n" + "\n".join(cold_lines))
    return "\n\n".join(sections)
