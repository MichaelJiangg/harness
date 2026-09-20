"""会话记忆核心：JSON 存储、对话筛选、模型摘要与失败降级。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import RLock

from ..client import DEFAULT_MODEL


MEMORY_FILE = ".harness/memory.json"
DEFAULT_RECENT_COUNT = 5
DEFAULT_MAX_RECORDS = 20
MAX_SUMMARY_CHARS = 1000
MAX_TOPICS = 8
MAX_TOPIC_CHARS = 40
SUMMARY_REQUEST_CHARS = 12000
SUMMARY_MAX_TOKENS = 900


class MemoryError(RuntimeError):
    """可安全显示给用户的本地记忆错误。"""


def _clean_text(value):
    return "".join(character for character in value if character == "\n" or character.isprintable())


def _truncate(value, limit):
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _infer_topics(summary):
    parts = [
        _truncate(part.strip(), MAX_TOPIC_CHARS)
        for part in re.split(r"[。；;\n]", summary)
        if part.strip()
    ]
    return list(dict.fromkeys(parts))[:3]


def _empty_document():
    return {"version": 1, "sessions": []}


def _validate_document(document):
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError
    sessions = document.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError
    for record in sessions:
        if not isinstance(record, dict):
            raise ValueError
        if (not isinstance(record.get("id"), (str, int))
                or not str(record["id"]).strip()
                or not isinstance(record.get("date"), str)
                or not isinstance(record.get("summary"), str)
                or not record["summary"].strip()):
            raise ValueError
        topics = record.get("topics", [])
        if not isinstance(topics, list) or not all(
            isinstance(topic, str) and topic.strip() for topic in topics
        ):
            raise ValueError
        key_points = record.get("key_points", [])
        if not isinstance(key_points, list) or not all(
            isinstance(point, str) and point.strip() for point in key_points
        ):
            raise ValueError
        if "message_count" in record and (
            type(record["message_count"]) is not int or record["message_count"] < 0
        ):
            raise ValueError


class MemoryStore:
    """在当前工作区保存有界、可读、可手动编辑的 JSON 记忆。"""

    def __init__(self, workspace=None, *, max_records=DEFAULT_MAX_RECORDS):
        if type(max_records) is not int or max_records < 1:
            raise ValueError("记忆条数上限必须是正整数。")
        self.workspace = (Path.cwd() if workspace is None else Path(workspace)).resolve()
        self.path = self.workspace / MEMORY_FILE
        self.max_records = max_records
        self._lock = RLock()

    def _load(self):
        try:
            content = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _empty_document()
        except (OSError, UnicodeError):
            raise MemoryError("无法读取本地记忆文件，请检查文件权限和 UTF-8 编码。") from None
        try:
            document = json.loads(content)
            _validate_document(document)
        except (json.JSONDecodeError, ValueError):
            raise MemoryError("本地记忆文件格式无效，已跳过加载；请修复或手动检查该文件。") from None
        return document

    def _save(self, document):
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not self.path.parent.is_dir():
                raise OSError
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                prefix=".memory-", suffix=".tmp", delete=False,
            ) as target:
                json.dump(document, target, ensure_ascii=False, indent=2)
                target.write("\n")
                temporary = Path(target.name)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass
            raise MemoryError("无法保存本地记忆文件，请检查工作目录权限。") from None

    def records(self):
        with self._lock:
            return [deepcopy(record) for record in self._load()["sessions"]]

    def recent(self, count=DEFAULT_RECENT_COUNT):
        if type(count) is not int or count < 0:
            raise ValueError("最近记忆条数必须是非负整数。")
        return self.records()[-count:]

    def add(self, summary, *, topics=None, key_points=None, date=None, message_count=None):
        summary = _truncate(_clean_text(summary), MAX_SUMMARY_CHARS)
        if not summary:
            raise ValueError("会话摘要不能为空。")
        topics = topics if topics is not None else []
        if not isinstance(topics, list):
            raise ValueError("topics 必须是列表。")
        topics = list(dict.fromkeys(
            _truncate(_clean_text(topic), MAX_TOPIC_CHARS)
            for topic in topics
            if isinstance(topic, str) and _clean_text(topic).strip()
        ))[:MAX_TOPICS]
        key_points = key_points if key_points is not None else []
        if not isinstance(key_points, list):
            raise ValueError("key_points 必须是列表。")
        key_points = list(dict.fromkeys(
            _clean_text(point).strip()[:400]
            for point in key_points
            if isinstance(point, str) and _clean_text(point).strip()
        ))[:12]
        if date is None:
            date = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not isinstance(date, str) or not date.strip():
            raise ValueError("记忆日期不能为空。")
        if message_count is not None and (
            type(message_count) is not int or message_count < 0
        ):
            raise ValueError("message_count 必须是非负整数。")

        with self._lock:
            document = self._load()
            sessions = document["sessions"]
            existing = []
            for record in sessions:
                try:
                    existing.append(int(record["id"]))
                except (TypeError, ValueError):
                    continue
            record = {
                "id": str(max(existing, default=0) + 1),
                "date": date,
                "summary": summary,
                "topics": topics,
                "key_points": key_points,
                **({"message_count": message_count} if message_count is not None else {}),
            }
            sessions.append(record)
            if len(sessions) > self.max_records:
                document["sessions"] = sessions[-self.max_records:]
            self._save(document)
            return deepcopy(record)

    def get(self, record_id):
        record_id = str(record_id).strip()
        for record in self.records():
            if str(record["id"]) == record_id:
                return deepcopy(record)
        return None

    def delete(self, record_id):
        record_id = str(record_id).strip()
        with self._lock:
            document = self._load()
            before = len(document["sessions"])
            document["sessions"] = [
                record for record in document["sessions"]
                if str(record["id"]) != record_id
            ]
            if len(document["sessions"]) == before:
                return False
            self._save(document)
            return True

    def clear(self):
        with self._lock:
            document = self._load()
            count = len(document["sessions"])
            if count:
                self._save(_empty_document())
            return count


def format_record(record):
    """只格式化已从可信存储读取的记录，不把记忆内容当作指令。"""
    date = _clean_text(str(record.get("date", ""))).replace("T", " ")[:10]
    record_id = _clean_text(str(record.get("id", "?")))[:32]
    summary = _clean_text(record.get("summary", "")).replace("\n", " ")
    topics = [_clean_text(topic) for topic in record.get("topics", []) if isinstance(topic, str)]
    topics = [topic for topic in topics if topic]
    result = f"[{record_id}] {date} — {summary}"
    if topics:
        result += f"\n    话题：{'、'.join(topics)}"
    key_points = [
        _clean_text(point) for point in record.get("key_points", [])
        if isinstance(point, str) and _clean_text(point)
    ]
    if key_points:
        result += "\n    要点：" + "；".join(key_points)
    return result


def has_memory_candidates(messages):
    """只有用户或助手产生过实质内容时才需要保存摘要。"""
    return any(
        message.get("role") in {"user", "assistant"} and (
            (isinstance(message.get("content"), str) and message["content"].strip())
            or bool(message.get("tool_calls"))
        )
        for message in messages
    )


def build_memory_transcript(messages, *, limit=SUMMARY_REQUEST_CHARS):
    """摘要只看用户／助手文字和工具名称，不读取工具结果正文。"""
    lines = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"用户：{_clean_text(content.strip())}")
        elif role == "assistant":
            content = message.get("content")
            parts = []
            if isinstance(content, str) and content.strip():
                parts.append(_clean_text(content.strip()))
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                names = []
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    name = function.get("name") if isinstance(function, dict) else None
                    if isinstance(name, str) and name.strip():
                        names.append(_clean_text(name.strip()))
                if names:
                    parts.append("工具调用：" + "、".join(dict.fromkeys(names)))
            if parts:
                lines.append("助手：" + "；".join(parts))
    if not lines:
        return ""
    transcript = "\n\n".join(lines)
    if len(transcript) <= limit:
        return transcript
    return "（较早内容已省略）\n" + transcript[-limit:]


def _response_content(response):
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return _clean_text(content.strip()) if isinstance(content, str) else ""


def _parse_summary(content, fallback):
    summary = _truncate(_clean_text(fallback), MAX_SUMMARY_CHARS)
    topics = _infer_topics(summary)
    if not content:
        return summary, topics, []
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError
        payload = json.loads(content[start:end + 1])
        candidate = payload.get("summary") if isinstance(payload, dict) else None
        candidate_topics = payload.get("topics") if isinstance(payload, dict) else None
        candidate_points = payload.get("key_points") if isinstance(payload, dict) else None
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError
        if not isinstance(candidate_topics, list):
            candidate_topics = []
        if not isinstance(candidate_points, list):
            candidate_points = []
        summary = _truncate(_clean_text(candidate), MAX_SUMMARY_CHARS)
        topics = list(dict.fromkeys(
            _truncate(_clean_text(topic), MAX_TOPIC_CHARS)
            for topic in candidate_topics
            if isinstance(topic, str) and _clean_text(topic).strip()
        ))[:MAX_TOPICS]
        if not topics:
            topics = _infer_topics(summary)
        key_points = list(dict.fromkeys(
            _clean_text(point).strip()[:400]
            for point in candidate_points
            if isinstance(point, str) and _clean_text(point).strip()
        ))[:12]
    except (json.JSONDecodeError, ValueError, TypeError):
        key_points = []
    return summary, topics, key_points


def summarize_session(client, messages, *, model=DEFAULT_MODEL):
    """模型摘要失败时使用最近文字降级保存，不让退出因记忆而失败。"""
    transcript = build_memory_transcript(messages)
    if not transcript:
        return {"summary": "", "topics": [], "response": None}
    request_messages = [
        {
            "role": "system",
            "content": (
                "你是会话记忆提取助手。阅读对话后只提取会影响后续工作的决定、约束、"
                "关键事实、产出和待办；不要复述寒暄、工具调用过程或工具输出。"
                "只返回一个 JSON 对象，不要 Markdown，格式为 "
                '{"summary":"不超过 1000 字的中文摘要",'
                '"key_points":["要点1","要点2"],"topics":["话题1","话题2"]}。'
            ),
        },
        {"role": "user", "content": f"对话内容：\n{transcript}"},
    ]
    response = None
    try:
        response = client.complete(
            model=model, messages=request_messages, tools=[],
            on_text=None, max_tokens=SUMMARY_MAX_TOKENS,
        )
    except Exception as error:
        response = getattr(error, "response", None)
    summary, topics, key_points = _parse_summary(_response_content(response), transcript)
    return {
        "summary": summary, "topics": topics, "key_points": key_points,
        "response": response,
    }
