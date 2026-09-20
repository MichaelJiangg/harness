"""可选向量记忆召回；未安装 ChromaDB 时使用本地文本相关度。"""

import re

from .session import DEFAULT_RECENT_COUNT


EMBEDDING_ERROR = "向量记忆不可用，已退回到文本召回。"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.45


def semantic_memory_available():
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


class VectorMemoryStore:
    """ChromaDB 持久化向量索引；任何安装或运行时失败都回退到 False。"""

    def __init__(self, workspace):
        self.workspace = workspace
        self._client = None
        self._collection = None
        self._available = semantic_memory_available()
        if self._available:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=str(workspace / ".harness" / "chroma"))
                self._collection = self._client.get_or_create_collection(
                    "harness_memory", metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                self._available = False
                self._client = None
                self._collection = None

    @property
    def available(self):
        return self._available and self._collection is not None

    def sync(self, records):
        if not self.available:
            return False
        try:
            items = [
                {
                    "id": str(record["id"]),
                    "document": _document_for_record(record),
                    "metadata": {"id": str(record["id"]), "date": record.get("date", "")},
                }
                for record in records
            ]
            if items:
                self._collection.upsert(
                    ids=[item["id"] for item in items],
                    documents=[item["document"] for item in items],
                    metadatas=[item["metadata"] for item in items],
                )
            return True
        except Exception:
            self._available = False
            self._collection = None
            return False

    def query(self, text, *, limit=DEFAULT_TOP_K):
        if not self.available or not text.strip():
            return []
        try:
            result = self._collection.query(query_texts=[text.strip()], n_results=limit)
        except Exception:
            self._available = False
            self._collection = None
            return []
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        matches = []
        for index, record_id in enumerate(ids[0]):
            try:
                distance = float(distances[0][index])
            except (IndexError, TypeError, ValueError):
                continue
            metadata = metadatas[0][index] if index < len(metadatas[0]) else {}
            document = documents[0][index] if index < len(documents[0]) else ""
            matches.append({
                "record_id": str(record_id),
                "score": 1.0 / (1.0 + max(0.0, distance)),
                "date": str(metadata.get("date", "") if isinstance(metadata, dict) else ""),
                "text": str(document),
            })
        return matches


def recall_memories(query, records, *, vector_store=None, limit=DEFAULT_TOP_K,
                   threshold=DEFAULT_THRESHOLD):
    """优先语义向量，失败或未安装时用中文字符 bigram 文本召回。"""
    if vector_store is not None:
        matched = vector_store.query(query, limit=limit)
        if matched:
            by_id = {str(record["id"]): record for record in records}
            return [
                (by_id[item["record_id"]], max(0.0, min(1.0, float(item["score"]))))
                for item in matched
                if item["record_id"] in by_id and item["score"] >= threshold
            ][:limit]
    return basic_recall(query, records, limit=limit, threshold=threshold)


def basic_recall(query, records, *, limit=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD):
    query_bigrams = _bigrams(query)
    if not query_bigrams:
        return []
    scored = []
    for record in records:
        document = _document_for_record(record)
        document_bigrams = _bigrams(document)
        overlap = len(query_bigrams & document_bigrams)
        if not overlap:
            continue
        score = overlap / (len(query_bigrams) + len(document_bigrams) - overlap)
        query_lower = query.casefold()
        document_lower = document.casefold()
        if query_lower in document_lower:
            score = max(score, 0.55)
        topics = " ".join(str(topic) for topic in record.get("topics", []))
        if any(topic.casefold() and topic.casefold() in query_lower for topic in topics.split()):
            score = max(score, 0.62)
        if score >= threshold:
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(record, score) for score, record in scored[:limit]]


def format_recall_result(record, score):
    date = str(record.get("date", "")).replace("T", " ")[:10]
    summary = re.sub(r"\s+", " ", record.get("summary", "")).strip()
    return f"[{score:.2f}] {date} — {summary}"


def _document_for_record(record):
    parts = [record.get("summary", "")]
    parts.extend(record.get("topics", []))
    parts.extend(record.get("key_points", []))
    return "\n".join(str(part) for part in parts if part).strip()


def _bigrams(text):
    normalized = re.sub(r"[\s\W_]+", "", str(text).casefold())
    return {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}
