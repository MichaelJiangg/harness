from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from harness.cli import run_cli
from harness.memory import MemoryStore
from harness.memory.search import basic_recall, format_recall_result, recall_memories
from harness.usage import UsageLedger


class MemorySearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(Path(self.directory.name))
        self.store.add(
            "数据库连接池配置：最大 20 连接，超时 30 秒。",
            topics=["数据库", "连接池"], key_points=["最大连接 20", "超时 30s"],
        )
        self.store.add(
            "PostgreSQL 索引优化讨论。",
            topics=["PostgreSQL", "索引"], key_points=["分析慢查询"],
        )
        self.store.add(
            "数据库选型：决定使用 PostgreSQL。",
            topics=["数据库"], key_points=["选择 PostgreSQL"],
        )

    def test_basic_recall_finds_semantically_related_chinese_records(self):
        matches = basic_recall("数据库性能", self.store.records(), limit=3)
        self.assertTrue(matches)
        self.assertTrue(any("连接池" in record["summary"] for record, _ in matches))
        self.assertTrue(format_recall_result(matches[0][0], matches[0][1]).startswith("[0."))

    def test_recall_filters_below_threshold(self):
        matches = basic_recall("完全无关的猫咪话题", self.store.records())
        self.assertEqual(matches, [])

    def test_vector_matches_are_merged_with_records_and_score_filtered(self):
        fake = Mock()
        fake.query.return_value = [
            {"record_id": "1", "score": 0.91, "date": "2026-09-20", "text": ""},
            {"record_id": "9", "score": 0.88, "date": "2026-09-20", "text": ""},
            {"record_id": "2", "score": 0.30, "date": "2026-09-20", "text": ""},
        ]
        matches = recall_memories("数据库", self.store.records(), vector_store=fake)
        self.assertEqual([record["id"] for record, _ in matches], ["1"])

    def test_cli_recall_command_uses_text_fallback_without_vector_store(self):
        client = Mock()
        output = StringIO()
        run_cli(
            client,
            input_stream=StringIO("/recall 数据库性能\n/exit\n"),
            output=output,
            memory_enabled=True,
            memory_store=self.store,
        )
        self.assertIn("向量搜索不可用", output.getvalue())
        self.assertIn("连接池配置", output.getvalue())
        client.complete.assert_not_called()

    def test_semantic_store_injects_relevant_memory_for_current_question(self):
        vector_store = Mock()
        vector_store.available = True
        vector_store.sync.return_value = True
        vector_store.query.return_value = [{
            "record_id": "1", "score": 0.92,
            "date": "2026-09-20", "text": "数据库连接池配置",
        }]
        client = Mock(complete=Mock(return_value={
            "model": "deepseek-flash",
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "已回忆。"},
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10,
            },
        }))
        run_cli(
            client,
            ledger=UsageLedger(),
            input_stream=StringIO("数据库连接池配置是什么？\n"),
            output=StringIO(),
            memory_enabled=True,
            memory_store=self.store,
            vector_store=vector_store,
        )
        system = client.complete.call_args_list[0].kwargs["messages"][0]["content"]
        self.assertIn("数据库连接池配置", system)


if __name__ == "__main__":
    unittest.main()
