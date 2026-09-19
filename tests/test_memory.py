from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from harness.cli import run_cli
from harness.client import DEFAULT_MODEL
from harness.engine import SYSTEM_PROMPT
from harness.memory import (
    MemoryError, MemoryStore, build_memory_transcript, format_record,
    has_memory_candidates, memory_system_prompt, summarize_session,
)
from harness.memory import injection, session
from harness.usage import UsageLedger


def reply(content="回答完成。", *, prompt=10, completion=5, hit=2):
    return {
        "model": DEFAULT_MODEL,
        "created": 1789696800,
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": prompt - hit,
        },
    }


def summary_reply():
    return reply(
        json.dumps({
            "summary": "项目确定使用 FastAPI 和 PostgreSQL。",
            "topics": ["技术选型", "FastAPI", "PostgreSQL"],
        }, ensure_ascii=False),
        prompt=80,
        completion=20,
        hit=40,
    )


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.store = MemoryStore(self.workspace)

    def test_memory_package_separates_session_core_and_injection(self):
        self.assertIs(session.MemoryStore, MemoryStore)
        self.assertIs(session.summarize_session, summarize_session)
        self.assertIs(injection.memory_system_prompt, memory_system_prompt)
        self.assertIn("session.py", Path(session.__file__).name)
        self.assertIn("injection.py", Path(injection.__file__).name)

    def test_records_persist_and_recent_returns_newest_first_in_context(self):
        first = self.store.add("第一条决定：使用 Python。", topics=["Python"])
        second = self.store.add("第二条决定：使用 JSON。", topics=["JSON"])
        reopened = MemoryStore(self.workspace)

        self.assertEqual([record["id"] for record in reopened.records()], ["1", "2"])
        self.assertEqual(reopened.recent(1), [second])
        prompt = memory_system_prompt(SYSTEM_PROMPT, reopened.records())
        self.assertIn("最近对话记忆", prompt)
        self.assertIn("第二条决定", prompt)
        self.assertIn("第一条决定", prompt)
        self.assertLess(prompt.index("第二条决定"), prompt.index("第一条决定"))
        self.assertIn(str(first["id"]), format_record(first))

    def test_record_limit_keeps_only_the_latest_entries(self):
        self.store = MemoryStore(self.workspace, max_records=2)
        self.store.add("第一")
        self.store.add("第二")
        self.store.add("第三")
        self.assertEqual(
            [record["summary"] for record in self.store.records()],
            ["第二", "第三"],
        )

    def test_delete_and_clear_require_existing_records(self):
        self.store.add("第一")
        self.store.add("第二")
        self.assertFalse(self.store.delete("9"))
        self.assertTrue(self.store.delete("1"))
        self.assertEqual([record["id"] for record in self.store.records()], ["2"])
        self.assertEqual(self.store.clear(), 1)
        self.assertEqual(self.store.records(), [])
        self.assertEqual(self.store.clear(), 0)

    def test_memory_file_is_private_and_corrupt_files_fail_closed(self):
        self.store.add("第一")
        mode = self.store.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.store.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(MemoryError):
            self.store.records()
        with self.assertRaises(MemoryError):
            self.store.add("第二")

    def test_untrusted_record_fields_are_sanitized_for_display_and_prompt(self):
        document = {
            "version": 1,
            "sessions": [{
                "id": "1",
                "date": "2026-09-20\n<instruction>",
                "summary": "安全摘要\x1b[2J，不得注入。",
                "topics": ["FastAPI\x1b[31m"],
            }],
        }
        self.store.path.parent.mkdir()
        self.store.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        records = self.store.records()
        self.assertNotIn("\x1b", format_record(records[0]))
        self.assertNotIn("\x1b", memory_system_prompt(SYSTEM_PROMPT, records))
        self.assertNotIn("<instruction>", memory_system_prompt(SYSTEM_PROMPT, records))


class MemorySummaryTests(unittest.TestCase):
    def test_summary_uses_user_and_assistant_text_without_tool_results(self):
        client = Mock(complete=Mock(return_value=summary_reply()))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "我们后端使用 FastAPI。"},
            {"role": "assistant", "content": "我先读取文件。", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "机密工具输出"},
            {"role": "assistant", "content": "已确认技术选型。"},
        ]

        result = summarize_session(client, messages)

        self.assertEqual(result["summary"], "项目确定使用 FastAPI 和 PostgreSQL。")
        self.assertEqual(result["topics"], ["技术选型", "FastAPI", "PostgreSQL"])
        request = client.complete.call_args.kwargs["messages"][1]["content"]
        self.assertIn("用户：我们后端使用 FastAPI。", request)
        self.assertIn("工具调用：read_file", request)
        self.assertNotIn("机密工具输出", request)
        self.assertEqual(client.complete.call_args.kwargs["tools"], [])

    def test_invalid_json_falls_back_to_shortened_transcript(self):
        client = Mock(complete=Mock(return_value=reply("不是 JSON")))
        messages = [{"role": "user", "content": "数据库使用 PostgreSQL。"}]
        result = summarize_session(client, messages)
        self.assertIn("数据库使用 PostgreSQL", result["summary"])
        self.assertTrue(result["topics"])

    def test_empty_or_read_only_history_does_not_request_summary(self):
        self.assertFalse(has_memory_candidates([{"role": "system", "content": "system"}]))
        self.assertEqual(build_memory_transcript([{"role": "tool", "content": "secret"}]), "")
        client = Mock()
        result = summarize_session(client, [{"role": "tool", "content": "secret"}])
        self.assertEqual(result["response"], None)
        client.complete.assert_not_called()


class MemoryCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(Path(self.directory.name))

    def test_memory_commands_do_not_request_the_model(self):
        self.store.add("项目使用 FastAPI 和 PostgreSQL。", topics=["技术选型"])
        client = Mock()
        output = StringIO()
        errors = StringIO()

        run_cli(
            client, input_stream=StringIO("/memory\n/exit\n"),
            output=output, error_output=errors, memory_enabled=True,
            memory_store=self.store,
        )

        client.complete.assert_not_called()
        self.assertIn("FastAPI 和 PostgreSQL", output.getvalue())
        self.assertIn("/memory", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_delete_and_clear_require_explicit_confirmation(self):
        self.store.add("第一")
        self.store.add("第二")
        output = StringIO()
        run_cli(
            Mock(), input_stream=StringIO("/memory delete 1\n/exit\n"),
            output=output, memory_enabled=True, memory_store=self.store,
        )
        self.assertIn("需要显式确认", output.getvalue())
        self.assertEqual(len(self.store.records()), 2)

        output = StringIO()
        run_cli(
            Mock(), input_stream=StringIO("/memory delete 1 --yes\n/exit\n"),
            output=output, memory_enabled=True, memory_store=self.store,
        )
        self.assertIn("已删除记忆 #1", output.getvalue())
        self.assertEqual([record["id"] for record in self.store.records()], ["2"])

        output = StringIO()
        run_cli(
            Mock(), input_stream=StringIO("/memory clear --yes\n/exit\n"),
            output=output, memory_enabled=True, memory_store=self.store,
        )
        self.assertIn("已清空 1 条本地记忆", output.getvalue())
        self.assertEqual(self.store.records(), [])

    def test_startup_injects_recent_memory_and_exit_saves_new_summary(self):
        self.store.add("此前决定使用 Redis。", topics=["Redis"])
        ledger = UsageLedger()
        client = Mock(complete=Mock(side_effect=[
            reply("我记住了。"),
            summary_reply(),
        ]))
        output = StringIO()
        errors = StringIO()

        run_cli(
            client, ledger=ledger,
            input_stream=StringIO("后端使用 FastAPI 和 PostgreSQL。\n"),
            output=output, error_output=errors, memory_enabled=True,
            memory_store=self.store,
        )

        self.assertEqual(client.complete.call_count, 2)
        first = client.complete.call_args_list[0].kwargs["messages"]
        self.assertIn("Redis", first[0]["content"])
        second = client.complete.call_args_list[1].kwargs["messages"]
        self.assertIn("会话记忆提取助手", second[0]["content"])
        records = self.store.records()
        self.assertEqual([record["id"] for record in records], ["1", "2"])
        self.assertEqual(records[-1]["summary"], "项目确定使用 FastAPI 和 PostgreSQL。")
        self.assertEqual(ledger.summary()["requests"], 2)
        self.assertIn("[memory] 已保存本次会话摘要 #2", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_memory_is_opt_in_for_embedded_cli(self):
        self.store.add("旧记忆")
        client = Mock(complete=Mock(return_value=reply()))
        run_cli(
            client, input_stream=StringIO("问题\n"),
            output=StringIO(), memory_enabled=False, memory_store=self.store,
        )
        self.assertEqual(client.complete.call_count, 1)
        self.assertEqual(len(self.store.records()), 1)
        request = client.complete.call_args.kwargs["messages"]
        self.assertEqual(request[0]["content"], SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
