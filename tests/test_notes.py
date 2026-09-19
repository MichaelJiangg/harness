from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from harness.cli import run_cli
from harness.client import DEFAULT_MODEL
from harness.engine import SYSTEM_PROMPT
from harness.notes import MAX_NOTES_BYTES, NotesError, NotesStore, notes_system_prompt
from harness.permissions import PermissionPolicy
from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.usage import UsageLedger


def reply(content="记住了。", *, prompt=10, completion=5, hit=2):
    return {
        "model": DEFAULT_MODEL,
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


def tool_call(call_id="notes-1", name="notes_append", arguments=None):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}, ensure_ascii=False),
        },
    }


class NotesStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.store = NotesStore(self.workspace)

    def test_append_replace_clear_and_persistence(self):
        self.assertEqual(self.store.read(), "")
        self.store.append("## 技术栈\n- FastAPI")
        self.store.append("- PostgreSQL")
        self.assertEqual(
            NotesStore(self.workspace).read(),
            "## 技术栈\n- FastAPI\n- PostgreSQL\n",
        )
        self.store.replace("# 新笔记")
        self.assertEqual(self.store.read(), "# 新笔记")
        self.store.clear()
        self.assertEqual(self.store.read(), "")

    def test_append_keeps_bullet_lists_and_separates_headings(self):
        self.store.append("- 第一项")
        self.store.append("- 第二项")
        self.store.append("## 架构")
        self.assertEqual(
            self.store.read(),
            "- 第一项\n- 第二项\n\n## 架构\n",
        )

    def test_notes_file_is_regular_and_writable(self):
        self.store.append("# 笔记")
        mode = self.store.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o644)

    def test_notes_system_prompt_includes_and_truncates_context(self):
        content = "# 规则\n- Black\n"
        prompt = notes_system_prompt(SYSTEM_PROMPT, content)
        self.assertIn("项目长期笔记", prompt)
        self.assertIn("Black", prompt)
        large = "# 大笔记\n" + ("长内容" * 5000)
        prompt = notes_system_prompt(SYSTEM_PROMPT, large)
        self.assertLess(len(prompt), len(SYSTEM_PROMPT) + 8500)
        self.assertIn("notes_read", prompt)

    def test_symlink_directory_binary_and_size_limits_are_rejected(self):
        self.workspace.joinpath("target.md").write_text("外部", encoding="utf-8")
        self.store.path.symlink_to(self.workspace / "target.md")
        with self.assertRaises(NotesError):
            self.store.read()
        with self.assertRaises(NotesError):
            self.store.append("内容")
        self.store.path.unlink()
        self.store.path.mkdir()
        with self.assertRaises(NotesError):
            self.store.read()
        self.store.path.rmdir()
        self.store.path.write_bytes(b"a\x00b")
        with self.assertRaises(NotesError):
            self.store.read()
        self.store.clear()
        with self.assertRaises(NotesError):
            self.store.replace("a" * (MAX_NOTES_BYTES + 1))


class NotesToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_notes_tools_are_registered_with_fixed_paths(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("notes_read", names)
        self.assertIn("notes_append", names)
        self.assertIn("notes_replace", names)
        self.assertEqual(
            set(REGISTRY.get("notes_read")[0].input_schema["properties"]),
            set(),
        )
        for name in ("notes_append", "notes_replace"):
            self.assertEqual(
                set(REGISTRY.get(name)[0].input_schema["properties"]),
                {"content"},
            )

    def test_default_policy_allows_append_and_requires_confirmation_for_replace(self):
        executor = create_tool_executor(self.root)
        appended = executor("notes_append", {
            "content": "## 编码规范\n- Black 行宽 88",
        })
        self.assertEqual(appended["status"], "success")
        replaced = executor("notes_replace", {"content": "# 新内容"})
        self.assertEqual(replaced["code"], "confirmation_required")
        executor = create_tool_executor(self.root, confirm=Mock(return_value=True))
        replaced = executor("notes_replace", {"content": "# 新内容"})
        self.assertEqual(replaced["status"], "success")
        self.assertEqual((self.root / "HARNESS.md").read_text(encoding="utf-8"), "# 新内容")

    def test_deny_still_blocks_notes_tools(self):
        executor = create_tool_executor(
            self.root,
            permissions=PermissionPolicy(
                allow=["notes_read", "notes_append"],
                deny=["notes_append", "notes_replace"],
            ),
        )
        self.assertEqual(executor("notes_read", {})["status"], "success")
        self.assertEqual(executor("notes_append", {"content": "禁止"})["code"], "permission_denied")
        self.assertEqual(executor("notes_replace", {"content": ""})["code"], "permission_denied")

    def test_model_can_save_manual_remember_instruction(self):
        from harness.engine import QueryState, query_loop

        responses = [
            {
                "model": DEFAULT_MODEL,
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call(arguments={
                            "content": "## 编码规范\n- 使用 Black，行宽 88\n- 使用 ruff",
                        })],
                    },
                }],
                "usage": {
                    "prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28,
                    "prompt_cache_hit_tokens": 10, "prompt_cache_miss_tokens": 10,
                },
            },
            reply(),
        ]
        client = Mock(complete=Mock(side_effect=responses))
        state = QueryState(client, UsageLedger(), tool_executor=create_tool_executor(self.root))
        state.messages.append({"role": "user", "content": "记住：我们团队使用 Black 和 ruff。"})

        answer = query_loop(state)

        self.assertEqual(answer, "记住了。")
        self.assertIn("Black", (self.root / "HARNESS.md").read_text(encoding="utf-8"))
        self.assertEqual(state.ledger.summary()["requests"], 2)


class NotesCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = NotesStore(Path(self.directory.name))

    def test_notes_commands_do_not_request_the_model(self):
        self.store.append("## 技术栈\n- FastAPI")
        client = Mock()
        output = StringIO()
        run_cli(
            client,
            input_stream=StringIO("/notes\n/notes append - PostgreSQL\n/notes\n/exit\n"),
            output=output, notes_enabled=True, notes_store=self.store,
        )
        client.complete.assert_not_called()
        self.assertIn("FastAPI", output.getvalue())
        self.assertIn("PostgreSQL", output.getvalue())
        self.assertIn("已追加项目笔记", output.getvalue())

    def test_replace_and_clear_require_explicit_confirmation(self):
        self.store.append("旧内容")
        output = StringIO()
        run_cli(
            Mock(),
            input_stream=StringIO("/notes replace 新内容\n/exit\n"),
            output=output, notes_enabled=True, notes_store=self.store,
        )
        self.assertIn("需要显式确认", output.getvalue())
        self.assertIn("旧内容", self.store.read())

        output = StringIO()
        run_cli(
            Mock(),
            input_stream=StringIO("/notes replace --yes 新内容\n/notes clear --yes\n/exit\n"),
            output=output, notes_enabled=True, notes_store=self.store,
        )
        self.assertIn("已替换项目笔记", output.getvalue())
        self.assertIn("已清空 HARNESS.md", output.getvalue())
        self.assertEqual(self.store.read(), "")

    def test_startup_injects_project_notes_only_when_enabled(self):
        self.store.append("## 编码规范\n- Black")
        client = Mock(complete=Mock(return_value=reply()))
        run_cli(
            client, input_stream=StringIO("问题\n"),
            output=StringIO(), notes_enabled=True, notes_store=self.store,
        )
        request = client.complete.call_args.kwargs["messages"]
        self.assertIn("Black", request[0]["content"])

        client = Mock(complete=Mock(return_value=reply()))
        run_cli(
            client, input_stream=StringIO("问题\n"),
            output=StringIO(), notes_enabled=False, notes_store=self.store,
        )
        request = client.complete.call_args.kwargs["messages"]
        self.assertEqual(request[0]["content"], SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
