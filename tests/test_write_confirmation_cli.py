import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from harness.cli import HELP
from harness.engine import query_loop
from harness.permissions import PermissionPolicy
from harness.tools import create_tool_executor
from test_cli import CLISession, QueuedInput, reply, visible_text


CONFIRM_PROMPT = "[确认] 输入 y 批准本次写入"


def write_call(path, content, call_id="write_1"):
    return {
        "id": call_id, "type": "function",
        "function": {"name": "write_file", "arguments": json.dumps({"path": path, "content": content}, ensure_ascii=False)},
    }


class WriteConfirmationCLITests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name).resolve()
        cwd = patch("harness.tools.executor.Path.cwd", return_value=self.root)
        cwd.start()
        self.addCleanup(cwd.stop)

    def start(self, calls, *, terminal=True, policy=None):
        if policy is not None:
            factory = patch("harness.cli.create_tool_executor", side_effect=lambda **kwargs:
                            create_tool_executor(permissions=policy, **kwargs))
            factory.start()
            self.addCleanup(factory.stop)
        client = Mock(complete=Mock(side_effect=[reply(None, tool_calls=calls), reply("写入请求处理完毕。")]))
        session = CLISession(client, lines=("请写入文件\n",), terminal=terminal, character_delay=0)
        self.addCleanup(session.close)
        return session, client

    def results(self, client):
        return [json.loads(message["content"])
                for message in client.complete.call_args_list[1].kwargs["messages"]
                if message["role"] == "tool"]

    def test_full_preview_precedes_approval_and_directory_creation(self):
        content = "开头" + "甲" * 6100 + "尾部"
        session, client = self.start([write_call("new/nested/note.txt", content)])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        output = visible_text(session.output.getvalue())
        self.assertIn(str(self.root / "new/nested/note.txt"), output)
        self.assertIn("操作：新建文件", output)
        self.assertIn("风险等级：中风险（写入文件）", output)
        self.assertIn(f"本会话授权目录：{self.root / 'new/nested'}（含子目录）", output)
        self.assertIn("写入成功后记住以上目录，当前会话内复用，重启失效", output)
        self.assertNotIn("高风险操作警告", output)
        self.assertIn(content, output)
        self.assertLess(output.index("尾部"), output.index(CONFIRM_PROMPT))
        self.assertFalse((self.root / "new").exists())
        client.complete.assert_called_once()
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual((self.root / "new/nested/note.txt").read_text(encoding="utf-8"), content)
        self.assertTrue(self.results(client)[0]["executed"])

    def test_existing_file_is_overwritten_only_after_confirmation(self):
        target = self.root / "existing.txt"
        target.write_text("原来的长内容。", encoding="utf-8")
        session, _ = self.start([write_call("existing.txt", "新内容\n第二行")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        self.assertIn("操作：覆盖已有文件的全部内容", session.output.getvalue())
        self.assertIn("│ 新内容\n│ 第二行", session.output.getvalue())
        self.assertEqual(target.read_text(encoding="utf-8"), "原来的长内容。")
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(target.read_text(encoding="utf-8"), "新内容\n第二行")

    def test_new_file_appearing_during_confirmation_invalidates_approval(self):
        target = self.root / "created_later.txt"
        session, client = self.start([write_call("created_later.txt", "模型计划写入的内容")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        self.assertIn("操作：新建文件", session.output.getvalue())
        target.write_text("确认期间另一个进程创建的内容", encoding="utf-8")
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(target.read_text(encoding="utf-8"), "确认期间另一个进程创建的内容")
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")
        self.assertIn("目标路径或文件存在状态已变化", session.output.getvalue())

    def test_symlink_retargeted_during_confirmation_invalidates_approval(self):
        first = self.root / "first.txt"
        second = self.root / "second.txt"
        link = self.root / "target.txt"
        first.write_text("第一个文件的原文", encoding="utf-8")
        second.write_text("第二个文件的原文", encoding="utf-8")
        link.symlink_to(first)
        session, client = self.start([write_call("target.txt", "新内容")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        self.assertIn(f"目标路径：{first}", session.output.getvalue())
        link.unlink()
        link.symlink_to(second)
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(first.read_text(encoding="utf-8"), "第一个文件的原文")
        self.assertEqual(second.read_text(encoding="utf-8"), "第二个文件的原文")
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")
        self.assertIn("目标路径或文件存在状态已变化", session.output.getvalue())

    def test_rejection_preserves_existing_content(self):
        target = self.root / "existing.txt"
        target.write_text("保留原文", encoding="utf-8")
        session, client = self.start([write_call("existing.txt", "替换内容")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("n\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(target.read_text(encoding="utf-8"), "保留原文")
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")

    def test_blank_confirmation_does_not_create_parent_directories(self):
        session, client = self.start([write_call("absent/child.txt", "内容")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertFalse((self.root / "absent").exists())
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")

    def test_writes_in_separate_directories_require_separate_confirmations(self):
        session, client = self.start([
            write_call("first/first.txt", "第一次"),
            write_call("second/second.txt", "第二次", "write_2"),
        ])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("│ 第二次"))
        self.assertEqual((self.root / "first/first.txt").read_text(encoding="utf-8"), "第一次")
        self.assertFalse((self.root / "second").exists())
        client.complete.assert_called_once()
        session.input.send("n\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(session.output.getvalue().count(CONFIRM_PROMPT), 2)
        self.assertEqual([result["executed"] for result in self.results(client)], [True, False])
        self.assertFalse((self.root / "second").exists())

    def test_successful_write_remembers_directory_and_subdirectories_for_this_session(self):
        session, client = self.start([
            write_call("src/first.py", "first"),
            write_call("src/second.py", "second", "write_2"),
            write_call("src/nested/third.py", "third", "write_3"),
        ])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        self.assertFalse((self.root / "src").exists())
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(session.output.getvalue().count(CONFIRM_PROMPT), 1)
        self.assertEqual([result["executed"] for result in self.results(client)], [True] * 3)
        for name, content in (("first.py", "first"), ("second.py", "second"), ("nested/third.py", "third")):
            self.assertEqual((self.root / "src" / name).read_text(encoding="utf-8"), content)

    def test_new_cli_session_requires_confirmation_for_previously_approved_directory(self):
        first, _ = self.start([write_call("src/first.py", "first")])
        self.assertTrue(first.output.wait_for(CONFIRM_PROMPT))
        first.input.send("y\n")
        self.assertTrue(first.output.wait_for("写入请求处理完毕。"))
        first.close()
        second, client = self.start([write_call("src/second.py", "second")])
        self.assertTrue(second.output.wait_for(CONFIRM_PROMPT))
        self.assertFalse((self.root / "src/second.py").exists())
        second.input.send("n\n")
        self.assertTrue(second.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")
        self.assertFalse((self.root / "src/second.py").exists())

    def test_deny_rule_wins_over_remembered_directory(self):
        policy = PermissionPolicy(rules=[{
            "tool": "write_file", "action": "deny", "directory": "src/restricted",
        }])
        session, client = self.start([
            write_call("src/first.py", "first"),
            write_call("src/restricted/secret.py", "blocked", "write_2"),
        ], policy=policy)
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual(session.output.getvalue().count(CONFIRM_PROMPT), 1)
        self.assertEqual((self.root / "src/first.py").read_text(encoding="utf-8"), "first")
        self.assertFalse((self.root / "src/restricted").exists())
        results = self.results(client)
        self.assertTrue(results[0]["executed"])
        self.assertEqual(results[1]["code"], "permission_denied")
        self.assertFalse(results[1]["executed"])

    def test_local_commands_and_invalid_answer_do_not_approve_or_reach_model(self):
        session, client = self.start([write_call("result.txt", "等待确认")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("/cost\n")
        self.assertTrue(session.output.wait_for("模型请求：1 次。"))
        session.input.send("/help\n")
        session.input.send("/compact\n")
        self.assertTrue(session.output.wait_for("暂时无法压缩"))
        self.assertEqual(session.output.getvalue().count("/help  查看帮助"), 2)
        session.input.send("yes\n")
        self.assertTrue(session.output.wait_for("正在等待本次写入确认"))
        self.assertFalse((self.root / "result.txt").exists())
        client.complete.assert_called_once()
        session.input.send("n\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        messages = client.complete.call_args_list[1].kwargs["messages"]
        self.assertEqual([message["content"] for message in messages if message["role"] == "user"], ["请写入文件"])

    def test_eof_rejects_pending_and_later_confirmation_without_deadlock(self):
        session, client = self.start([
            write_call("first/file.txt", "第一次"),
            write_call("second/file.txt", "第二次", "write_2"),
        ])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.close()
        self.assertFalse((self.root / "first").exists())
        self.assertFalse((self.root / "second").exists())
        self.assertEqual(session.output.getvalue().count(CONFIRM_PROMPT), 1)
        self.assertEqual([result["code"] for result in self.results(client)], ["confirmation_denied"] * 2)

    def test_eof_before_model_reply_rejects_late_write(self):
        requested = Event()
        release = Event()

        def initial_reply(**_):
            requested.set()
            if not release.wait(3):
                raise AssertionError("Model request was not released")
            return reply(None, tool_calls=[write_call("late/file.txt", "内容")])

        responses = iter([initial_reply, lambda **_: reply("已取消写入。")])
        client = Mock(complete=Mock(side_effect=lambda **kwargs: next(responses)(**kwargs)))
        session = CLISession(client, lines=("请写入文件\n", ""), terminal=True, character_delay=0)
        try:
            self.assertTrue(requested.wait(3))
            self.assertTrue(session.input.eof_read.wait(3))
            release.set()
            session.join()
            self.assertFalse((self.root / "late").exists())
            self.assertNotIn(CONFIRM_PROMPT, session.output.getvalue())
            self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")
        finally:
            release.set()
            session.close()

    def test_exit_wakes_pending_confirmation_and_stops_next_model_request(self):
        finished = Event()

        def tracked_query(state):
            try:
                return query_loop(state)
            finally:
                finished.set()

        with patch("harness.cli.query_loop", side_effect=tracked_query):
            session, client = self.start([write_call("exit/file.txt", "内容")])
            self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
            session.input.send("/exit\n")
            session.join()
            self.assertTrue(finished.wait(3))
            self.assertFalse((self.root / "exit").exists())
            client.complete.assert_called_once()

    def test_keyboard_interrupt_wakes_pending_confirmation(self):
        finished = Event()
        original_readline = QueuedInput.readline

        def readline(stream):
            line = original_readline(stream)
            if line == "interrupt\n":
                raise KeyboardInterrupt
            return line

        def tracked_query(state):
            try:
                return query_loop(state)
            finally:
                finished.set()

        with patch.object(QueuedInput, "readline", readline), patch("harness.cli.query_loop", side_effect=tracked_query):
            session, client = self.start([write_call("interrupt/file.txt", "内容")])
            self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
            session.input.send("interrupt\n")
            session.join()
            self.assertTrue(finished.wait(3))
            self.assertFalse((self.root / "interrupt").exists())
            self.assertIn("查询已停止", session.output.getvalue())
            client.complete.assert_called_once()

    def test_non_interactive_mode_rejects_without_prompting(self):
        session, client = self.start([write_call("pipe/file.txt", "内容")], terminal=False)
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertIn("非交互模式无法确认写入", session.output.getvalue())
        self.assertNotIn(CONFIRM_PROMPT, session.output.getvalue())
        self.assertFalse((self.root / "pipe").exists())
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")

    def test_terminal_control_sequences_are_visible_and_do_not_change_written_content(self):
        content = "原文\x1b[2J\r末尾\u202e"
        session, _ = self.start([write_call("controls.txt", content)])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        output = visible_text(session.output.getvalue())
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\u202e", output)
        self.assertIn("原文\\u001b[2J\\u000d末尾\\u202e", output)
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("写入请求处理完毕。"))
        self.assertEqual((self.root / "controls.txt").read_bytes(), content.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
