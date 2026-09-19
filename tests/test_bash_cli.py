import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from harness.cli import HELP
from harness.engine import query_loop
from test_cli import CLISession, QueuedInput, reply, visible_text


CONFIRM_PROMPT = "[确认] 输入 y 批准本次命令执行"


def bash_call(command, call_id="bash_1", **options):
    return {
        "id": call_id, "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": command, **options}, ensure_ascii=False),
        },
    }


class BashCLITests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        cwd = patch("harness.tools.executor.Path.cwd", return_value=self.root)
        cwd.start()
        self.addCleanup(cwd.stop)

    def start(self, calls, *, terminal=True):
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=calls), reply("命令请求处理完毕。"),
        ]))
        session = CLISession(client, lines=("请执行命令\n",), terminal=terminal, character_delay=0)
        self.addCleanup(session.close)
        return session, client

    def results(self, client):
        return [json.loads(message["content"])
                for message in client.complete.call_args_list[1].kwargs["messages"]
                if message["role"] == "tool"]

    def start_running_command(self, *, timeout=5):
        started = Event()
        processes = []
        popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            started.set()
            return process

        patcher = patch("harness.tools.bash.subprocess.Popen", side_effect=launch)
        patcher.start()
        self.addCleanup(patcher.stop)
        session, client = self.start([bash_call("sleep 10", timeout=timeout)])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("y\n")
        self.assertTrue(started.wait(3))
        return session, client, processes[0]

    def test_full_command_workspace_and_default_timeout_precede_execution(self):
        command = "# 开头" + "甲" * 6100 + "尾部\nprintf done > marker.txt"
        session, client = self.start([bash_call(command)])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        output = visible_text(session.output.getvalue())
        self.assertIn(f"工作目录：{self.root}", output)
        self.assertIn("超时：30 秒", output)
        self.assertIn("│ " + command.replace("\n", "\n│ "), output)
        self.assertIn("风险等级：中风险", output)
        self.assertNotIn("高风险操作警告", output)
        self.assertLess(output.index("printf done > marker.txt"), output.index(CONFIRM_PROMPT))
        self.assertFalse((self.root / "marker.txt").exists())
        client.complete.assert_called_once()
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
        self.assertEqual((self.root / "marker.txt").read_text(), "done")
        self.assertEqual(self.results(client)[0]["exit_code"], 0)
        self.assertTrue(self.results(client)[0]["executed"])

    def test_stdout_stderr_and_nonzero_exit_are_passed_back_to_model(self):
        session, client = self.start([
            bash_call("printf stdout-value; printf stderr-value >&2; exit 9", timeout=7),
        ])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        self.assertIn("超时：7 秒", session.output.getvalue())
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
        result = self.results(client)[0]
        self.assertEqual(result["stdout"], "stdout-value")
        self.assertEqual(result["stderr"], "stderr-value")
        self.assertEqual(result["exit_code"], 9)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["executed"])
        self.assertFalse(result["timed_out"])

    def test_rejected_and_blank_answers_never_execute(self):
        for answer in ("n\n", "\n"):
            with self.subTest(answer=answer):
                session, client = self.start([bash_call("printf changed > marker.txt")])
                self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
                session.input.send(answer)
                self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
                self.assertFalse((self.root / "marker.txt").exists())
                self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")
                session.close()

    def test_each_command_requires_separate_confirmation(self):
        session, client = self.start([
            bash_call("printf first > first.txt"),
            bash_call("printf second > second.txt", "bash_2"),
        ])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("│ printf second > second.txt"))
        self.assertEqual((self.root / "first.txt").read_text(), "first")
        self.assertFalse((self.root / "second.txt").exists())
        client.complete.assert_called_once()
        session.input.send("n\n")
        self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
        self.assertEqual(session.output.getvalue().count(CONFIRM_PROMPT), 2)
        self.assertEqual(session.output.getvalue().count("风险等级：中风险"), 2)
        self.assertNotIn("高风险操作警告", session.output.getvalue())
        self.assertEqual([result["executed"] for result in self.results(client)], [True, False])

    def test_destructive_command_warns_before_preview_and_is_not_executed_when_rejected(self):
        command = "rm -rf important"
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            session, client = self.start([bash_call(command)])
            self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
            output = session.output.getvalue()
            self.assertIn("⚠ 高风险操作警告：可能具有破坏性的终端命令", output)
            self.assertIn("可能修改或删除文件", output)
            self.assertIn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", output)
            self.assertLess(output.index("高风险操作警告"), output.index("┌── 命令开始"))
            self.assertLess(output.index("│ " + command), output.index(CONFIRM_PROMPT))
            client.complete.assert_called_once()
            launch.assert_not_called()
            session.input.send("n\n")
            self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
            session.close()
            launch.assert_not_called()
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")

    def test_local_commands_and_invalid_answer_do_not_approve(self):
        session, client = self.start([bash_call("printf changed > marker.txt")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.input.send("/cost\n")
        self.assertTrue(session.output.wait_for("模型请求：1 次。"))
        session.input.send("/help\n")
        session.input.send("/compact\n")
        self.assertTrue(session.output.wait_for("暂时无法压缩"))
        self.assertEqual(session.output.getvalue().count("/help  查看帮助"), 2)
        session.input.send("yes\n")
        self.assertTrue(session.output.wait_for("正在等待本次命令执行确认"))
        self.assertFalse((self.root / "marker.txt").exists())
        client.complete.assert_called_once()
        session.input.send("n\n")
        self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
        messages = client.complete.call_args_list[1].kwargs["messages"]
        self.assertEqual([message["content"] for message in messages if message["role"] == "user"], ["请执行命令"])

    def test_eof_rejects_pending_and_later_commands_without_deadlock(self):
        session, client = self.start([
            bash_call("printf first > first.txt"),
            bash_call("printf second > second.txt", "bash_2"),
        ])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        session.close()
        self.assertFalse((self.root / "first.txt").exists())
        self.assertFalse((self.root / "second.txt").exists())
        self.assertEqual(session.output.getvalue().count(CONFIRM_PROMPT), 1)
        self.assertEqual([result["code"] for result in self.results(client)], ["confirmation_denied"] * 2)

    def test_eof_before_model_reply_rejects_late_command(self):
        requested = Event()
        release = Event()

        def initial_reply(**_):
            requested.set()
            if not release.wait(3):
                raise AssertionError("Model request was not released")
            return reply(None, tool_calls=[bash_call("printf changed > marker.txt")])

        responses = iter([initial_reply, lambda **_: reply("已取消命令。")])
        client = Mock(complete=Mock(side_effect=lambda **kwargs: next(responses)(**kwargs)))
        session = CLISession(client, lines=("请执行命令\n", ""), terminal=True, character_delay=0)
        try:
            self.assertTrue(requested.wait(3))
            self.assertTrue(session.input.eof_read.wait(3))
            release.set()
            session.join()
            self.assertFalse((self.root / "marker.txt").exists())
            self.assertNotIn(CONFIRM_PROMPT, session.output.getvalue())
            self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")
        finally:
            release.set()
            session.close()

    def test_exit_rejects_pending_command_and_stops_next_request(self):
        finished = Event()

        def tracked_query(state):
            try:
                return query_loop(state)
            finally:
                finished.set()

        with patch("harness.cli.query_loop", side_effect=tracked_query):
            session, client = self.start([bash_call("printf changed > marker.txt")])
            self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
            session.input.send("/exit\n")
            session.join()
            self.assertTrue(finished.is_set())
            self.assertFalse((self.root / "marker.txt").exists())
            client.complete.assert_called_once()

    def test_non_interactive_mode_rejects_without_prompting(self):
        session, client = self.start([bash_call("printf changed > marker.txt")], terminal=False)
        self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
        self.assertIn("非交互模式无法确认命令执行", session.output.getvalue())
        self.assertNotIn(CONFIRM_PROMPT, session.output.getvalue())
        self.assertFalse((self.root / "marker.txt").exists())
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")

    def test_exit_allows_running_command_to_clean_up(self):
        session, client, process = self.start_running_command()
        session.input.send("/cost\n")
        self.assertTrue(session.output.wait_for("模型请求：1 次。"))
        session.input.send("/exit\n")
        session.join()
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        client.complete.assert_called_once()

    def test_keyboard_interrupt_allows_running_command_to_clean_up(self):
        original_readline = QueuedInput.readline

        def readline(stream):
            line = original_readline(stream)
            if line == "interrupt\n":
                raise KeyboardInterrupt
            return line

        with patch.object(QueuedInput, "readline", readline):
            session, client, process = self.start_running_command()
            session.input.send("interrupt\n")
            session.join()
            self.assertIsNotNone(process.poll())
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
            self.assertIn("查询已停止", session.output.getvalue())
            client.complete.assert_called_once()

    def test_eof_waits_for_approved_command_with_finite_timeout(self):
        session, client, process = self.start_running_command(timeout=1)
        session.close()
        self.assertIsNotNone(process.poll())
        result = self.results(client)[0]
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["cancelled"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("命令请求处理完毕。", session.output.getvalue())

    def test_command_control_characters_are_escaped_without_changing_execution(self):
        content = "原文\x1b[2J\r末尾\u202e"
        session, client = self.start([bash_call(f"printf '%s' '{content}'")])
        self.assertTrue(session.output.wait_for(CONFIRM_PROMPT))
        output = visible_text(session.output.getvalue())
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\u202e", output)
        self.assertIn("原文\\u001b[2J\\u000d末尾\\u202e", output)
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("命令请求处理完毕。"))
        self.assertEqual(self.results(client)[0]["stdout"], content)


if __name__ == "__main__":
    unittest.main()
