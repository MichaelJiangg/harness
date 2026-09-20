import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import Mock, call, patch

from harness.cli import run_cli
from harness.client import APIError
from harness.tools import create_tool_executor
from harness.usage import UsageLedger
from test_cli import CLISession, reply, visible_text


class TerminalBuffer(StringIO):
    def __init__(self, text=""):
        super().__init__(text)
        self.writes = []

    def isatty(self):
        return True

    def write(self, value):
        self.writes.append(value)
        return super().write(value)


class DisplayTests(unittest.TestCase):
    def test_terminal_writes_each_character_with_default_delay(self):
        abort = Event()
        output = TerminalBuffer()

        def complete(**request):
            request["on_text"]("甲乙丙")
            return reply("甲乙丙")

        with patch("harness.cli.Event", return_value=abort), \
                patch.object(abort, "wait", return_value=False) as wait:
            run_cli(Mock(complete=complete), input_stream=TerminalBuffer("问题\n\n\n"), output=output)
        self.assertEqual(wait.call_args_list, [call(0.02), call(0.02), call(0.02)])
        self.assertIn("甲乙丙", output.getvalue())
        self.assertLess(output.getvalue().index("甲"), output.getvalue().index("乙"))
        self.assertLess(output.getvalue().index("乙"), output.getvalue().index("丙"))
        self.assertIn("Assistant", output.getvalue())

    def test_terminal_shows_spinner_and_tool_progress(self):
        tool_call = {
            "id": "call_read_1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "example.txt"}),
            },
        }
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=[tool_call]),
            reply("读取完成。"),
        ]))
        with TemporaryDirectory() as directory:
            Path(directory, "example.txt").write_text("abc", encoding="utf-8")
            with patch("harness.cli.create_tool_executor",
                       return_value=create_tool_executor(directory)):
                session = CLISession(
                    client, lines=("读取文件\n",),
                    terminal=True, character_delay=0,
                )
                self.addCleanup(session.close)
                self.assertTrue(session.output.wait_for("读取完成。"))
                session.close()
        output = visible_text(session.output.getvalue())
        self.assertIn("思考中...", output)
        self.assertIn("⚙ 执行工具：read_file", output)
        self.assertIn('"path": "example.txt"', output)
        self.assertIn("✓ 完成", output)

    def test_pipe_output_is_not_artificially_delayed(self):
        abort = Event()
        output = StringIO()

        def complete(**request):
            request["on_text"]("完整片段")
            return reply("完整片段")

        with patch("harness.cli.Event", return_value=abort), patch.object(abort, "wait") as wait:
            run_cli(Mock(complete=complete), input_stream=StringIO("问题\n"), output=output)
        wait.assert_not_called()
        self.assertEqual(output.getvalue().count("完整片段"), 1)

    def test_terminal_renders_markdown_headers_code_lists_and_links(self):
        abort = Event()
        output = TerminalBuffer()
        content = (
            "# 快速排序\n\n"
            "**平均复杂度**为 *O(n log n)*。\n\n"
            "```python\n"
            "def quicksort(items):\n"
            "    return items\n"
            "```\n\n"
            "- 稳定排序\n"
            "- 原地排序\n\n"
            "[官方文档](https://example.com)"
        )

        def complete(**request):
            request["on_text"](content)
            return reply(content)

        with patch("harness.cli.Event", return_value=abort), \
                patch.object(abort, "wait", return_value=False):
            run_cli(Mock(complete=complete), input_stream=TerminalBuffer("请解释\n\n\n"), output=output)
        text = visible_text(output.getvalue())
        self.assertIn("快速排序", text)
        self.assertIn("平均复杂度", text)
        self.assertIn("def quicksort", text)
        self.assertIn("稳定排序", text)
        self.assertIn("官方文档", text)

    def test_cost_and_exit_are_responsive_while_character_delay_is_waiting(self):
        abort = Event()
        waiting = Event()
        release = Event()
        workers = []

        def pause(delay):
            self.assertEqual(delay, 0.02)
            waiting.set()
            if not release.wait(3):
                raise AssertionError("Test did not release character delay")
            return abort.is_set()

        def create_worker(*args, **kwargs):
            worker = Thread(*args, **kwargs)
            workers.append(worker)
            return worker

        def complete(**request):
            request["on_text"]("甲乙丙")
            return reply("甲乙丙")

        with patch("harness.cli.Event", return_value=abort), \
                patch.object(abort, "wait", side_effect=pause), \
                patch("harness.cli.Thread", side_effect=create_worker):
            session = CLISession(Mock(complete=complete), terminal=True, lines=("问题\n",))
            try:
                self.assertTrue(waiting.wait(3))
                session.input.send("/cost\n")
                self.assertTrue(session.output.wait_for("尚未返回的请求用量将在返回后记录"))
                session.input.send("/exit\n")
                session.join()
                self.assertTrue(abort.is_set())
                release.set()
                workers[0].join(timeout=3)
                self.assertFalse(workers[0].is_alive())
                self.assertIn("甲", session.output.getvalue())
                self.assertNotIn("乙", session.output.getvalue())
                self.assertEqual(session.errors.getvalue(), "")
            finally:
                release.set()
                session.close()
                if workers:
                    workers[0].join(timeout=3)

    def test_partial_retry_is_marked_and_new_answer_starts_on_its_own_line(self):
        abort = Event()
        output = StringIO()
        ledger = UsageLedger()
        attempts = 0

        def complete(**request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                request["on_text"]("旧草稿")
                raise APIError("暂时断线", retryable=True)
            request["on_text"]("新回答")
            return reply("新回答")

        with patch("harness.cli.Event", return_value=abort), \
                patch.object(abort, "wait", return_value=False):
            run_cli(Mock(complete=complete), ledger=ledger, input_stream=StringIO("问题\n"), output=output)
        text = output.getvalue()
        self.assertLess(text.index("旧草稿"), text.index("上次输出未完成，重新生成"))
        self.assertLess(text.index("重新生成"), text.index("\nDeepSeek > 新回答"))
        self.assertNotIn("旧草稿新回答", text)
        self.assertEqual(text.count("新回答"), 1)
        self.assertEqual(ledger.summary()["requests"], 2)
        self.assertEqual(ledger.summary()["missing_usage_requests"], 1)


if __name__ == "__main__":
    unittest.main()
