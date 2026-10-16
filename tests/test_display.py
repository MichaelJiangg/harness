import json
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import Mock, call, patch

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

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
    def test_mixed_tool_response_is_progress_then_a_single_final_answer(self):
        progress = "关键信息到手，继续核实 Instin"
        answer = "final-answer-marker"
        search = {"id": "search-2", "type": "function", "function": {
            "name": "web_search", "arguments": '{"query":"Instinct"}',
        }}
        requests = []

        def complete(**request):
            requests.append(request)
            content = progress if len(requests) == 1 else answer
            request["on_text"](content)
            return reply(content, tool_calls=[search] if len(requests) == 1 else None)

        def make_live(*args, **kwargs):
            return Live(*args, auto_refresh=False, **kwargs)

        execute = Mock(return_value={"status": "success", "executed": True, "tool": "web_search"})
        with patch.dict(os.environ, {"TERM": "xterm"}), \
                patch("harness.cli.Live", side_effect=make_live), \
                patch("harness.cli.Panel", wraps=Panel) as panels, \
                patch("harness.cli.create_tool_executor", return_value=execute):
            session = CLISession(Mock(complete=complete), terminal=True,
                                 character_delay=0, lines=("研究\n",))
            try:
                self.assertTrue(session.output.wait_for(answer))
                output_text = visible_text(session.output.getvalue())
                session.input.send("/activity\n")
                self.assertTrue(session.output.wait_for("progress"))
                session.close()
            finally:
                session.close()
        self.assertIn("过程进度", output_text)
        self.assertIn("继续执行工具", output_text)
        self.assertLess(output_text.index(progress), output_text.index("⚙ 执行工具：web_search"))
        self.assertLess(output_text.index("⚙ 执行工具：web_search"), output_text.index(answer))
        self.assertEqual(output_text.count(progress), 1)
        self.assertEqual(output_text.count(answer), 1)
        titles = [item.kwargs.get("title") for item in panels.call_args_list]
        self.assertIn("生成中（未完成）", titles)
        self.assertIn("过程进度", titles)
        self.assertEqual(titles.count("Assistant"), 1)
        self.assertIn(progress, requests[1]["messages"][2]["content"])
        self.assertEqual(execute.call_count, 1)

    def test_pipe_marks_preview_progress_and_completion_without_repeating_text(self):
        output = StringIO()
        attempts = 0
        search = {"id": "search-2", "type": "function", "function": {
            "name": "web_search", "arguments": '{"query":"公开资料"}',
        }}

        def complete(**request):
            nonlocal attempts
            attempts += 1
            text = "process-pipe-marker" if attempts == 1 else "final-pipe-marker"
            request["on_text"](text)
            return reply(text, tool_calls=[search] if attempts == 1 else None)

        execute = Mock(return_value={"status": "success", "executed": True, "tool": "web_search"})
        with patch("harness.cli.create_tool_executor", return_value=execute):
            run_cli(Mock(complete=complete), input_stream=StringIO("研究\n"), output=output)
        text = output.getvalue()
        self.assertEqual(text.count("process-pipe-marker"), 1)
        self.assertEqual(text.count("final-pipe-marker"), 1)
        self.assertIn("生成中（未完成）", text)
        self.assertIn("[过程进度] 继续执行工具。", text)
        self.assertIn("[回答完成]", text)
        self.assertNotIn("\x1b", text)

    def test_failed_stream_is_preserved_as_incomplete_without_an_assistant_panel(self):
        for failure in ("length", "disconnect"):
            with self.subTest(failure=failure):
                output = TerminalBuffer()
                errors = StringIO()

                def complete(**request):
                    request["on_text"]("unfinished-marker")
                    if failure == "disconnect":
                        raise APIError("连接中断，本轮未完成。")
                    result = reply("unfinished-marker")
                    result["choices"][0]["finish_reason"] = "length"
                    return result

                def make_live(*args, **kwargs):
                    return Live(*args, auto_refresh=False, **kwargs)

                with patch.dict(os.environ, {"TERM": "xterm"}), \
                        patch("harness.cli.Live", side_effect=make_live), \
                        patch("harness.cli.Panel", wraps=Panel) as panels:
                    run_cli(Mock(complete=complete), input_stream=TerminalBuffer("问题\n\n\n"),
                            output=output, error_output=errors, character_delay=0)
                text = visible_text(output.getvalue())
                self.assertEqual(text.count("unfinished-marker"), 1)
                self.assertIn("未完成", text)
                self.assertIn("本轮未完成", visible_text(errors.getvalue()))
                self.assertNotIn("Assistant", [item.kwargs.get("title") for item in panels.call_args_list])

    def test_child_response_end_events_remain_silent(self):
        def query(state):
            state.on_event({"type": "response_start"})
            for agent in ("delegate", "background", "swarm"):
                for kind in ("tool_calls", "final"):
                    state.on_event({"type": "response_end", "kind": kind,
                                    "content": f"hidden-{agent}-{kind}", "agent": agent,
                                    "finish_reason": "tool_calls" if kind == "tool_calls" else "stop"})
            state.on_event({"type": "text", "text": "主回答。"})
            state.on_event({"type": "response_end", "kind": "final",
                            "content": "主回答。", "finish_reason": "stop"})
            return "主回答。"

        output = StringIO()
        with patch("harness.cli.query_loop", side_effect=query):
            run_cli(Mock(), input_stream=StringIO("问题\n"), output=output)
        self.assertNotIn("hidden-", output.getvalue())
        self.assertEqual(output.getvalue().count("主回答。"), 1)
        self.assertIn("[回答完成]", output.getvalue())

    def test_child_retry_keeps_main_preview_active_without_repeating_text(self):
        def query(state):
            state.on_event({"type": "response_start"})
            state.on_event({"type": "text", "text": "main-prefix-"})
            for agent in ("delegate", "background", "swarm"):
                state.on_event({"type": "retry", "agent": agent, "partial": True,
                                "message": "child-retry-marker", "delay": 0, "attempt": 1})
            state.on_event({"type": "text", "text": "main-suffix"})
            state.on_event({"type": "response_end", "kind": "final",
                            "content": "main-prefix-main-suffix", "finish_reason": "stop"})
            return "main-prefix-main-suffix"

        def make_live(*args, **kwargs):
            return Live(*args, auto_refresh=False, **kwargs)

        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                output = TerminalBuffer() if terminal else StringIO()
                input_stream = TerminalBuffer("问题\n\n\n") if terminal else StringIO("问题\n")
                with patch.dict(os.environ, {"TERM": "xterm"}), \
                        patch("harness.cli.query_loop", side_effect=query), \
                        patch("harness.cli.Live", side_effect=make_live), \
                        patch("harness.cli.Panel", wraps=Panel) as panels:
                    run_cli(Mock(), input_stream=input_stream, output=output, character_delay=0)
                text = visible_text(output.getvalue())
                self.assertEqual(text.count("main-prefix-"), 1)
                self.assertEqual(text.count("main-prefix-main-suffix"), 1)
                self.assertNotIn("child-retry-marker", text)
                titles = [item.kwargs.get("title") for item in panels.call_args_list]
                self.assertNotIn("未完成", titles)
                if terminal:
                    previews = [item.args[0].markup for item in panels.call_args_list
                                if item.kwargs.get("title") == "生成中（未完成）"]
                    self.assertEqual(previews[-1], "main-prefix-main-suffix")
                    self.assertEqual(titles.count("Assistant"), 1)
                else:
                    self.assertEqual(text.count("[回答完成]"), 1)

    def test_tool_truncation_preserves_terminal_success_and_failure(self):
        for status, marker in (("success", "✓ 完成"), ("error", "✗ 失败")):
            with self.subTest(status=status):
                output = TerminalBuffer()
                tool_call = {"id": "search-1", "type": "function", "function": {
                    "name": "web_search", "arguments": '{"query":"研究"}',
                }}
                client = Mock(complete=Mock(side_effect=[
                    reply(None, tool_calls=[tool_call]), reply("处理完成。"),
                ]))
                result = {"status": status, "executed": True, "tool": "web_search",
                          "message": "很长的结果" * 4000}
                with patch("harness.cli.create_tool_executor", return_value=lambda *args: result):
                    run_cli(client, input_stream=TerminalBuffer("研究\n\n\n"),
                            output=output, character_delay=0)
                text = visible_text(output.getvalue())
                self.assertIn(marker, text)
                self.assertIn("已截断", text)
                returned = json.loads(next(message["content"]
                                           for message in client.complete.call_args_list[1].kwargs["messages"]
                                           if message["role"] == "tool"))
                self.assertEqual(returned["status"], status)

    def test_structurally_trimmed_search_reports_success_and_trimming(self):
        output = TerminalBuffer()
        tool_call = {"id": "search-1", "type": "function", "function": {
            "name": "web_search", "arguments": '{"query":"研究"}',
        }}
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=[tool_call]), reply("处理完成。"),
        ]))
        result = {"status": "success", "executed": True, "tool": "web_search",
                  "message": "搜索完成，返回 10 条结果。", "truncated": True}
        with patch("harness.cli.create_tool_executor", return_value=lambda *args: result):
            run_cli(client, input_stream=TerminalBuffer("研究\n\n\n"),
                    output=output, character_delay=0)
        text = visible_text(output.getvalue())
        self.assertIn("✓ 完成", text)
        self.assertIn("结果已裁剪", text)

    def test_long_streamed_answer_is_printed_once_when_preview_stops(self):
        output = TerminalBuffer()
        content = "\n\n".join(
            f"第 {index} 段长回答。" for index in range(30)
        ) + "\n\nanswer-final-marker"

        def complete(**request):
            request["on_text"](content)
            return reply(content)

        def make_console(**kwargs):
            return Console(width=80, height=8, **kwargs)

        def make_live(*args, **kwargs):
            return Live(*args, auto_refresh=False, **kwargs)

        with patch.dict(os.environ, {"TERM": "xterm"}), \
                patch("harness.cli.Console", side_effect=make_console), \
                patch("harness.cli.Live", side_effect=make_live):
            run_cli(
                Mock(complete=complete), input_stream=TerminalBuffer("问题\n\n\n"),
                output=output, character_delay=0,
            )
        text = visible_text(output.getvalue())
        self.assertEqual(text.count("answer-final-marker"), 1)
        for index in range(30):
            self.assertEqual(text.count(f"第 {index} 段长回答。"), 1)

    def test_long_chinese_table_cells_wrap_without_ellipsis(self):
        cell = "正文" + "".join(chr(0x4e00 + index) for index in range(80)) + "末尾"
        content = f"| 说明 |\n| --- |\n| {cell} |"
        for width in (40, 80):
            with self.subTest(width=width):
                output = TerminalBuffer()

                def make_console(**kwargs):
                    return Console(width=width, **kwargs)

                with patch.dict(os.environ, {"TERM": "xterm"}), \
                        patch("harness.cli.Console", side_effect=make_console):
                    run_cli(
                        Mock(complete=Mock(return_value=reply(content))),
                        input_stream=TerminalBuffer("问题\n\n\n"),
                        output=output, character_delay=0,
                    )
                text = visible_text(output.getvalue())
                normalized = "".join(
                    character for character in text
                    if not character.isspace() and character != "│"
                )
                self.assertIn(cell, normalized)
                self.assertNotIn("…", text)

    def test_narrow_terminal_keeps_long_paragraph_and_code_block(self):
        paragraph = "开头" + "长中文段落需要完整换行显示" * 10 + "段落末尾"
        code = 'payload = "' + "abcdefghijklmnopqrstuvwxyz" * 4 + '"'
        content = paragraph + "\n\n```python\n" + code + "\n```"
        for width in (40, 80):
            with self.subTest(width=width):
                output = TerminalBuffer()

                def make_console(**kwargs):
                    return Console(width=width, **kwargs)

                with patch.dict(os.environ, {"TERM": "xterm"}), \
                        patch("harness.cli.Console", side_effect=make_console):
                    run_cli(
                        Mock(complete=Mock(return_value=reply(content))),
                        input_stream=TerminalBuffer("问题\n\n\n"),
                        output=output, character_delay=0,
                    )
                normalized = "".join(
                    character for character in visible_text(output.getvalue())
                    if not character.isspace() and character != "│"
                )
                self.assertIn(paragraph, normalized)
                self.assertIn(code.replace(" ", ""), normalized)

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

    def test_terminal_shows_tool_progress(self):
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
        self.assertIn("Assistant", output)
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
        self.assertLess(text.index("重新生成"), text.index("\nDeepSeek > [生成中（未完成）] 新回答"))
        self.assertNotIn("旧草稿新回答", text)
        self.assertEqual(text.count("新回答"), 1)
        self.assertEqual(ledger.summary()["requests"], 2)
        self.assertEqual(ledger.summary()["missing_usage_requests"], 1)


if __name__ == "__main__":
    unittest.main()
