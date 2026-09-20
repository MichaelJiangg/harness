from io import StringIO
import unittest
from unittest.mock import Mock

from harness.cli import _read_multiline_text, run_cli
from test_cli import reply, visible_text


class TerminalStream(StringIO):
    def isatty(self):
        return True


class MultilineInputTests(unittest.TestCase):
    def run_terminal(self, source, client):
        output = TerminalStream()
        run_cli(client, input_stream=TerminalStream(source), output=output)
        return output.getvalue()

    def test_multiline_input_sends_after_two_blank_lines_and_shows_line_numbers(self):
        requested = []

        def complete(**request):
            requested.append(request["messages"][-1]["content"])
            return reply("已收到。")

        client = Mock(complete=Mock(side_effect=complete))
        source = (
            "帮我审查下面这段代码：\\\n"
            "    result = eval(data)\n"
            "\n"
            "\n"
        )
        output = self.run_terminal(source, client)
        self.assertEqual(
            requested,
            ["帮我审查下面这段代码：\n    result = eval(data)"],
        )
        text = visible_text(output)
        self.assertIn("... 1│", text)
        self.assertIn("... 2│", text)
        self.assertIn("已收到。", text)

    def test_pasted_internal_blank_line_is_preserved(self):
        requested = []

        def complete(**request):
            requested.append(request["messages"][-1]["content"])
            return reply("已收到。")

        source = "第一行\\\n\n第二行\n\n\n"
        self.run_terminal(source, Mock(complete=Mock(side_effect=complete)))
        self.assertEqual(requested, ["第一行\n\n第二行"])

    def test_terminal_command_uses_double_blank_completion(self):
        client = Mock()
        output = TerminalStream()
        run_cli(client, input_stream=TerminalStream("/tools\n"), output=output)
        self.assertIn("Available tools (", output.getvalue())
        self.assertIn("read_file", output.getvalue())
        client.complete.assert_not_called()

    def test_pipe_input_keeps_single_line_behavior(self):
        requested = []

        def complete(**request):
            requested.append(request["messages"][-1]["content"])
            return reply("已收到。")

        output = StringIO()
        run_cli(
            Mock(complete=Mock(side_effect=complete)),
            input_stream=StringIO("普通问题\n"),
            output=output,
        )
        self.assertEqual(requested, ["普通问题"])
        self.assertNotIn("... 1│", output.getvalue())

    def test_terminal_single_line_sends_on_one_enter(self):
        requested = []

        def complete(**request):
            requested.append(request["messages"][-1]["content"])
            return reply("已收到。")

        self.run_terminal("普通问题\n", Mock(complete=Mock(side_effect=complete)))
        self.assertEqual(requested, ["普通问题"])

    def test_pending_input_detection_keeps_pasted_lines_multiline(self):
        lines = iter(["第一行", "", "第二行", "", ""])
        prompts = []
        result = _read_multiline_text(
            lambda _prompt: next(lines),
            lambda number: prompts.append(number),
            has_pending_input=lambda: True,
        )
        self.assertEqual(result, "第一行\n\n第二行")
        self.assertEqual(prompts, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
