import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from harness.cli import run_cli
from harness.engine import QueryState, query_loop
from harness.hooks import HookManager, HookResult
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


class HookManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        (self.workspace / ".harness").mkdir()

    def write_hooks(self, hooks):
        path = self.workspace / ".harness" / "hooks.json"
        path.write_text(json.dumps({"hooks": hooks}), encoding="utf-8")
        return path

    def test_shell_and_prompt_hooks_run_for_matching_events(self):
        self.write_hooks([
            {"event": "session_start", "type": "shell",
             "name": "开始记录", "command": "echo started"},
            {"event": "before_send_message", "type": "prompt",
             "name": "补充约束", "prompt": "回答保持简洁"},
        ])
        manager = HookManager(self.workspace)
        with patch("harness.hooks.subprocess.run", return_value=SimpleNamespace(
            stdout="started\n", stderr="",
        )) as run:
            start = manager.run("session_start", {})
        run.assert_called_once()
        self.assertIn("开始记录", start.outputs[0])
        before = manager.run("before_send_message", {})
        self.assertEqual(before.prompts, ["回答保持简洁"])

    def test_python_hook_imports_user_function_and_can_return_prompt(self):
        (self.workspace / ".harness" / "hook_functions.py").write_text(
            'def before_message(context):\n'
            '    return {"prompt": "优先使用中文", "message": "python hook ok"}\n',
            encoding="utf-8",
        )
        self.write_hooks([{
            "event": "before_send_message", "type": "python",
            "name": "python hook", "module": "hook_functions",
            "function": "before_message",
        }])
        manager = HookManager(self.workspace)
        result = manager.run("before_send_message", {})
        self.assertEqual(result.prompts, ["优先使用中文"])
        self.assertIn("python hook ok", result.outputs[0])

    def test_malformed_config_is_reported_without_loading_any_hooks(self):
        path = self.workspace / ".harness" / "hooks.json"
        path.write_text("{broken", encoding="utf-8")
        manager = HookManager(self.workspace)
        self.assertIn("不是有效 JSON", manager.load_error)
        self.assertEqual(manager.hooks, [])


class HookCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        (self.workspace / ".harness").mkdir()

    def test_session_and_send_hooks_run_without_calling_model(self):
        (self.workspace / ".harness" / "hooks.json").write_text(json.dumps({
            "hooks": [
                {"event": "session_start", "type": "shell",
                 "name": "start", "command": "echo start"},
                {"event": "before_send_message", "type": "prompt",
                 "name": "style", "prompt": "保持简洁"},
                {"event": "session_end", "type": "shell",
                 "name": "end", "command": "echo end"},
            ],
        }), encoding="utf-8")
        client = Mock(complete=Mock(return_value=reply("回答完成。")))
        output = StringIO()
        with patch("harness.cli.Path.cwd", return_value=self.workspace), \
                patch("harness.hooks.subprocess.run", return_value=SimpleNamespace(
                    stdout="ok\n", stderr="",
                )):
            run_cli(
                client, input_stream=StringIO("你好\n/exit\n"),
                output=output, memory_enabled=False, notes_enabled=False,
                hooks_enabled=True,
            )
        text = output.getvalue()
        self.assertIn("[hook] session_start", text)
        self.assertIn("[hook] session_end", text)
        first_messages = client.complete.call_args_list[0].kwargs["messages"]
        self.assertTrue(any(
            "保持简洁" in message.get("content", "")
            for message in first_messages if message.get("role") == "user"
        ))

    def test_engine_runs_tool_hooks_before_and_after_execution(self):
        manager = Mock()
        manager.run.side_effect = [
            HookResult(outputs=["before"]), HookResult(outputs=["after"]),
        ]
        responses = [
            reply(None, [tool_call("hook-read", "read_file", '{"path":"a.py"}')]),
            reply("读取完成。"),
        ]
        state = QueryState(FakeClient(responses), UsageLedger(), hooks=manager)
        state.messages.append({"role": "user", "content": "读取文件"})
        self.assertEqual(query_loop(state), "读取完成。")
        self.assertEqual(
            [call.args[0] for call in manager.run.call_args_list],
            ["before_tool", "after_tool"],
        )
        self.assertEqual(manager.run.call_args_list[1].args[1]["result"]["tool"], "read_file")


if __name__ == "__main__":
    unittest.main()
