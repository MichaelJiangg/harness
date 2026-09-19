import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from harness.permissions import PermissionPolicy
from harness.tools import ToolRegistry, create_tool_executor
from harness.tools.definition import ToolDefinition
from test_cli import CLISession, reply, visible_text


class PermissionCLITests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cwd = patch("harness.tools.executor.Path.cwd", return_value=Path(directory.name).resolve())
        cwd.start()
        self.addCleanup(cwd.stop)
        self.handler = Mock(return_value={"message": "示例工具已完成。"})
        registry = ToolRegistry()
        registry.register(ToolDefinition("example", "示例工具。", {
            "type": "object", "properties": {"value": {"type": "string"}},
            "required": ["value"], "additionalProperties": False,
        }), self.handler)
        for module in ("harness.tools.executor", "harness.tools.registry"):
            mocked = patch(module + ".REGISTRY", registry)
            mocked.start()
            self.addCleanup(mocked.stop)

    def start(self, *, policy=None, terminal=True, twice=False):
        if policy is not None:
            factory = patch("harness.cli.create_tool_executor", side_effect=lambda **kwargs:
                            create_tool_executor(permissions=policy, **kwargs))
            factory.start()
            self.addCleanup(factory.stop)
        self.call = {"id": "example_1", "type": "function", "function": {
            "name": "example", "arguments": json.dumps({"value": "中文\u001b[31m"}, ensure_ascii=False),
        }}
        responses = [reply(None, tool_calls=[self.call])]
        if twice:
            responses.append(reply(None, tool_calls=[{**self.call, "id": "example_2"}]))
        responses.append(reply("权限请求已处理。"))
        client = Mock(complete=Mock(side_effect=responses))
        session = CLISession(client, lines=("执行示例工具\n",), terminal=terminal, character_delay=0)
        self.addCleanup(session.close)
        return session, client

    def result(self, client):
        message = next(item for item in client.complete.call_args_list[1].kwargs["messages"]
                       if item["role"] == "tool")
        self.assertEqual(message["tool_call_id"], "example_1")
        return json.loads(message["content"])

    def test_new_tool_previews_parameters_and_requires_confirmation_each_time(self):
        session, client = self.start(twice=True)
        self.assertTrue(session.output.wait_for("[确认] 输入 y 批准本次工具调用"))
        self.handler.assert_not_called()
        output = visible_text(session.output.getvalue())
        self.assertIn("[工具确认] example", output)
        self.assertIn("风险等级：中风险", output)
        self.assertNotIn("高风险操作警告", output)
        self.assertIn('"value": "中文\\u001b[31m"', output)
        self.assertNotIn("\u001b", output)
        session.input.send("/cost\n")
        self.assertTrue(session.output.wait_for("模型请求：1 次。"))
        session.input.send("y\n")
        self.assertTrue(session.output.wait_for("[请求 #2"))
        # 第二轮已经返回 usage，随后会重新进入同一确认通道。
        with session.output.changed:
            self.assertTrue(session.output.changed.wait_for(
                lambda: session.output.getvalue().count("[工具确认] example") == 2, timeout=3))
        self.assertEqual(self.handler.call_count, 1)
        session.input.send("n\n")
        self.assertTrue(session.output.wait_for("权限请求已处理。"))
        session.close()
        self.assertTrue(self.result(client)["executed"])
        results = [json.loads(item["content"]) for item in client.complete.call_args_list[2].kwargs["messages"]
                   if item["role"] == "tool"]
        self.assertEqual(results[-1]["code"], "confirmation_denied")
        self.assertEqual(self.handler.call_count, 1)
        self.assertEqual(session.errors.getvalue(), "")

    def test_deny_returns_error_to_model_without_prompt_or_execution(self):
        session, client = self.start(policy=PermissionPolicy(allow=["example"], deny=["example"]))
        self.assertTrue(session.output.wait_for("权限请求已处理。"))
        session.close()
        self.handler.assert_not_called()
        self.assertEqual(self.result(client)["code"], "permission_denied")
        self.assertNotIn("[工具确认]", session.output.getvalue())
        self.assertEqual(session.errors.getvalue(), "")

    def test_explicit_allow_runs_without_prompt_in_noninteractive_mode(self):
        session, client = self.start(policy=PermissionPolicy(allow=["example"]), terminal=False)
        self.assertTrue(session.output.wait_for("权限请求已处理。"))
        session.close()
        self.handler.assert_called_once()
        self.assertTrue(self.result(client)["executed"])
        self.assertNotIn("[确认]", session.output.getvalue())

    def test_new_tool_in_noninteractive_mode_is_not_approved(self):
        session, client = self.start(terminal=False)
        self.assertTrue(session.output.wait_for("权限请求已处理。"))
        session.close()
        self.handler.assert_not_called()
        self.assertEqual(self.result(client)["code"], "confirmation_denied")
        self.assertIn("非交互模式无法确认工具调用", session.output.getvalue())

    def test_eof_while_waiting_for_generic_confirmation_does_not_execute(self):
        session, client = self.start()
        self.assertTrue(session.output.wait_for("[确认] 输入 y 批准本次工具调用"))
        session.input.send("")
        session.join()
        self.handler.assert_not_called()
        self.assertFalse(self.result(client)["executed"])


class BuiltinRiskCLITests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "sample.py").write_text("needle = 1\n", encoding="utf-8")
        cwd = patch("harness.tools.executor.Path.cwd", return_value=self.root)
        cwd.start()
        self.addCleanup(cwd.stop)

    def start(self, calls, *, policy=None):
        factory = patch("harness.cli.create_tool_executor", side_effect=lambda **kwargs:
                        create_tool_executor(permissions=policy, **kwargs))
        factory.start()
        self.addCleanup(factory.stop)
        tool_calls = [{"id": f"call_{index}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments),
        }} for index, (name, arguments) in enumerate(calls)]
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=tool_calls), reply("风险请求已处理。"),
        ]))
        session = CLISession(client, lines=("执行工具\n",), terminal=True, character_delay=0)
        self.addCleanup(session.close)
        return session, client

    def results(self, client):
        return [json.loads(message["content"])
                for message in client.complete.call_args_list[1].kwargs["messages"]
                if message["role"] == "tool"]

    def test_read_and_search_run_without_confirmation_by_default(self):
        session, client = self.start([
            ("read_file", {"path": "sample.py"}), ("grep", {"keyword": "needle"}),
        ])
        self.assertTrue(session.output.wait_for("风险请求已处理。"))
        session.close()
        self.assertNotIn("[确认]", session.output.getvalue())
        self.assertNotIn("高风险操作警告", session.output.getvalue())
        results = self.results(client)
        self.assertEqual([result["executed"] for result in results], [True, True])
        self.assertEqual(results[0]["content"], "needle = 1\n")
        self.assertEqual(results[1]["matches"][0]["path"], "sample.py")

    def test_explicit_ask_keeps_read_and_search_low_risk_but_requires_confirmation(self):
        for name, arguments in (("read_file", {"path": "sample.py"}), ("grep", {"keyword": "needle"})):
            with self.subTest(tool=name):
                session, client = self.start([(name, arguments)], policy=PermissionPolicy(ask=[name]))
                self.assertTrue(session.output.wait_for("[确认] 输入 y 批准本次工具调用"))
                self.assertIn("风险等级：低风险（只读；当前权限规则要求确认）", session.output.getvalue())
                self.assertNotIn("高风险操作警告", session.output.getvalue())
                client.complete.assert_called_once()
                session.input.send("n\n")
                self.assertTrue(session.output.wait_for("风险请求已处理。"))
                session.close()
                self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")

    def test_deny_bash_has_no_warning_no_confirmation_and_no_execution(self):
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            session, client = self.start([("bash", {"command": "pwd"})],
                                         policy=PermissionPolicy(allow=["bash"], deny=["bash"]))
            self.assertTrue(session.output.wait_for("风险请求已处理。"))
            session.close()
            launch.assert_not_called()
        self.assertNotIn("高风险操作警告", session.output.getvalue())
        self.assertNotIn("[确认]", session.output.getvalue())
        self.assertEqual(self.results(client)[0]["code"], "permission_denied")

    def test_simple_read_only_shell_command_executes_without_confirmation(self):
        session, client = self.start([("bash", {"command": "pwd"})])
        self.assertTrue(session.output.wait_for("风险请求已处理。"))
        session.close()
        self.assertNotIn("[确认]", session.output.getvalue())
        self.assertNotIn("高风险操作警告", session.output.getvalue())
        result = self.results(client)[0]
        self.assertTrue(result["executed"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), str(self.root))

    def test_explicit_ask_confirms_read_only_shell_command_without_high_risk_warning(self):
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            session, client = self.start([("bash", {"command": "pwd"})],
                                         policy=PermissionPolicy(ask=["bash"]))
            self.assertTrue(session.output.wait_for("[确认] 输入 y 批准本次命令执行"))
            output = session.output.getvalue()
            self.assertIn("风险等级：低风险（只读；当前权限规则要求确认）", output)
            self.assertNotIn("高风险操作警告", output)
            self.assertLess(output.index("风险等级：低风险"), output.index("│ pwd"))
            launch.assert_not_called()
            client.complete.assert_called_once()
            session.input.send("n\n")
            self.assertTrue(session.output.wait_for("风险请求已处理。"))
            session.close()
            launch.assert_not_called()
        self.assertEqual(self.results(client)[0]["code"], "confirmation_denied")


if __name__ == "__main__":
    unittest.main()
