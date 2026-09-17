from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from harness.engine import QueryState, query_loop
from harness.permissions import PermissionPolicy
from harness.tools import ToolRegistry, create_tool_executor, execute_tool
from harness.tools.definition import ToolDefinition
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


class ToolConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.handler = Mock(return_value={"message": "已完成。"})
        self.registry.register(ToolDefinition("write", "需要确认的操作。", {
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"], "additionalProperties": False,
        }), self.handler)
        registry_patch = patch("harness.tools.executor.REGISTRY", self.registry)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        self.arguments = {"path": "note.txt", "content": "待确认内容。"}
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.workspace = Path(workspace.name).resolve()

    def execute(self, **kwargs):
        return execute_tool("write", self.arguments, workspace=self.workspace, **kwargs)

    def test_missing_confirmation_callback_never_executes(self):
        result = self.execute()
        self.assertEqual(result["code"], "confirmation_required")
        self.assertFalse(result["executed"])
        self.handler.assert_not_called()

    def test_only_boolean_true_approves_each_independent_call(self):
        for approval in (False, None, "y", "true", 1, {}, [True]):
            with self.subTest(approval=approval):
                result = self.execute(confirm=Mock(return_value=approval))
                self.assertEqual(result["code"], "confirmation_denied")
                self.assertFalse(result["executed"])
        self.handler.assert_not_called()
        confirm = Mock(return_value=True)
        for _ in range(2):
            self.assertTrue(self.execute(confirm=confirm)["executed"])
        self.assertEqual(confirm.call_count, 2)
        self.assertEqual(self.handler.call_count, 2)
        confirm.assert_called_with("write", self.arguments, self.workspace)

    def test_invalid_arguments_and_model_approval_fields_rejected_before_confirmation(self):
        confirm = Mock(return_value=True)
        for arguments in ({}, {"path": "note", "content": None},
                          {**self.arguments, "approved": True}, {**self.arguments, "confirmed": True}):
            result = execute_tool("write", arguments, workspace=self.workspace, confirm=confirm)
            self.assertEqual(result["code"], "invalid_arguments")
        confirm.assert_not_called()
        self.handler.assert_not_called()

    def test_callback_error_is_returned_without_private_details_or_execution(self):
        result = self.execute(confirm=Mock(side_effect=RuntimeError("private callback details")))
        self.assertEqual(result["code"], "confirmation_failed")
        self.assertNotIn("private callback details", json.dumps(result))
        self.handler.assert_not_called()

    def test_execution_uses_snapshot_even_if_callback_mutates_arguments(self):
        expected = deepcopy(self.arguments)

        def confirm(name, arguments, workspace):
            self.assertEqual(arguments, expected)
            arguments["content"] = "not approved"
            self.arguments["path"] = "different.txt"
            return True

        self.assertTrue(self.execute(confirm=confirm)["executed"])
        self.handler.assert_called_once_with(expected, self.workspace)

    def test_abort_before_or_during_confirmation_prevents_execution(self):
        abort = Event()
        abort.set()
        confirm = Mock(return_value=True)
        self.assertEqual(self.execute(confirm=confirm, abort=abort)["code"], "cancelled")
        confirm.assert_not_called()
        abort.clear()

        def approve_then_cancel(*args):
            abort.set()
            return True

        self.assertEqual(self.execute(confirm=approve_then_cancel, abort=abort)["code"], "cancelled")
        self.handler.assert_not_called()

    def test_read_only_tools_do_not_request_confirmation(self):
        self.registry.register(ToolDefinition("read", "无副作用。", {"type": "object"}), self.handler)
        confirm = Mock(side_effect=AssertionError("不应询问"))
        self.assertTrue(execute_tool("read", {}, confirm=confirm,
                                     permissions=PermissionPolicy(allow=["read"]),
                                     workspace=self.workspace)["executed"])
        confirm.assert_not_called()

    def test_factory_binds_workspace_and_passes_confirmation_and_cancellation(self):
        confirm = Mock(return_value=True)
        abort = Event()
        with patch("harness.tools.executor.Path.cwd", return_value=self.workspace):
            executor = create_tool_executor(confirm=confirm, abort=abort)
        with patch("harness.tools.executor.Path.cwd", return_value=Path("/later-workspace")):
            self.assertTrue(executor("write", self.arguments)["executed"])
        confirm.assert_called_once_with("write", self.arguments, self.workspace)
        self.handler.assert_called_once_with(self.arguments, self.workspace)
        abort.set()
        self.assertEqual(executor("write", self.arguments)["code"], "cancelled")
        self.assertEqual(self.handler.call_count, 1)


class WriteFileIntegrationTests(unittest.TestCase):
    def test_approved_write_then_read_returns_results_with_matching_ids_and_usage(self):
        with TemporaryDirectory() as workspace:
            arguments = {"path": "notes/result.txt", "content": "第一行\r\n第二行。"}
            confirm = Mock(return_value=True)
            client = FakeClient([
                reply(None, [tool_call("write-1", "write_file", json.dumps(arguments, ensure_ascii=False))]),
                reply(None, [tool_call("read-1", "read_file", json.dumps({"path": arguments["path"]}))]),
                reply("已写入并检查内容。"),
            ])
            state = QueryState(client, UsageLedger(), tool_executor=create_tool_executor(workspace, confirm=confirm))
            state.messages.append({"role": "user", "content": "创建笔记"})
            self.assertEqual(query_loop(state), "已写入并检查内容。")
            written = client.requests[1]["messages"][-1]
            self.assertEqual(written["tool_call_id"], "write-1")
            self.assertTrue(json.loads(written["content"])["executed"])
            self.assertNotIn("content", json.loads(written["content"]))
            read = client.requests[2]["messages"][-1]
            self.assertEqual(read["tool_call_id"], "read-1")
            self.assertEqual(json.loads(read["content"])["content"], arguments["content"])
            self.assertEqual(Path(workspace, arguments["path"]).read_bytes(), arguments["content"].encode("utf-8"))
            confirm.assert_called_once_with("write_file", arguments, Path(workspace).resolve())
            self.assertEqual(state.ledger.summary()["requests"], 3)
            self.assertEqual(state.ledger.summary()["total_tokens"], 360)

    def test_denied_write_returns_error_to_model_without_creating_parent_directory(self):
        with TemporaryDirectory() as workspace:
            client = FakeClient([
                reply(None, [tool_call("denied-write", "write_file", '{"path":"new/note.txt","content":"正文"}')]),
                reply("您未批准，未写入文件。"),
            ])
            state = QueryState(client, UsageLedger(), tool_executor=create_tool_executor(workspace, confirm=lambda *args: False))
            state.messages.append({"role": "user", "content": "准备笔记"})
            self.assertEqual(query_loop(state), "您未批准，未写入文件。")
            result = client.requests[1]["messages"][-1]
            self.assertEqual(result["tool_call_id"], "denied-write")
            self.assertEqual(json.loads(result["content"])["code"], "confirmation_denied")
            self.assertFalse(Path(workspace, "new").exists())
            self.assertEqual(state.ledger.summary()["requests"], 2)


if __name__ == "__main__":
    unittest.main()
