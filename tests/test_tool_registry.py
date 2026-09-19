import json
from pathlib import Path
from pkgutil import ModuleInfo
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from harness.permissions import PermissionPolicy
from harness.tools import REGISTRY, ToolRegistry, execute_tool, get_tool_definitions
from harness.tools.definition import ToolDefinition


def definition(name="example"):
    return ToolDefinition(name, "示例工具。", {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    })


class ToolRegistryTests(unittest.TestCase):
    def test_package_discovers_tools_and_definitions_are_fresh_snapshots(self):
        registered = REGISTRY.get("read_file")
        self.assertIsNotNone(registered)
        tool, handler = registered
        self.assertEqual(tool.name, "read_file")
        self.assertTrue(callable(handler))
        descriptions = get_tool_definitions()
        write_tool, write_handler = REGISTRY.get("write_file")
        self.assertTrue(callable(write_handler))
        bash_tool, bash_handler = REGISTRY.get("bash")
        self.assertTrue(callable(bash_handler))
        self.assertTrue(bash_tool.supports_cancellation)
        search_tool, search_handler = REGISTRY.get("grep")
        self.assertTrue(callable(search_handler))
        self.assertTrue(search_tool.supports_cancellation)
        delegate_tool, delegate_handler = REGISTRY.get("delegate")
        self.assertTrue(callable(delegate_handler))
        background_check_tool, _ = REGISTRY.get("background_check")
        background_submit_tool, _ = REGISTRY.get("background_submit")
        swarm_tool, _ = REGISTRY.get("swarm")
        run_verify_tool, _ = REGISTRY.get("run_verify")
        notes_append_tool, _ = REGISTRY.get("notes_append")
        notes_read_tool, _ = REGISTRY.get("notes_read")
        notes_replace_tool, _ = REGISTRY.get("notes_replace")
        self.assertEqual(descriptions, [
            background_check_tool.to_deepseek(), background_submit_tool.to_deepseek(),
            bash_tool.to_deepseek(), delegate_tool.to_deepseek(), search_tool.to_deepseek(),
            notes_append_tool.to_deepseek(), notes_read_tool.to_deepseek(),
            notes_replace_tool.to_deepseek(), tool.to_deepseek(),
            run_verify_tool.to_deepseek(), swarm_tool.to_deepseek(), write_tool.to_deepseek(),
        ])
        self.assertNotIn("requires_confirmation", json.dumps(descriptions))
        self.assertNotIn("supports_cancellation", json.dumps(descriptions))
        json.dumps(descriptions)
        descriptions[8]["function"]["parameters"]["properties"].clear()
        self.assertIn("path", get_tool_definitions()[8]["function"]["parameters"]["properties"])
        self.assertIn("path", tool.input_schema["properties"])

    def test_register_maps_name_to_definition_and_handler_and_rejects_duplicates(self):
        registry = ToolRegistry()
        tool = definition()
        handler = Mock()
        registry.register(tool, handler)
        self.assertEqual(registry.get("example"), (tool, handler))
        self.assertIsNone(registry.get("missing"))
        self.assertEqual(registry.definitions(), [tool.to_deepseek()])
        with self.assertRaises(ValueError):
            registry.register(definition(), Mock())
        self.assertEqual(registry.get("example"), (tool, handler))

    def test_invalid_registration_cannot_enter_registry(self):
        for tool, handler in (
            (None, Mock()),
            ({"name": "example"}, Mock()),
            (definition(""), Mock()),
            (definition(" \t"), Mock()),
            (definition(), None),
            (definition(), "not callable"),
        ):
            with self.subTest(tool=tool, handler=handler):
                registry = ToolRegistry()
                with self.assertRaises(ValueError):
                    registry.register(tool, handler)
                self.assertEqual(registry.definitions(), [])

    def test_discovery_skips_infrastructure_helpers_and_subpackages(self):
        modules = [
            ModuleInfo(None, "registry", False),
            ModuleInfo(None, "definition", False),
            ModuleInfo(None, "executor", False),
            ModuleInfo(None, "_helper", False),
            ModuleInfo(None, "__init__", False),
            ModuleInfo(None, "nested", True),
            ModuleInfo(None, "z_reader", False),
            ModuleInfo(None, "a_reader", False),
        ]
        first = SimpleNamespace(DEFINITION=definition("first"), execute=Mock())
        second = SimpleNamespace(DEFINITION=definition("second"), execute=Mock())
        implementations = {"harness.tools.a_reader": first, "harness.tools.z_reader": second}
        registry = ToolRegistry()
        with patch("harness.tools.registry.iter_modules", return_value=modules), \
                patch("harness.tools.registry.import_module", side_effect=implementations.__getitem__) as importer:
            registry.discover()
        self.assertEqual(importer.call_args_list, [call("harness.tools.a_reader"), call("harness.tools.z_reader")])
        self.assertEqual(registry.get("first"), (first.DEFINITION, first.execute))
        self.assertEqual(registry.get("second"), (second.DEFINITION, second.execute))
        self.assertIsNone(registry.get("a_reader"))
        self.assertEqual([item["function"]["name"] for item in registry.definitions()], ["first", "second"])

    def test_discovery_reports_malformed_tool_modules(self):
        modules = [ModuleInfo(None, "broken", False)]
        for implementation in (
            SimpleNamespace(execute=Mock()),
            SimpleNamespace(DEFINITION=definition()),
            SimpleNamespace(DEFINITION=definition(), execute=False),
        ):
            with self.subTest(implementation=implementation):
                registry = ToolRegistry()
                with patch("harness.tools.registry.iter_modules", return_value=modules), \
                        patch("harness.tools.registry.import_module", return_value=implementation):
                    with self.assertRaises(ValueError):
                        registry.discover()
                self.assertEqual(registry.definitions(), [])

    def test_discovery_reports_same_name_in_different_modules(self):
        modules = [ModuleInfo(None, "first", False), ModuleInfo(None, "second", False)]
        implementations = [
            SimpleNamespace(DEFINITION=definition("shared"), execute=Mock()),
            SimpleNamespace(DEFINITION=definition("shared"), execute=Mock()),
        ]
        registry = ToolRegistry()
        with patch("harness.tools.registry.iter_modules", return_value=modules), \
                patch("harness.tools.registry.import_module", side_effect=implementations):
            with self.assertRaises(ValueError):
                registry.discover()
        self.assertEqual(registry.get("shared"), (implementations[0].DEFINITION, implementations[0].execute))


class ToolExecutorValidationTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.handler = Mock(return_value={"content": "selected lines"})
        self.registry.register(definition(), self.handler)
        self.registry_patch = patch("harness.tools.executor.REGISTRY", self.registry)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.workspace = Path(workspace.name).resolve()
        self.permissions = PermissionPolicy(allow=["example", "long_running"])

    def test_schema_validation_rejects_bad_arguments_before_execution(self):
        invalid = [None, [], "path", 1, True, {}, {"path": ""}, {"path": 1},
                   {"path": "file", "extra": True}]
        for value in (True, False, -1, 1.5, "0", None, [], {}):
            invalid.append({"path": "file", "offset": value})
        for value in (True, False, 0, -1, 1.5, "1", None, [], {}):
            invalid.append({"path": "file", "limit": value})
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                result = execute_tool("example", arguments, workspace=self.workspace,
                                      permissions=self.permissions)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["code"], "invalid_arguments")
                self.assertFalse(result["executed"])
                self.assertTrue(result["message"])
        self.handler.assert_not_called()

    def test_optional_arguments_are_passed_to_handler_without_changing_call(self):
        for arguments in ({"path": "file"}, {"path": "file", "offset": 0, "limit": 1},
                          {"path": "file", "offset": 50, "limit": 100}):
            with self.subTest(arguments=arguments):
                original = dict(arguments)
                result = execute_tool("example", arguments, workspace=self.workspace,
                                      permissions=self.permissions)
                self.handler.assert_called_with(original, self.workspace)
                self.assertEqual(arguments, original)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["tool"], "example")
                self.assertTrue(result["executed"])
                self.assertEqual(result["content"], "selected lines")

    def test_unknown_tool_does_not_execute_a_registered_handler(self):
        result = execute_tool("missing", {"path": "file"}, workspace=self.workspace)
        self.assertEqual(result["code"], "unknown_tool")
        self.handler.assert_not_called()

    def test_integer_upper_bound_is_checked_before_confirmation(self):
        self.registry.register(ToolDefinition("bounded", "有限超时。", {
            "type": "object", "properties": {"timeout": {"type": "integer", "minimum": 1, "maximum": 120}},
            "required": ["timeout"], "additionalProperties": False,
        }), self.handler)
        confirm = Mock(return_value=True)
        for value in (0, 121, True, 1.5, "30"):
            with self.subTest(value=value):
                result = execute_tool("bounded", {"timeout": value}, confirm=confirm,
                                      workspace=self.workspace)
                self.assertEqual(result["code"], "invalid_arguments")
        confirm.assert_not_called()
        self.handler.assert_not_called()
        for value in (1, 120):
            self.assertTrue(execute_tool("bounded", {"timeout": value}, confirm=confirm,
                                         workspace=self.workspace)["executed"])
        self.assertEqual(self.handler.call_count, 2)

    def test_started_tool_failure_is_preserved_as_executed(self):
        self.handler.return_value = {"status": "error", "exit_code": 7, "stdout": "部分输出", "stderr": "错误输出"}
        result = execute_tool("example", {"path": "file"}, workspace=self.workspace,
                              permissions=self.permissions)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["executed"])
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["stderr"], "错误输出")

    def test_only_cancellation_capable_tools_receive_abort_keyword(self):
        from threading import Event

        abort = Event()
        self.registry.register(ToolDefinition("long_running", "可以取消。", {"type": "object"},
                                              supports_cancellation=True), self.handler)
        execute_tool("long_running", {}, workspace=self.workspace, abort=abort,
                     permissions=self.permissions)
        self.handler.assert_called_once_with({}, self.workspace, abort=abort)
        self.handler.reset_mock()
        execute_tool("example", {"path": "file"}, workspace=self.workspace, abort=abort,
                     permissions=self.permissions)
        self.handler.assert_called_once_with({"path": "file"}, self.workspace)


if __name__ == "__main__":
    unittest.main()
