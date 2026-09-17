import json
import unittest
from unittest.mock import patch

from harness.tools import TOOL_DEFINITIONS, execute_tool


class ToolTests(unittest.TestCase):
    def test_definitions_describe_only_placeholder_tools(self):
        self.assertEqual(
            [tool["function"]["name"] for tool in TOOL_DEFINITIONS],
            ["read_file", "run_command"],
        )
        for definition, parameter in zip(TOOL_DEFINITIONS, ("path", "command")):
            with self.subTest(parameter=parameter):
                self.assertEqual(definition["type"], "function")
                schema = definition["function"]["parameters"]
                self.assertEqual(schema["type"], "object")
                self.assertEqual(list(schema["properties"]), [parameter])
                self.assertEqual(schema["properties"][parameter]["type"], "string")
                self.assertEqual(schema["required"], [parameter])
                self.assertIs(schema["additionalProperties"], False)

    def test_read_file_does_not_read_the_requested_path(self):
        arguments = {"path": "/does-not-exist/harness-placeholder/file.txt"}
        with patch("builtins.open") as open_file:
            result = execute_tool("read_file", arguments)
        open_file.assert_not_called()
        self.assertEqual(
            result,
            {
                "status": "not_implemented",
                "executed": False,
                "tool": "read_file",
                "arguments": arguments,
                "message": "工具尚未实现，未执行任何操作。",
            },
        )
        self.assertIs(result["arguments"], arguments)

    def test_run_command_does_not_execute_the_requested_command(self):
        arguments = {"command": "harness-placeholder-command-that-does-not-exist"}
        with patch("subprocess.run") as run_command, patch("os.system") as system:
            result = execute_tool("run_command", arguments)
        run_command.assert_not_called()
        system.assert_not_called()
        self.assertEqual(
            result,
            {
                "status": "not_implemented",
                "executed": False,
                "tool": "run_command",
                "arguments": arguments,
                "message": "工具尚未实现，未执行任何操作。",
            },
        )
        self.assertIs(result["arguments"], arguments)

    def test_unknown_or_invalid_tool_names_return_errors(self):
        for name in ("delete_file", "__dict__", "", None, 123, [], {}):
            with self.subTest(name=name):
                result = execute_tool(name, {})
                self.assertEqual(result["status"], "error")
                self.assertIs(result["executed"], False)
                self.assertEqual(result["tool"], name if isinstance(name, str) else None)
                json.dumps(result)

    def test_invalid_parameters_return_errors(self):
        for name, parameter in (("read_file", "path"), ("run_command", "command")):
            invalid_arguments = (
                None,
                [],
                "text",
                42,
                True,
                {},
                {parameter: None},
                {parameter: 42},
                {parameter: True},
                {parameter: []},
                {parameter: {}},
                {parameter: "value", "extra": "unexpected"},
            )
            for arguments in invalid_arguments:
                with self.subTest(name=name, arguments=arguments):
                    result = execute_tool(name, arguments)
                    self.assertEqual(result["status"], "error")
                    self.assertIs(result["executed"], False)
                    self.assertIsInstance(result["message"], str)
                    json.dumps(result)

    def test_empty_strings_are_valid_under_the_declared_schema(self):
        for name, parameter in (("read_file", "path"), ("run_command", "command")):
            with self.subTest(name=name):
                result = execute_tool(name, {parameter: ""})
                self.assertEqual(result["status"], "not_implemented")
                self.assertIs(result["executed"], False)


if __name__ == "__main__":
    unittest.main()
