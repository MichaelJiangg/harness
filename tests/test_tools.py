import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.tools import ToolRegistry, create_tool_executor, execute_tool, get_tool_definitions
from harness.tools.definition import ToolDefinition
from harness.tools.executor import ToolError
from harness.tools.read_file import MAX_FILE_BYTES


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def read(self, path):
        return execute_tool("read_file", {"path": str(path)}, workspace=self.workspace)

    def assert_error(self, result, code, tool="read_file"):
        self.assertEqual(result["status"], "error")
        self.assertIs(result["executed"], False)
        self.assertEqual(result["tool"], tool)
        self.assertEqual(result["code"], code)
        self.assertTrue(result["message"])
        self.assertNotIn("content", result)
        json.dumps(result)

    def test_definitions_describe_registered_tools(self):
        definitions = get_tool_definitions()
        self.assertEqual([tool["function"]["name"] for tool in definitions], ["bash", "grep", "read_file", "write_file"])
        definition = next(tool for tool in definitions if tool["function"]["name"] == "read_file")
        self.assertEqual(definition["type"], "function")
        self.assertTrue(definition["function"]["description"])
        schema = definition["function"]["parameters"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["properties"]), {"path", "offset", "limit"})
        self.assertEqual(schema["properties"]["path"]["type"], "string")
        self.assertEqual(schema["properties"]["path"]["minLength"], 1)
        self.assertEqual(schema["properties"]["offset"]["type"], "integer")
        self.assertEqual(schema["properties"]["offset"]["minimum"], 0)
        self.assertEqual(schema["properties"]["limit"]["type"], "integer")
        self.assertEqual(schema["properties"]["limit"]["minimum"], 1)
        self.assertEqual(schema["required"], ["path"])
        self.assertIs(schema["additionalProperties"], False)
        json.dumps(definitions)

    def test_new_tool_is_dispatched_through_the_registry(self):
        handler = Mock(return_value={"content": "工具输出", "message": "已完成。"})
        definition = ToolDefinition("example", "示例工具。", {"type": "object"})
        registry = ToolRegistry()
        registry.register(definition, handler)
        arguments = {"value": "参数"}
        with patch("harness.tools.executor.REGISTRY", registry):
            result = execute_tool(definition.name, arguments, workspace=self.workspace)
        handler.assert_called_once_with(arguments, self.workspace)
        self.assertEqual(result["status"], "success")
        self.assertIs(result["executed"], True)
        self.assertEqual(result["tool"], "example")
        self.assertEqual(result["content"], "工具输出")

    def test_definition_contains_no_callable_and_cannot_modify_tool_schema(self):
        parameters = {"type": "object", "properties": {"value": {"type": "string"}}}
        tool = ToolDefinition("example", "示例工具。", parameters)
        definition = tool.to_deepseek()
        self.assertEqual(definition["function"]["name"], "example")
        self.assertEqual(definition["function"]["description"], "示例工具。")
        json.dumps(definition)
        definition["function"]["parameters"]["properties"]["value"]["type"] = "number"
        self.assertEqual(parameters["properties"]["value"]["type"], "string")
        self.assertEqual(tool.to_deepseek()["function"]["parameters"], parameters)

    def test_known_tool_errors_are_returned_without_raising(self):
        handler = Mock(side_effect=ToolError("example_error", "可修复的错误。"))
        registry = ToolRegistry()
        registry.register(ToolDefinition("example", "示例工具。", {"type": "object"}), handler)
        with patch("harness.tools.executor.REGISTRY", registry):
            result = execute_tool("example", {}, workspace=self.workspace)
        self.assert_error(result, "example_error", tool="example")
        self.assertEqual(result["message"], "可修复的错误。")

    def test_unexpected_handler_errors_do_not_leak_exception_details(self):
        handler = Mock(side_effect=RuntimeError("private error details"))
        registry = ToolRegistry()
        registry.register(ToolDefinition("example", "示例工具。", {"type": "object"}), handler)
        with patch("harness.tools.executor.REGISTRY", registry):
            result = execute_tool("example", {}, workspace=self.workspace)
        self.assert_error(result, "execution_error", tool="example")
        self.assertNotIn("private error details", json.dumps(result))

    def test_unknown_or_invalid_tool_names_return_errors(self):
        for name in ("run_command", "delete_file", "__dict__", "", None, 123, [], {}):
            with self.subTest(name=name):
                result = execute_tool(name, {}, workspace=self.workspace)
                self.assert_error(result, "unknown_tool", tool=name if isinstance(name, str) else None)

    def test_invalid_parameters_return_errors_without_opening_a_file(self):
        invalid_arguments = (
            None, [], "text", 42, True, {},
            {"path": None}, {"path": 42}, {"path": True},
            {"path": []}, {"path": {}}, {"path": ""}, {"path": " \t\n"},
            {"path": "bad\x00path"}, {"path": "file.txt", "extra": "unexpected"},
        )
        with patch.object(Path, "open") as open_file:
            for arguments in invalid_arguments:
                with self.subTest(arguments=arguments):
                    result = execute_tool("read_file", arguments, workspace=self.workspace)
                    self.assert_error(result, "invalid_arguments")
        open_file.assert_not_called()

    def test_reads_chinese_text_and_preserves_original_newlines(self):
        path = self.workspace / "说明.txt"
        content = "第一行\r\n第二行\n结束。"
        path.write_bytes(content.encode("utf-8"))
        result = self.read("说明.txt")
        self.assertEqual(result["status"], "success")
        self.assertIs(result["executed"], True)
        self.assertEqual(result["tool"], "read_file")
        self.assertEqual(result["path"], "说明.txt")
        self.assertEqual(result["content"], content)
        self.assertTrue(result["message"])
        self.assertNotIn(content, result["message"])

    def test_empty_file_is_a_success(self):
        (self.workspace / "empty.txt").write_bytes(b"")
        result = self.read("empty.txt")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "")

    def test_relative_and_absolute_paths_within_workspace_are_supported(self):
        directory = self.workspace / "notes"
        directory.mkdir()
        path = directory / "has spaces.txt"
        path.write_text("内容", encoding="utf-8")
        for requested in ("notes/has spaces.txt", path):
            with self.subTest(requested=requested):
                result = self.read(requested)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["path"], "notes/has spaces.txt")
                self.assertEqual(result["content"], "内容")

    def test_missing_file_returns_not_found(self):
        self.assert_error(self.read("missing.txt"), "not_found")

    def test_directory_and_path_under_file_are_not_files(self):
        (self.workspace / "regular.txt").write_text("内容", encoding="utf-8")
        for path in (".", "regular.txt/child.txt"):
            with self.subTest(path=path):
                self.assert_error(self.read(path), "not_a_file")

    def test_fifo_is_rejected_before_opening(self):
        os.mkfifo(self.workspace / "pipe")
        with patch.object(Path, "open") as open_file:
            result = self.read("pipe")
        self.assert_error(result, "not_a_file")
        open_file.assert_not_called()

    def test_permission_failure_returns_error_without_raw_exception(self):
        (self.workspace / "private.txt").write_bytes(b"example")
        with patch.object(Path, "open", side_effect=PermissionError("private error details")):
            result = self.read("private.txt")
        self.assert_error(result, "permission_denied")
        self.assertNotIn("private error details", json.dumps(result))

    def test_read_failure_returns_read_error(self):
        (self.workspace / "file.txt").write_bytes(b"example")
        with patch.object(Path, "open", side_effect=OSError("private error details")):
            result = self.read("file.txt")
        self.assert_error(result, "read_error")
        self.assertNotIn("private error details", json.dumps(result))

    def test_invalid_utf8_and_binary_content_are_rejected(self):
        for data in (b"\xff\xfe", b"valid prefix\x00binary"):
            with self.subTest(data=data):
                (self.workspace / "binary.dat").write_bytes(data)
                self.assert_error(self.read("binary.dat"), "invalid_encoding")

    def test_sensitive_paths_are_rejected_without_accessing_contents(self):
        for path in (".env", ".env.local", "nested/.env.production", ".git/config", ".git", ".ENV", ".GIT/config"):
            with self.subTest(path=path), patch.object(Path, "open") as open_file:
                self.assert_error(self.read(path), "access_denied")
                open_file.assert_not_called()

    def test_parent_and_absolute_outside_paths_are_rejected(self):
        for path in ("../outside.txt", self.root / "outside.txt"):
            with self.subTest(path=path), patch.object(Path, "open") as open_file:
                self.assert_error(self.read(path), "access_denied")
                open_file.assert_not_called()

    def test_symlink_outside_workspace_is_rejected(self):
        (self.workspace / "outside-link").symlink_to(self.root / "outside.txt")
        with patch.object(Path, "open") as open_file:
            result = self.read("outside-link")
        self.assert_error(result, "access_denied")
        open_file.assert_not_called()

    def test_symlink_to_sensitive_path_is_rejected(self):
        for index, target in enumerate((".env", ".git/config")):
            link = self.workspace / f"link-{index}"
            link.symlink_to(self.workspace / target)
            with self.subTest(target=target), patch.object(Path, "open") as open_file:
                self.assert_error(self.read(link), "access_denied")
                open_file.assert_not_called()

    def test_symlink_to_ordinary_workspace_file_can_be_read(self):
        target = self.workspace / "target.txt"
        target.write_text("通过软链接读取", encoding="utf-8")
        (self.workspace / "alias.txt").symlink_to(target)
        result = self.read("alias.txt")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["path"], "target.txt")
        self.assertEqual(result["content"], "通过软链接读取")

    def test_symlink_loop_returns_read_error(self):
        link = self.workspace / "loop"
        link.symlink_to(link)
        self.assert_error(self.read(link), "read_error")

    def test_exact_byte_limit_is_allowed(self):
        content = "a" * MAX_FILE_BYTES
        (self.workspace / "limit.txt").write_bytes(content.encode("utf-8"))
        result = self.read("limit.txt")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], content)

    def test_oversized_file_is_rejected_before_reading(self):
        (self.workspace / "large.txt").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
        with patch.object(Path, "open") as open_file:
            result = self.read("large.txt")
        self.assert_error(result, "file_too_large")
        open_file.assert_not_called()

    def test_growing_file_is_read_with_a_limit_and_rejected(self):
        (self.workspace / "growing.txt").write_bytes(b"small")
        with patch.object(Path, "open") as open_file:
            source = open_file.return_value.__enter__.return_value
            source.read.return_value = b"a" * (MAX_FILE_BYTES + 1)
            result = self.read("growing.txt")
        source.read.assert_called_once_with(MAX_FILE_BYTES + 1)
        self.assert_error(result, "file_too_large")

    def test_executor_captures_startup_directory(self):
        (self.workspace / "file.txt").write_text("原目录", encoding="utf-8")
        (self.root / "file.txt").write_text("新目录", encoding="utf-8")
        with patch.object(Path, "cwd", return_value=self.workspace):
            executor = create_tool_executor()
        with patch.object(Path, "cwd", return_value=self.root) as cwd:
            result = executor("read_file", {"path": "file.txt"})
        cwd.assert_not_called()
        self.assertEqual(result["content"], "原目录")

    def test_executor_can_use_an_explicit_workspace(self):
        (self.workspace / "file.txt").write_text("指定目录", encoding="utf-8")
        with patch.object(Path, "cwd") as cwd:
            executor = create_tool_executor(str(self.workspace))
            result = executor("read_file", {"path": "file.txt"})
        cwd.assert_not_called()
        self.assertEqual(result["content"], "指定目录")


if __name__ == "__main__":
    unittest.main()
