from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from harness.permissions import PermissionPolicy, SessionPermissionCache
from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.tools.executor import ToolError
from harness.tools.run_verify import DEFINITION, execute


class RunVerifyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.root.joinpath("check.js").write_text(
            "console.log('node-ok')", encoding="utf-8",
        )
        self.root.joinpath("check.py").write_text(
            "print('python-ok')", encoding="utf-8",
        )
        suite = self.root / "suite"
        suite.mkdir()
        suite.joinpath("check.test.js").write_text(
            "const { test } = require('node:test'); test('ok', () => {});",
            encoding="utf-8",
        )

    def test_definition_is_registered_with_restricted_parameters(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("run_verify", names)
        self.assertIs(REGISTRY.get("run_verify")[0], DEFINITION)
        schema = DEFINITION.to_deepseek()["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"target", "runner", "timeout"})
        self.assertEqual(schema["required"], ["target"])

    def test_executor_runs_existing_scripts_and_always_asks(self):
        confirm = Mock(return_value=True)
        executor = create_tool_executor(
            self.root, confirm=confirm,
            permissions=PermissionPolicy(allow=["run_verify"]),
        )
        result = executor("run_verify", {
            "target": "check.js", "runner": "node", "timeout": 2,
        })
        self.assertEqual(result["status"], "success")
        self.assertIn("node-ok", result["stdout"])
        confirm.assert_called_once()

    def test_auto_mode_runs_verification_without_confirmation(self):
        confirm = Mock(side_effect=AssertionError("auto 模式不应要求确认。"))
        executor = create_tool_executor(
            self.root, confirm=confirm,
            permissions=PermissionPolicy(mode="auto"),
        )
        result = executor("run_verify", {
            "target": "check.js", "runner": "node", "timeout": 2,
        })
        self.assertEqual(result["status"], "success")

        confirm = Mock(return_value=False)
        denied = create_tool_executor(
            self.root, confirm=confirm,
            permissions=PermissionPolicy(allow=["run_verify"]),
        )
        result = denied("run_verify", {"target": "check.py", "runner": "python3"})
        self.assertEqual(result["code"], "confirmation_denied")

    def test_validation_rejects_inline_shell_and_outside_targets(self):
        for arguments in (
            {"target": "check.js", "runner": "shell"},
            {"target": "missing.js", "runner": "node"},
            {"target": "check.txt", "runner": "node"},
            {"target": "../outside.js", "runner": "node"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ToolError):
                execute(arguments, self.root)

    def test_approved_directory_reuses_verification_without_repeated_prompts(self):
        cache = SessionPermissionCache()
        confirm = Mock(return_value=True)
        executor = create_tool_executor(
            self.root, confirm=confirm, session_cache=cache,
            permissions=PermissionPolicy(allow=["run_verify"]),
        )
        first = executor("run_verify", {
            "target": "check.js", "runner": "node", "timeout": 2,
        })
        second = executor("run_verify", {
            "target": "check.js", "runner": "node", "timeout": 2,
        })
        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        confirm.assert_called_once()

    def test_node_test_runner_handles_directories(self):
        executor = create_tool_executor(
            self.root, confirm=Mock(return_value=True),
            permissions=PermissionPolicy(allow=["run_verify"]),
        )
        result = executor("run_verify", {
            "target": "suite", "runner": "node-test", "timeout": 2,
        })
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
