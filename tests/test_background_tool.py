from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import Mock

from harness.background import BackgroundManager, COMPLETED
from harness.permissions import PermissionPolicy
from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.tools.background_check import DEFINITION as CHECK_DEFINITION, execute as check
from harness.tools.background_submit import DEFINITION as SUBMIT_DEFINITION, execute as submit
from harness.tools.executor import ToolError


class BackgroundToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.manager = BackgroundManager(max_concurrent=2, default_timeout=2)
        self.addCleanup(self.manager.shutdown)

    def wait_completed(self, task_id, timeout=1):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            task = self.manager.check(task_id)
            if task["status"] == COMPLETED:
                return task
            sleep(0.005)
        return self.manager.check(task_id)

    def test_definitions_are_registered_with_expected_schema(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("background_submit", names)
        self.assertIn("background_check", names)
        self.assertIs(REGISTRY.get("background_submit")[0], SUBMIT_DEFINITION)
        self.assertIs(REGISTRY.get("background_check")[0], CHECK_DEFINITION)
        submit_schema = SUBMIT_DEFINITION.to_deepseek()["function"]["parameters"]
        self.assertEqual(set(submit_schema["properties"]), {"description", "command", "task", "timeout"})
        self.assertEqual(submit_schema["required"], ["description"])
        check_schema = CHECK_DEFINITION.to_deepseek()["function"]["parameters"]
        self.assertEqual(check_schema["required"], ["task_id"])

    def test_handlers_reject_invalid_arguments_and_missing_runner(self):
        for arguments in (
            {"description": ""},
            {"description": "任务", "command": "printf ok", "task": "分析"},
            {"description": "任务"},
            {"description": "任务", "command": "bad\x00command"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ToolError):
                submit(arguments, self.root)
        with self.assertRaises(ToolError):
            check({"task_id": 1}, self.root)

    def test_executor_submits_and_checks_command_across_calls(self):
        executor = create_tool_executor(
            self.root,
            confirm=Mock(return_value=True),
            permissions=PermissionPolicy(allow=["background_submit"]),
            background_manager=self.manager,
        )
        submitted = executor("background_submit", {
            "description": "后台打印", "command": "printf 'background ok'", "timeout": 2,
        })
        self.assertEqual(submitted["status"], "success")
        task_id = submitted["task"]["task_id"]
        task = self.wait_completed(task_id)
        self.assertEqual(task["status"], COMPLETED)
        checked = executor("background_check", {"task_id": task_id})
        self.assertEqual(checked["status"], "success")
        self.assertEqual(checked["task"]["task_id"], task_id)
        self.assertEqual(checked["task"]["result"]["status"], "success")
        self.assertIn("background ok", checked["task"]["result"]["stdout"])

    def test_bash_deny_and_confirmation_are_not_bypassed(self):
        denied = create_tool_executor(
            self.root,
            permissions=PermissionPolicy(allow=["background_submit"], deny=["bash"]),
            background_manager=self.manager,
        )
        result = denied("background_submit", {
            "description": "被禁止的命令", "command": "printf denied",
        })
        self.assertEqual(result["code"], "permission_denied")
        self.assertFalse(result["executed"])

        confirm = Mock(return_value=False)
        asking = create_tool_executor(
            self.root,
            confirm=confirm,
            permissions=PermissionPolicy(allow=["background_submit"]),
            background_manager=self.manager,
        )
        result = asking("background_submit", {
            "description": "需要确认", "command": "touch ask.txt",
        })
        self.assertEqual(result["code"], "confirmation_denied")
        confirm.assert_called_once()

    def test_model_cannot_inject_manager_or_runner_fields(self):
        executor = create_tool_executor(
            self.root,
            permissions=PermissionPolicy(allow=["background_submit"]),
            background_manager=self.manager,
        )
        result = executor("background_submit", {
            "description": "注入", "command": "printf no", "runner": "ignored",
        })
        self.assertEqual(result["code"], "invalid_arguments")
        self.assertFalse(result["executed"])

    def test_check_unknown_task_is_safe_error(self):
        executor = create_tool_executor(
            self.root,
            permissions=PermissionPolicy(allow=["background_submit"]),
            background_manager=self.manager,
        )
        result = executor("background_check", {"task_id": 999})
        self.assertEqual(result["code"], "background_task_not_found")
        self.assertFalse(result["executed"])


if __name__ == "__main__":
    unittest.main()
