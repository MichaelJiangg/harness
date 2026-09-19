from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from harness.audit import AuditError, PermissionAuditLog
from harness.config import get_settings
from harness.permissions import PermissionPolicy, SessionPermissionCache
from harness.tools import REGISTRY, ToolRegistry, create_tool_executor, get_tool_definitions
from harness.tools.definition import ToolDefinition
from harness.tools.delegate import DEFINITION, execute
from harness.tools.executor import ToolError


class DelegateToolTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.arguments = {"description": "检查入口", "task": "读取项目入口并说明用途。"}
        self.policy = PermissionPolicy(allow=["delegate"])

    def context(self, *, tools=None, runner=None, parent=True):
        module = ModuleType("harness.orchestration")
        state = SimpleNamespace(
            tools=get_tool_definitions() if tools is None else tools,
            tool_result_limit=get_settings()["context"]["tool_result_chars"],
        )
        module.get_current_query = Mock(return_value=state if parent else None)
        module.run_delegate = runner or Mock(return_value={"content": "子任务报告"})
        return module, patch.dict("sys.modules", {"harness.orchestration": module})

    def test_definition_is_discovered_with_only_model_task_parameters(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("delegate", names)
        definition, handler = REGISTRY.get("delegate")
        self.assertEqual(definition, DEFINITION)
        self.assertIs(handler, execute)
        self.assertTrue(definition.supports_cancellation)
        schema = definition.to_deepseek()["function"]["parameters"]
        self.assertEqual(schema["required"], ["description", "task"])
        self.assertEqual(set(schema["properties"]), {"description", "task"})
        self.assertFalse(schema["additionalProperties"])

    def test_delegate_description_prioritizes_directory_and_multifile_analysis(self):
        description = REGISTRY.get("delegate")[0].description
        self.assertIn("目录级、多文件、代码质量、跨文件对比", description)
        self.assertIn("开始时直接调用", description)
        self.assertIn("不要先执行目录清点或行数统计", description)

    def test_handler_invokes_internal_runner_with_task_only(self):
        runner = Mock(return_value={"content": "独立结果"})
        result = execute(self.arguments, self.root, runner=runner)
        self.assertEqual(result, {"content": "独立结果"})
        runner.assert_called_once_with(**self.arguments)

    def test_invalid_task_text_and_cancellation_never_call_runner(self):
        runner = Mock()
        for name in ("description", "task"):
            for value in ("", " \t\n", "bad\x00text", None, 7):
                with self.subTest(name=name, value=value), self.assertRaises(ToolError) as raised:
                    execute({**self.arguments, name: value}, self.root, runner=runner)
                self.assertEqual(raised.exception.code, "invalid_arguments")
        abort = Event()
        abort.set()
        with self.assertRaises(ToolError) as raised:
            execute(self.arguments, self.root, runner=runner, abort=abort)
        self.assertEqual(raised.exception.code, "execution_cancelled")
        runner.assert_not_called()

    def test_model_cannot_supply_runtime_callbacks_or_approval(self):
        executor = create_tool_executor(self.root, permissions=self.policy)
        module, context = self.context()
        with context:
            for name in ("runner", "permissions", "approved", "registry"):
                with self.subTest(name=name):
                    result = executor("delegate", {**self.arguments, name: "injected"})
                    self.assertEqual(result["code"], "invalid_arguments")
                    self.assertFalse(result["executed"])
        module.get_current_query.assert_not_called()
        module.run_delegate.assert_not_called()

    def test_no_query_context_returns_clear_error_without_running_child(self):
        with self.assertRaises(ToolError) as raised:
            execute(self.arguments, self.root)
        self.assertEqual(raised.exception.code, "delegation_unavailable")
        module, context = self.context(parent=False)
        with context:
            executor = create_tool_executor(self.root, permissions=self.policy)
            result = executor("delegate", self.arguments)
        self.assertEqual(result["code"], "delegation_unavailable")
        self.assertFalse(result["executed"])
        module.run_delegate.assert_not_called()

    def test_deny_rejection_and_audit_failure_precede_child_creation(self):
        for reason in ("deny", "reject", "audit"):
            with self.subTest(reason=reason):
                module, context = self.context()
                confirm = Mock(return_value=False)
                policy = (PermissionPolicy(deny=["delegate"]) if reason == "deny" else
                          PermissionPolicy(ask=["delegate"]) if reason == "reject" else self.policy)
                audit = Mock()
                if reason == "audit":
                    audit.record.side_effect = AuditError("log unavailable")
                executor = create_tool_executor(self.root, confirm=confirm, permissions=policy, audit=audit)
                with context:
                    result = executor("delegate", self.arguments)
                self.assertEqual(result["code"], {"deny": "permission_denied", "reject": "confirmation_denied",
                                                  "audit": "audit_failed"}[reason])
                self.assertFalse(result["executed"])
                module.get_current_query.assert_not_called()
                module.run_delegate.assert_not_called()
                self.assertEqual(confirm.call_count, int(reason == "reject"))

    def test_child_schema_and_dispatch_are_same_parent_visible_subset_without_delegate(self):
        visible = [REGISTRY.get(name)[0].to_deepseek()
                   for name in ("read_file", "delegate", "background_check",
                                "background_submit", "swarm")]
        visible += [{"type": "function", "function": {"name": "missing"}}, visible[0]]
        result_holder = {}

        def runner(*, executor_factory, tools, **task):
            result_holder["tools"] = tools
            child = executor_factory(abort=Event())
            for name, params in (("delegate", self.arguments), ("write_file", {"path": "no.txt", "content": "no"}),
                                 ("bash", {"command": "printf unused"}), ("grep", {"keyword": "no"}),
                                 ("background_check", {"task_id": 1}),
                                 ("background_submit", self.arguments),
                                 ("swarm", self.arguments)):
                result_holder[name] = child(name, params)
            result_holder["read_file"] = child("read_file", {"path": "sample.txt"})
            return {"content": "限制后的子任务结果"}

        (self.root / "sample.txt").write_text("visible content", encoding="utf-8")
        module, context = self.context(tools=visible, runner=runner)
        with context:
            result = create_tool_executor(self.root, permissions=self.policy)("delegate", self.arguments)
        self.assertTrue(result["executed"])
        self.assertEqual([item["function"]["name"] for item in result_holder["tools"]], ["read_file"])
        self.assertEqual(result_holder["read_file"]["content"], "visible content")
        for name in ("delegate", "write_file", "bash", "grep",
                     "background_check", "background_submit", "swarm"):
            self.assertEqual(result_holder[name]["code"], "unknown_tool")
            self.assertFalse(result_holder[name]["executed"])
        self.assertFalse((self.root / "no.txt").exists())

    def test_child_inherits_workspace_policy_confirmation_cache_audit_and_abort(self):
        cache = SessionPermissionCache()
        cache.remember("write_file", "src", workspace=self.root)
        audit = PermissionAuditLog(self.root)
        confirm = Mock(return_value=False)
        abort = Event()
        policy = PermissionPolicy(allow=["delegate"], deny=["read_file"])
        inherited = {}

        def runner(*, executor_factory, tools, **task):
            child = executor_factory(abort=abort)
            inherited.update(child.keywords)
            self.assertEqual(child("read_file", {"path": "sample.txt"})["code"], "permission_denied")
            self.assertEqual(child("write_file", {"path": "other/no.txt", "content": "no"})["code"],
                             "confirmation_denied")
            self.assertTrue(child("write_file", {"path": "src/yes.txt", "content": "yes"})["executed"])
            abort.set()
            self.assertEqual(child("write_file", {"path": "src/cancelled.txt", "content": "no"})["code"], "cancelled")
            return {"content": "继承验证完成"}

        module, context = self.context(runner=runner)
        with context:
            create_tool_executor(self.root, confirm=confirm, permissions=policy, session_cache=cache,
                                 audit=audit, abort=abort)("delegate", self.arguments)
        self.assertEqual(inherited["workspace"], self.root)
        for name, expected in (("permissions", policy), ("confirm", confirm), ("session_cache", cache),
                               ("audit", audit), ("abort", abort)):
            self.assertIs(inherited[name], expected)
        confirm.assert_called_once()
        self.assertEqual((self.root / "src/yes.txt").read_text(encoding="utf-8"), "yes")
        self.assertFalse((self.root / "other").exists())
        self.assertFalse((self.root / "src/cancelled.txt").exists())

    def test_executor_registry_is_frozen_and_excludes_tools_added_later(self):
        registry = ToolRegistry()
        definition = ToolDefinition("example", "示例", {"type": "object", "additionalProperties": False})
        handler = Mock(return_value={"content": "ok"})
        registry.register(definition, handler)
        executor = create_tool_executor(self.root, registry=registry, permissions=PermissionPolicy(allow=["example"]))
        definition.input_schema["additionalProperties"] = True
        registry.register(ToolDefinition("later", "晚注册", {"type": "object"}), handler)
        self.assertEqual(executor("later", {})["code"], "unknown_tool")
        self.assertEqual(executor("example", {"unexpected": True})["code"], "invalid_arguments")
        self.assertTrue(executor("example", {})["executed"])
        handler.assert_called_once_with({}, self.root)
        with self.assertRaises(ValueError):
            create_tool_executor(self.root, registry={})

    def test_child_cannot_gain_tool_missing_from_parent_executor_registry(self):
        registry = ToolRegistry()
        registry.register(DEFINITION, execute)
        registry.register(*REGISTRY.get("read_file"))
        observed = {}

        def runner(*, executor_factory, tools, **task):
            observed["tools"] = tools
            observed["write"] = executor_factory(abort=Event())("write_file", {"path": "never.txt", "content": "no"})
            return {"content": "done"}

        module, context = self.context(runner=runner)
        with context:
            result = create_tool_executor(self.root, registry=registry, permissions=self.policy)("delegate", self.arguments)
        self.assertTrue(result["executed"])
        self.assertEqual([tool["function"]["name"] for tool in observed["tools"]], ["read_file"])
        self.assertEqual(observed["write"]["code"], "unknown_tool")
        self.assertFalse((self.root / "never.txt").exists())


if __name__ == "__main__":
    unittest.main()
