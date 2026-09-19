import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.engine import QueryState, query_loop
from harness.permissions import PermissionPolicy
from harness.tools import create_tool_executor
from harness.usage import UsageLedger
from test_cli import CLISession
from test_engine import FakeClient, reply, tool_call


class PermissionRulesIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.write_allow = {
            "tool": "write_file", "action": "allow", "directory": "tests", "priority": 100,
        }

    def executor(self, rules=None, *, confirm=None, **policy):
        return create_tool_executor(
            self.root, confirm=confirm,
            permissions=PermissionPolicy(rules=[self.write_allow] if rules is None else rules, **policy),
        )

    def test_directory_allow_writes_nested_files_without_a_confirmation_callback(self):
        execute = self.executor()
        for path in ("tests/nested/new.txt", str(self.root / "tests/absolute.txt")):
            with self.subTest(path=path):
                result = execute("write_file", {"path": path, "content": "许可的正文。\n"})
                self.assertEqual(result["status"], "success")
                self.assertTrue(result["executed"])
                self.assertEqual((self.root / path).read_text(encoding="utf-8"), "许可的正文。\n")

    def test_directory_allow_is_bound_to_the_executor_startup_workspace(self):
        confirm = Mock(return_value=False)
        execute = self.executor(confirm=confirm)
        with TemporaryDirectory() as later:
            with patch("harness.tools.executor.Path.cwd", return_value=Path(later)):
                result = execute("write_file", {"path": "tests/note.txt", "content": "固定根目录"})
            self.assertTrue(result["executed"])
            self.assertFalse(Path(later, "tests").exists())
        self.assertEqual((self.root / "tests/note.txt").read_text(encoding="utf-8"), "固定根目录")
        confirm.assert_not_called()

    def test_sibling_and_parent_escape_paths_still_require_confirmation(self):
        execute = self.executor()
        for path in ("tests2/new.txt", "tests/../src/new.txt", "src/new.txt"):
            with self.subTest(path=path):
                result = execute("write_file", {"path": path, "content": "不可自动写入"})
                self.assertEqual(result["code"], "confirmation_required")
                self.assertFalse(result["executed"])
                self.assertFalse((self.root / path).exists())
        self.assertEqual([path for path in self.root.iterdir() if path.name != ".harness"], [])

    def test_unmatched_write_obeys_rejection_without_creating_directories(self):
        confirm = Mock(return_value=False)
        arguments = {"path": "src/new.txt", "content": "未批准"}
        result = self.executor(confirm=confirm)("write_file", arguments)
        self.assertEqual(result["code"], "confirmation_denied")
        self.assertFalse(result["executed"])
        self.assertFalse((self.root / "src").exists())
        confirm.assert_called_once_with("write_file", arguments, self.root)

    def test_priority_and_declaration_order_determine_whether_a_write_needs_confirmation(self):
        ask = {"tool": "write_file", "action": "ask", "directory": "tests/private", "priority": 200}
        cases = (
            ([self.write_allow, ask], "confirmation_denied"),
            ([self.write_allow, {**ask, "priority": 50}], None),
            ([{**ask, "priority": 100}, self.write_allow], "confirmation_denied"),
            ([self.write_allow, {**ask, "priority": 100}], None),
        )
        for index, (rules, error) in enumerate(cases):
            with self.subTest(case=index):
                confirm = Mock(return_value=False)
                path = f"tests/private/result_{index}.txt"
                result = self.executor(rules, confirm=confirm)("write_file", {"path": path, "content": "正文"})
                if error:
                    self.assertEqual(result["code"], error)
                    self.assertFalse((self.root / path).exists())
                    confirm.assert_called_once()
                else:
                    self.assertTrue(result["executed"])
                    self.assertEqual((self.root / path).read_text(encoding="utf-8"), "正文")
                    confirm.assert_not_called()

    def test_rule_deny_and_tool_deny_override_directory_allow_without_asking(self):
        deny_rule = {"tool": "write_file", "action": "deny", "directory": "tests/private", "priority": -100}
        for policy in ({"rules": [self.write_allow, deny_rule]},
                       {"rules": [self.write_allow], "deny": ["write_file"]}):
            with self.subTest(policy=policy):
                confirm = Mock(return_value=True)
                result = self.executor(confirm=confirm, **policy)(
                    "write_file", {"path": "tests/private/note.txt", "content": "禁止写入"},
                )
                self.assertEqual(result["code"], "permission_denied")
                self.assertFalse(result["executed"])
                confirm.assert_not_called()
        self.assertFalse((self.root / "tests").exists())

    def test_rule_denial_and_allowed_write_return_corresponding_tool_ids_to_the_model(self):
        rules = [self.write_allow, {
            "tool": "write_file", "action": "deny", "directory": "tests/private", "priority": 0,
        }]
        calls = [tool_call(call_id, "write_file", json.dumps({"path": path, "content": "正文"}))
                 for call_id, path in (("blocked-write", "tests/private/note.txt"),
                                       ("allowed-write", "tests/public/note.txt"))]
        client = FakeClient([reply(None, calls), reply("规则请求处理完毕。")])
        confirm = Mock(return_value=True)
        state = QueryState(client, UsageLedger(), tool_executor=self.executor(rules, confirm=confirm))
        state.messages.append({"role": "user", "content": "按照权限规则写入文件"})
        self.assertEqual(query_loop(state), "规则请求处理完毕。")
        messages = [message for message in client.requests[1]["messages"] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in messages], ["blocked-write", "allowed-write"])
        self.assertEqual(json.loads(messages[0]["content"])["code"], "permission_denied")
        self.assertTrue(json.loads(messages[1]["content"])["executed"])
        self.assertFalse((self.root / "tests/private").exists())
        self.assertEqual((self.root / "tests/public/note.txt").read_text(encoding="utf-8"), "正文")
        confirm.assert_not_called()

    def test_symlinks_cannot_extend_directory_allow_inside_or_outside_the_workspace(self):
        (self.root / "tests").mkdir()
        (self.root / "src").mkdir()
        (self.root / "tests/inside").symlink_to(self.root / "src", target_is_directory=True)
        (self.root / "alias").symlink_to(self.root / "tests", target_is_directory=True)
        with TemporaryDirectory() as external:
            (self.root / "tests/outside").symlink_to(external, target_is_directory=True)
            execute = self.executor()
            for path in ("tests/inside/new.txt", "tests/outside/new.txt", "alias/new.txt"):
                with self.subTest(path=path):
                    result = execute("write_file", {"path": path, "content": "不可自动写入"})
                    self.assertEqual(result["code"], "confirmation_required")
                    self.assertFalse(result["executed"])
                    self.assertFalse((self.root / path).exists())
            self.assertEqual(list(Path(external).iterdir()), [])

    def test_alias_into_denied_directory_cannot_bypass_file_tool_rules(self):
        (self.root / "private").mkdir()
        file = self.root / "private/note.txt"
        file.write_text("原始正文", encoding="utf-8")
        (self.root / "alias").symlink_to(self.root / "private", target_is_directory=True)
        confirm = Mock(return_value=True)
        rules = [{"tool": name, "action": "deny", "directory": "private"}
                 for name in ("read_file", "write_file")]
        execute = self.executor(rules, confirm=confirm)
        for name, arguments in (
            ("read_file", {"path": "alias/note.txt"}),
            ("write_file", {"path": "alias/note.txt", "content": "不应覆盖"}),
        ):
            with self.subTest(tool=name):
                result = execute(name, arguments)
                self.assertEqual(result["code"], "permission_denied")
                self.assertFalse(result["executed"])
        self.assertEqual(file.read_text(encoding="utf-8"), "原始正文")
        confirm.assert_not_called()

    def test_confirmation_cannot_redirect_a_read_into_a_denied_directory(self):
        (self.root / "public").mkdir()
        (self.root / "private").mkdir()
        (self.root / "public/note.txt").write_text("公开内容", encoding="utf-8")
        (self.root / "private/note.txt").write_text("禁止读取", encoding="utf-8")
        alias = self.root / "alias"
        alias.symlink_to(self.root / "public", target_is_directory=True)

        def confirm(*_):
            alias.unlink()
            alias.symlink_to(self.root / "private", target_is_directory=True)
            return True

        execute = self.executor([
            {"tool": "read_file", "action": "deny", "directory": "private"},
            {"tool": "read_file", "action": "ask"},
        ], confirm=confirm)
        with patch("pathlib.Path.open") as open_file:
            result = execute("read_file", {"path": "alias/note.txt"})
        self.assertEqual(result["code"], "permission_denied")
        self.assertFalse(result["executed"])
        open_file.assert_not_called()

    def test_directory_restrictions_cover_case_and_unicode_aliases_before_creation(self):
        confirm = Mock(return_value=True)
        for directory, path in (("private", "PRIVATE/new.txt"), ("café", "cafe\u0301/new.txt")):
            with self.subTest(directory=directory):
                execute = self.executor([{"tool": "write_file", "action": "deny", "directory": directory}],
                                        confirm=confirm)
                result = execute("write_file", {"path": path, "content": "不应写入"})
                self.assertEqual(result["code"], "permission_denied")
                self.assertFalse(result["executed"])
                self.assertFalse((self.root / path).exists())
        confirm.assert_not_called()

    def test_existing_file_cannot_bypass_directory_deny_through_case_variant(self):
        (self.root / "private").mkdir()
        (self.root / "private/note.txt").write_text("禁止读取", encoding="utf-8")
        execute = self.executor([{"tool": "read_file", "action": "deny", "directory": "private"}])
        with patch("pathlib.Path.open") as open_file:
            result = execute("read_file", {"path": "PRIVATE/note.txt"})
        self.assertEqual(result["code"], "permission_denied")
        open_file.assert_not_called()

    def test_allow_does_not_override_protected_paths(self):
        confirm = Mock(return_value=True)
        execute = self.executor([{**self.write_allow, "directory": "."}], confirm=confirm)
        for path in ("tests/.env.example", "tests/.git/config"):
            with self.subTest(path=path):
                result = execute("write_file", {"path": path, "content": "不可写入"})
                self.assertEqual(result["code"], "access_denied")
                self.assertFalse(result["executed"])
                self.assertFalse((self.root / path).exists())
        self.assertFalse((self.root / "tests").exists())
        confirm.assert_not_called()

    def test_deny_pattern_searches_complete_compound_commands_without_launching_a_process(self):
        rules = [{"tool": "bash", "action": "deny", "command_pattern": r"\brm\s+-(?:rf|fr)\b"}]
        confirm = Mock(return_value=True)
        execute = self.executor(rules, confirm=confirm)
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            for command in ("rm -rf sample", "printf begin; rm -fr sample", "printf begin\nrm -rf sample"):
                with self.subTest(command=command):
                    result = execute("bash", {"command": command})
                    self.assertEqual(result["code"], "permission_denied")
                    self.assertFalse(result["executed"])
            launch.assert_not_called()
        confirm.assert_not_called()

    def test_unmatched_bash_still_requires_confirmation_and_obeys_rejection(self):
        rules = [{"tool": "bash", "action": "deny", "command_pattern": r"\brm\s+-rf\b"}]
        arguments = {"command": "touch harmless.txt"}
        confirm = Mock(return_value=False)
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            self.assertEqual(self.executor(rules)("bash", arguments)["code"], "confirmation_required")
            result = self.executor(rules, confirm=confirm)("bash", arguments)
            self.assertEqual(result["code"], "confirmation_denied")
            self.assertFalse(result["executed"])
            launch.assert_not_called()
        confirm.assert_called_once_with("bash", arguments, self.root)

    def test_factory_freezes_config_rules_for_the_existing_executor(self):
        settings = {"permissions": {"allow": ["read_file", "grep"], "ask": ["write_file"],
                                    "deny": [], "rules": [dict(self.write_allow)]}}
        with patch("harness.tools.executor.get_settings", return_value=settings):
            existing = create_tool_executor(self.root)
            settings["permissions"]["rules"][0]["directory"] = "src"
            later = create_tool_executor(self.root)
        arguments = {"path": "tests/approved.txt", "content": "已加载规则"}
        self.assertTrue(existing("write_file", arguments)["executed"])
        self.assertEqual(later("write_file", arguments)["code"], "confirmation_required")
        self.assertEqual(existing("write_file", {"path": "src/new.txt", "content": "无授权"})["code"],
                         "confirmation_required")
        self.assertFalse((self.root / "src").exists())

    def test_noninteractive_cli_only_writes_within_the_authorized_directory(self):
        policy = PermissionPolicy(rules=[self.write_allow])
        calls = [tool_call(call_id, "write_file", json.dumps({"path": path, "content": "正文"}))
                 for call_id, path in (("allowed-write", "tests/generated/note.txt"),
                                       ("unconfirmed-write", "src/note.txt"))]
        client = FakeClient([reply(None, calls), reply("目录权限处理完毕。")])
        with patch("harness.cli.create_tool_executor", side_effect=lambda **kwargs:
                   create_tool_executor(self.root, permissions=policy, **kwargs)):
            session = CLISession(client, lines=("写入两个文件\n",), terminal=False, character_delay=0)
            try:
                self.assertTrue(session.output.wait_for("目录权限处理完毕。"))
            finally:
                session.close()
        messages = [message for message in client.requests[1]["messages"] if message["role"] == "tool"]
        self.assertTrue(json.loads(messages[0]["content"])["executed"])
        self.assertEqual(json.loads(messages[1]["content"])["code"], "confirmation_denied")
        self.assertEqual((self.root / "tests/generated/note.txt").read_text(encoding="utf-8"), "正文")
        self.assertFalse((self.root / "src").exists())
        self.assertNotIn("[写入确认]", session.output.getvalue())
        self.assertEqual(session.errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
