from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from harness.permissions import (
    PermissionPolicy, PermissionRule, check_permission, get_risk_level, parse_rules,
)


class PermissionRuleTests(unittest.TestCase):
    def test_first_match_returns_without_consulting_later_rules(self):
        first = Mock(action="ask", matches=Mock(return_value=False))
        second = Mock(action="allow", matches=Mock(return_value=True))
        last = Mock(matches=Mock(side_effect=AssertionError("命中后不能继续匹配。")))
        self.assertEqual(check_permission("example", {}, (first, second, last)), "allow")
        last.matches.assert_not_called()

    def test_fallback_uses_risk_with_existing_tool_configuration_overrides(self):
        expected = {"read_file": ("low", "allow"), "grep": ("low", "allow"),
                    "write_file": ("medium", "ask"), "bash": ("high", "ask"),
                    "new_tool": ("medium", "ask")}
        for name, (risk, decision) in expected.items():
            with self.subTest(tool=name):
                self.assertEqual(get_risk_level(name, {}), risk)
                self.assertEqual(check_permission(name, {}, ()), decision)
                self.assertEqual(PermissionPolicy().check(name), decision)
        policy = PermissionPolicy(ask=["read_file"], allow=["new_tool"])
        self.assertEqual(policy.check("read_file"), "ask")
        self.assertEqual(policy.check("new_tool"), "allow")

    def test_snapshot_sorts_denies_then_priority_preserving_ties(self):
        values = [
            {"tool": "example", "action": "allow", "priority": 10},
            {"tool": "example", "action": "ask", "priority": 20},
            {"tool": "example", "action": "deny", "priority": -10},
            {"tool": "example", "action": "allow", "priority": 20},
        ]
        policy = PermissionPolicy(rules=values)
        values[2]["action"] = "allow"
        values.clear()
        self.assertEqual([(rule.action, rule.priority) for rule in policy.rules],
                         [("deny", -10), ("ask", 20), ("allow", 20), ("allow", 10)])
        self.assertEqual(policy.check("example"), "deny")
        with self.assertRaises(FrozenInstanceError):
            policy.rules[0].action = "allow"

    def test_tool_names_match_exactly_and_unmatched_rule_uses_default(self):
        policy = PermissionPolicy(rules=[PermissionRule("read_file", "deny")])
        self.assertEqual(policy.check("read_file"), "deny")
        self.assertEqual(policy.check("read_file_extra"), "ask")
        self.assertEqual(policy.check("grep"), "allow")

    def test_command_pattern_searches_full_text_without_interpreting_shell(self):
        rule = PermissionRule("bash", "deny", command_pattern=r"\brm\s+-(?:rf|fr)\b")
        for command in ("rm -rf tmp", "pwd; /bin/rm -fr tmp", "printf done\nrm\t-rf tmp"):
            with self.subTest(command=command):
                self.assertTrue(rule.matches("bash", {"command": command}))
        self.assertFalse(rule.matches("bash", {"command": "printf harmless"}))
        self.assertFalse(rule.matches("example", {"command": "rm -rf tmp"}))
        self.assertTrue(rule.matches("bash", {"command": "echo 'rm -rf tmp'"}))
        # 这是文本规则，不解析变量；未命中的 Bash 仍走高风险确认。
        self.assertFalse(rule.matches("bash", {"command": "cmd=rm; $cmd -rf tmp"}))
        self.assertEqual(PermissionPolicy(rules=[rule]).check("bash", {"command": "cmd=rm; $cmd -rf tmp"}), "ask")

    def test_relative_and_absolute_paths_share_the_same_workspace_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rule = PermissionRule("write_file", "allow", directory="tests")
            for name in ("tests/new/a.py", "./tests/child/../a.py", str(root / "tests/a.py")):
                with self.subTest(path=name):
                    self.assertTrue(rule.matches("write_file", {"path": name}, workspace=root))
            for name in ("tests2/a.py", "tests/../src/a.py", "../tests/a.py"):
                with self.subTest(path=name):
                    self.assertFalse(rule.matches("write_file", {"path": name}, workspace=root))
            with self.assertRaises(ValueError):
                rule.matches("write_file", {"path": "tests/\x00"}, workspace=root)

    def test_restriction_covers_configured_symlink_directory_and_its_real_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "real").mkdir()
            (root / "private").symlink_to(root / "real", target_is_directory=True)
            (root / "alias").symlink_to(root / "private", target_is_directory=True)
            for action in ("deny", "ask"):
                rule = PermissionRule("write_file", action, directory="private")
                for path in ("private/a.txt", "real/a.txt", "real/child/a.txt", "alias/a.txt",
                             "REAL/new.txt", str(root / "real/a.txt")):
                    with self.subTest(action=action, path=path):
                        self.assertTrue(rule.matches("write_file", {"path": path}, workspace=root))
                self.assertFalse(rule.matches("write_file", {"path": "real2/a.txt"}, workspace=root))

    def test_resolving_restricted_directory_does_not_expand_allow_rules(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "real").mkdir()
            (root / "private").symlink_to(root / "real", target_is_directory=True)
            rule = PermissionRule("write_file", "allow", directory="private")
            for path in ("private/a.txt", "real/a.txt"):
                with self.subTest(path=path):
                    self.assertFalse(rule.matches("write_file", {"path": path}, workspace=root))

    def test_default_rule_names_preserve_original_indices_after_priority_sort(self):
        rules = parse_rules([
            {"tool": "example", "action": "allow", "priority": 1},
            {"tool": "example", "action": "ask", "priority": 20},
            {"tool": "example", "action": "deny", "priority": -10},
            {"tool": "example", "action": "ask", "priority": 30, "name": "explicit-name"},
        ])
        self.assertEqual([rule.name for rule in rules],
                         ["rules[2]", "explicit-name", "rules[1]", "rules[0]"])
        decision = PermissionPolicy(rules=rules).evaluate("example")
        self.assertEqual(decision.matched_rule, "rules[2]")
        self.assertEqual(parse_rules([PermissionRule("example", "ask")])[0].name, "rules[0]")

    def test_rule_names_must_be_unique_including_generated_names(self):
        for names in (("same", "same"), ("rules[1]", None)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                parse_rules([PermissionRule("example", "ask", name=name) for name in names])

    def test_rule_names_reject_invalid_values_and_accept_printable_boundary(self):
        for name in ("", " \t", "line\nbreak", "escape\x1b[31m", "x" * 129, 1, []):
            with self.subTest(name=name), self.assertRaises(ValueError):
                PermissionRule("example", "ask", name=name)
        self.assertEqual(PermissionRule("example", "ask", name="界" * 128).name, "界" * 128)

    def test_empty_and_invalid_rules_cannot_silently_become_permissions(self):
        self.assertEqual(parse_rules([]), ())
        for value in (None, "allow", {}, [None], [{"tool": "bash"}],
                      [{"tool": "bash", "action": "deny", "unknown": True}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_rules(value)
        for pattern in ("[", "a{999999999999999999}", "(" * 2000):
            with self.subTest(pattern_length=len(pattern)), self.assertRaises(ValueError):
                PermissionRule("bash", "deny", command_pattern=pattern)


if __name__ == "__main__":
    unittest.main()
