import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.audit import PermissionAuditLog
from harness.permissions import PermissionPolicy, SessionPermissionCache
from harness.tools import create_tool_executor


class PermissionSessionAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = SessionPermissionCache()
        self.audit = PermissionAuditLog(self.root)

    def executor(self, *, confirm=None, rules=(), cache=None, audit=None, deny=()):
        return create_tool_executor(
            self.root, confirm=confirm, permissions=PermissionPolicy(rules=rules, deny=deny),
            session_cache=self.cache if cache is None else cache,
            audit=self.audit if audit is None else audit,
        )

    def entries(self):
        return [json.loads(line) for line in
                (self.root / ".harness/permission.log").read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def write(execute, path, content="approved content"):
        return execute("write_file", {"path": path, "content": content})

    def test_audit_links_approval_rejection_unavailable_and_remembered_calls(self):
        confirm = Mock(side_effect=[True, False])
        execute = self.executor(confirm=confirm)
        self.assertTrue(self.write(execute, "src/first.txt")["executed"])
        self.assertTrue(self.write(execute, "src/child/second.txt")["executed"])
        self.assertEqual(self.write(execute, "other/rejected.txt")["code"], "confirmation_denied")
        unavailable = self.executor(cache=SessionPermissionCache())
        self.assertEqual(self.write(unavailable, "third/unavailable.txt")["code"], "confirmation_required")
        entries = self.entries()
        self.assertTrue({"approved", "rejected", "unavailable", "remembered"}
                        <= {entry["confirmation"] for entry in entries})
        self.assertEqual({entry["session_id"] for entry in entries}, {self.audit.session_id})
        calls = {}
        for entry in entries:
            self.assertTrue(entry["call_id"])
            self.assertTrue(entry["matched_rule"])
            calls.setdefault(entry["params"]["path"], set()).add(entry["call_id"])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(len(ids) == 1 for ids in calls.values()))
        self.assertEqual(len(set.union(*calls.values())), 4)
        remembered = [entry for entry in entries if entry["confirmation"] == "remembered"]
        self.assertTrue(remembered)
        self.assertTrue(all(entry["matched_rule"] == "session:write_directory" for entry in remembered))
        self.assertEqual(confirm.call_count, 2)

    def test_success_remembers_parent_and_children_but_cannot_override_deny(self):
        rules = [{"name": "private-deny", "tool": "write_file", "action": "deny",
                  "directory": "src/private"}]
        confirm = Mock(side_effect=[True, False])
        execute = self.executor(confirm=confirm, rules=rules)
        self.assertTrue(self.write(execute, "src/first.txt")["executed"])
        self.assertTrue(self.write(execute, "src/next.txt")["executed"])
        self.assertTrue(self.write(execute, "src/nested/next.txt")["executed"])
        denied = self.write(execute, "src/private/never.txt")
        self.assertEqual(denied["code"], "permission_denied")
        self.assertFalse(denied["executed"])
        self.assertFalse((self.root / "src/private").exists())
        self.assertEqual(self.write(execute, "src2/never.txt")["code"], "confirmation_denied")
        self.assertFalse((self.root / "src2").exists())
        self.assertEqual(confirm.call_count, 2)
        self.assertTrue(any(entry["matched_rule"] == "private-deny" and entry["decision"] == "deny"
                            for entry in self.entries()))

    def test_failed_write_does_not_remember_directory(self):
        confirm = Mock(return_value=True)
        execute = self.executor(confirm=confirm)
        result = self.write(execute, "src/bad.txt", "invalid\x00content")
        self.assertEqual(result["code"], "invalid_encoding")
        self.assertFalse(self.cache.is_approved("write_file", {"path": "src/next.txt"}, workspace=self.root))
        self.assertTrue(self.write(execute, "src/good.txt")["executed"])
        self.assertEqual(confirm.call_count, 2)
        self.assertFalse((self.root / "src/bad.txt").exists())

    def test_new_executor_and_cache_do_not_inherit_prior_session_approval(self):
        first = self.executor(confirm=Mock(return_value=True))
        self.assertTrue(self.write(first, "src/first.txt")["executed"])
        next_audit = PermissionAuditLog(self.root)
        second = self.executor(cache=SessionPermissionCache(), audit=next_audit)
        result = self.write(second, "src/second.txt")
        self.assertEqual(result["code"], "confirmation_required")
        self.assertFalse((self.root / "src/second.txt").exists())
        self.assertEqual({entry["session_id"] for entry in self.entries()},
                         {self.audit.session_id, next_audit.session_id})

    def test_audit_sync_failure_before_confirmation_or_execution_blocks_write(self):
        real_sync = os.fsync
        for fail_at in (1, 3):
            with self.subTest(fail_at=fail_at):
                count = 0

                def sync(descriptor):
                    nonlocal count
                    count += 1
                    if count == fail_at:
                        raise OSError("private storage details")
                    return real_sync(descriptor)

                confirm = Mock(return_value=True)
                execute = self.executor(confirm=confirm, cache=SessionPermissionCache())
                with patch("harness.audit.os.fsync", side_effect=sync):
                    result = self.write(execute, f"new{fail_at}/never.txt")
                self.assertEqual(result["code"], "audit_failed")
                self.assertFalse(result["executed"])
                self.assertFalse((self.root / f"new{fail_at}").exists())
                self.assertNotIn("private storage details", result["message"])
                self.assertEqual(confirm.call_count, 0 if fail_at == 1 else 1)

    def test_confirmation_cannot_redirect_write_into_denied_directory_or_grant_cache(self):
        for name in ("public", "private"):
            (self.root / name).mkdir()
            (self.root / name / "note.txt").write_text(name, encoding="utf-8")
        alias = self.root / "alias"
        alias.symlink_to(self.root / "public", target_is_directory=True)

        def confirm(*_):
            alias.unlink()
            alias.symlink_to(self.root / "private", target_is_directory=True)
            return True

        execute = self.executor(confirm=confirm, rules=[{
            "name": "private-deny", "tool": "write_file", "action": "deny", "directory": "private",
        }])
        result = self.write(execute, "alias/note.txt", "must not be written")
        self.assertEqual(result["code"], "permission_denied")
        self.assertFalse(result["executed"])
        for name in ("public", "private"):
            self.assertEqual((self.root / name / "note.txt").read_text(encoding="utf-8"), name)
            self.assertFalse(self.cache.is_approved("write_file", {"path": f"{name}/next.txt"}, workspace=self.root))
        self.assertTrue(any(entry["event"] == "guard_denial" and entry["matched_rule"] == "private-deny"
                            for entry in self.entries()))

    def redirect_during_audit(self, *, remembered, destination="private"):
        for name in ("public/inside", "public/other", "private"):
            (self.root / name).mkdir(parents=True)
            (self.root / name / "note.txt").write_text(name, encoding="utf-8")
        alias = self.root / "public/alias"
        alias.symlink_to(self.root / "public/inside", target_is_directory=True)
        rules = [{"name": "private-deny", "tool": "write_file", "action": "deny", "directory": "private"}]
        if not remembered:
            rules.append({"name": "public-allow", "tool": "write_file", "action": "allow", "directory": "public"})
        confirm = Mock(return_value=True)
        execute = self.executor(confirm=confirm, rules=rules)
        if remembered:
            self.assertTrue(self.write(execute, "public/seed.txt")["executed"])
        real_record = self.audit.record

        def redirect(*args, **kwargs):
            real_record(*args, **kwargs)
            if kwargs.get("event") == "before_execute":
                alias.unlink()
                alias.symlink_to(self.root / destination, target_is_directory=True)

        with patch.object(self.audit, "record", side_effect=redirect):
            result = self.write(execute, "public/alias/note.txt", "must not be written")
        self.assertFalse(result["executed"])
        for name in ("public/inside", "public/other", "private"):
            self.assertEqual((self.root / name / "note.txt").read_text(encoding="utf-8"), name)
        self.assertEqual(confirm.call_count, int(remembered))
        guard = [entry for entry in self.entries() if entry["event"] == "guard_denial"]
        self.assertEqual(len(guard), 1)
        self.assertEqual(guard[0]["decision"], "deny")
        return result, guard[0]

    def test_rule_allow_rechecks_deny_after_audit_redirects_target(self):
        result, guard = self.redirect_during_audit(remembered=False)
        self.assertEqual(result["code"], "permission_denied")
        self.assertEqual(guard["matched_rule"], "private-deny")

    def test_cache_allow_rechecks_deny_after_audit_redirects_target(self):
        result, guard = self.redirect_during_audit(remembered=True)
        self.assertEqual(result["code"], "permission_denied")
        self.assertEqual(guard["matched_rule"], "private-deny")
        self.assertEqual(guard["confirmation"], "remembered")

    def test_target_change_with_unchanged_allow_rule_is_also_rejected(self):
        result, guard = self.redirect_during_audit(remembered=False, destination="public/other")
        self.assertEqual(result["code"], "permission_changed")
        self.assertEqual(guard["matched_rule"], "guard:target_changed")

    def test_denial_messages_explain_action_reason_and_alternative_without_secrets(self):
        execute = self.executor(deny=["write_file", "bash"])
        secrets = ("private-file-body-123", "private-command-value-456")
        for name, params in (
            ("write_file", {"path": "src/no.txt", "content": secrets[0]}),
            ("bash", {"command": f"printf '{secrets[1]}'"}),
        ):
            with self.subTest(tool=name):
                result = execute(name, params)
                self.assertEqual(result["code"], "permission_denied")
                self.assertFalse(result["executed"])
                self.assertIn(name, result["message"])
                self.assertIn("原因：", result["message"])
                self.assertIn("建议：", result["message"])
                for secret in secrets:
                    self.assertNotIn(secret, result["message"])
        log = (self.root / ".harness/permission.log").read_text(encoding="utf-8")
        for secret in secrets:
            self.assertNotIn(secret, log)
        entries = self.entries()
        self.assertEqual(entries[0]["params"], {"path": "src/no.txt", "content_chars": len(secrets[0])})
        self.assertIn("command_sha256", entries[1]["params"])
        self.assertNotIn("command", entries[1]["params"])

    def test_audit_directory_is_protected_from_file_tools_and_recursive_search(self):
        self.audit.record("read_file", {"path": "seed.txt"}, "allow", "default", risk="low")
        private = self.root / ".harness/private.txt"
        private.write_text("private-needle-789", encoding="utf-8")
        execute = self.executor(rules=[{"tool": "write_file", "action": "allow", "directory": "."}])
        for name, params in (
            ("read_file", {"path": ".harness/private.txt"}),
            ("write_file", {"path": ".harness/private.txt", "content": "replacement"}),
            ("grep", {"path": ".harness", "keyword": "private-needle-789"}),
        ):
            with self.subTest(tool=name):
                result = execute(name, params)
                self.assertEqual(result["code"], "access_denied")
                self.assertFalse(result["executed"])
        self.assertEqual(private.read_text(encoding="utf-8"), "private-needle-789")
        search = execute("grep", {"path": ".", "keyword": "private-needle-789"})
        self.assertEqual(search["status"], "success")
        self.assertEqual(search["matches"], [])
        self.assertGreaterEqual(search["skipped_directories"], 1)


if __name__ == "__main__":
    unittest.main()
