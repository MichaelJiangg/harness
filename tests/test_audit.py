from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID

from harness.audit import AuditError, PermissionAuditLog, sanitize


class SanitizationTests(unittest.TestCase):
    def test_only_path_numbers_lengths_and_command_fingerprint_are_saved(self):
        params = {
            "path": "src/config.py", "offset": 0, "column": 12, "limit": 20, "timeout": 3,
            "max_results": 50, "content": "正文与秘密", "keyword": "私密关键词",
            "command": "printf 'sensitive-token'", "password": "password-value",
            "nested": {"content": "nested-secret"}, "glob": "secret-pattern",
        }
        clean = sanitize(params)
        self.assertEqual(clean, {
            "path": "src/config.py", "offset": 0, "column": 12, "limit": 20, "timeout": 3,
            "max_results": 50, "content_chars": 5, "keyword_chars": 5,
            "command_chars": len(params["command"]),
            "command_sha256": hashlib.sha256(params["command"].encode()).hexdigest(),
        })
        self.assertEqual(params["content"], "正文与秘密")

    def test_invalid_values_are_not_converted_to_strings(self):
        clean = sanitize({"path": {"secret": "value"}, "offset": True, "column": "secret",
                          "limit": "secret", "timeout": float("inf"),
                          "max_results": float("nan"), "content": ["secret"],
                          "command": None, "keyword": 100})
        self.assertEqual(clean, {})
        self.assertEqual(sanitize(None), {})
        self.assertEqual(sanitize(["secret"]), {})

    def test_long_paths_are_bounded_and_empty_text_lengths_preserved(self):
        self.assertEqual(sanitize({"path": "长" * 1000, "content": "", "timeout": 1.5}),
                         {"path": "长" * 512, "content_chars": 0, "timeout": 1.5})


class PermissionAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.directory = self.workspace / ".harness"
        self.path = self.directory / "permission.log"
        self.audit = PermissionAuditLog(self.workspace)

    def record(self, **kwargs):
        values = {"tool": "write_file", "params": {"path": "src/a.py", "content": "secret"},
                  "decision": "ask", "matched_rule": "default", "risk": "write"}
        values.update(kwargs)
        self.audit.record(**values)

    def entries(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_records_metadata_and_confirmation_without_raw_parameters(self):
        self.assertFalse(self.directory.exists())
        self.record(call_id="call-1")
        self.record(call_id="call-1", event="confirmation", confirmation=True,
                    decision="allow", matched_rule="rule:0")
        entries = self.entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["params"], {"path": "src/a.py", "content_chars": 6})
        self.assertEqual(entries[0]["source"], "tool_executor")
        self.assertEqual(entries[0]["tool"], "write_file")
        self.assertEqual(entries[0]["risk"], "write")
        self.assertEqual(entries[0]["decision"], "ask")
        self.assertEqual(entries[0]["event"], "decision")
        self.assertIsNone(entries[0]["confirmation"])
        self.assertEqual(entries[1]["event"], "confirmation")
        self.assertTrue(entries[1]["confirmation"])
        self.assertEqual(entries[1]["decision"], "allow")
        self.assertEqual(entries[1]["matched_rule"], "rule:0")
        for entry in entries:
            self.assertEqual(entry["call_id"], "call-1")
            self.assertEqual(entry["session_id"], self.audit.session_id)
            self.assertEqual(datetime.fromisoformat(entry["timestamp"]).utcoffset().total_seconds(), 0)
        self.assertNotIn("secret", self.path.read_text(encoding="utf-8"))

    def test_log_is_one_json_value_per_line_even_for_newlines_in_paths(self):
        self.record(params={"path": "src/换行\nname.py", "command": "printf password"})
        entry, = self.entries()
        self.assertEqual(entry["params"]["path"], "src/换行\nname.py")
        self.assertNotIn("password", self.path.read_text(encoding="utf-8"))

    def test_new_sessions_use_distinct_ids_without_changing_old_entries(self):
        self.record()
        other = PermissionAuditLog(self.workspace)
        other.record("read_file", {"path": "a.py"}, "allow", "default", risk="read_only")
        entries = self.entries()
        UUID(self.audit.session_id)
        UUID(other.session_id)
        self.assertNotEqual(entries[0]["session_id"], entries[1]["session_id"])
        self.assertEqual(entries[1]["session_id"], other.session_id)

    def test_private_modes_are_created_and_restored(self):
        self.record()
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.directory.chmod(0o755)
        self.path.chmod(0o644)
        self.record()
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_concurrent_sessions_append_complete_noninterleaved_entries(self):
        audits = [self.audit, PermissionAuditLog(self.workspace)]

        def record_call(index):
            audits[index % 2].record("read_file", {"path": f"file-{index}"},
                                     "allow", "default", risk="read_only", call_id=str(index))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(record_call, range(80)))
        entries = self.entries()
        self.assertEqual(len(entries), 80)
        self.assertEqual({entry["call_id"] for entry in entries}, {str(index) for index in range(80)})
        self.assertEqual({entry["session_id"] for entry in entries},
                         {audit.session_id for audit in audits})

    def test_creation_race_opens_existing_log_without_recreating_it(self):
        real_open = os.open
        file_flags = []

        def open_after_concurrent_create(path, flags, *args, **kwargs):
            if path == "permission.log":
                file_flags.append(flags)
                if len(file_flags) == 1:
                    self.assertTrue(flags & os.O_EXCL)
                    other_fd = real_open(path, flags, *args, **kwargs)
                    try:
                        os.write(other_fd, b'{"previous": true}\n')
                    finally:
                        os.close(other_fd)
            return real_open(path, flags, *args, **kwargs)

        with patch("harness.audit.os.open", side_effect=open_after_concurrent_create):
            self.record()
        self.assertEqual(len(file_flags), 2)
        self.assertFalse(file_flags[1] & os.O_CREAT)
        self.assertTrue(file_flags[1] & os.O_NOFOLLOW)
        entries = self.entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0], {"previous": True})
        self.assertEqual(entries[1]["session_id"], self.audit.session_id)

    def test_log_disappearing_after_creation_conflict_fails_without_retry_or_write(self):
        self.directory.mkdir()
        self.path.write_text("unchanged\n", encoding="utf-8")
        real_open = os.open
        file_flags = []

        def disappearing_log(path, flags, *args, **kwargs):
            if path == "permission.log":
                file_flags.append(flags)
                if not flags & os.O_CREAT:
                    raise FileNotFoundError(2, "simulated disappearance")
            return real_open(path, flags, *args, **kwargs)

        with patch("harness.audit.os.open", side_effect=disappearing_log), \
                patch("harness.audit.os.write") as write:
            with self.assertRaises(AuditError):
                self.record()
        self.assertEqual(len(file_flags), 2)
        self.assertTrue(file_flags[0] & os.O_EXCL)
        self.assertFalse(file_flags[1] & os.O_CREAT)
        write.assert_not_called()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "unchanged\n")

    def test_symlink_directory_does_not_create_a_log_in_the_target(self):
        target = self.workspace / "other"
        target.mkdir()
        self.directory.symlink_to(target, target_is_directory=True)
        with self.assertRaises(AuditError):
            self.record()
        self.assertEqual(list(target.iterdir()), [])

    def test_symlink_file_is_not_written_or_chmodded(self):
        self.directory.mkdir()
        target = self.workspace / "other.txt"
        target.write_text("unchanged", encoding="utf-8")
        target.chmod(0o644)
        self.path.symlink_to(target)
        with self.assertRaises(AuditError):
            self.record()
        self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_hardlinked_log_cannot_modify_its_other_name(self):
        self.directory.mkdir()
        target = self.workspace / "other.txt"
        target.write_text("unchanged", encoding="utf-8")
        os.link(target, self.path)
        with self.assertRaises(AuditError):
            self.record()
        self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")

    def test_fifo_in_place_of_directory_is_rejected(self):
        os.mkfifo(self.directory)
        with self.assertRaises(AuditError):
            self.record()

    def test_fifo_log_is_rejected_before_open_and_cannot_block(self):
        self.directory.mkdir()
        os.mkfifo(self.path)
        with patch("harness.audit.os.open", wraps=os.open) as open_file:
            with self.assertRaises(AuditError):
                self.record()
        self.assertEqual(open_file.call_count, 2)

    def test_directory_or_socket_log_is_rejected(self):
        for kind in ("directory", "socket"):
            with self.subTest(kind=kind), TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                root.joinpath(".harness").mkdir()
                target = root / ".harness" / "permission.log"
                if kind == "directory":
                    target.mkdir()
                else:
                    sock = socket.socket(socket.AF_UNIX)
                    self.addCleanup(sock.close)
                    sock.bind(str(target))
                with self.assertRaises(AuditError):
                    PermissionAuditLog(root).record("read_file", {}, "allow", "default", risk="read_only")

    def test_opened_descriptor_is_checked_again_before_writing(self):
        real_fstat = os.fstat

        def hardlinked_info(fd):
            values = list(real_fstat(fd))
            values[3] = 2
            return os.stat_result(values)

        with patch("harness.audit.os.fstat", side_effect=hardlinked_info), \
                patch("harness.audit.os.write") as write:
            with self.assertRaises(AuditError):
                self.record()
        write.assert_not_called()

    def test_io_errors_are_sanitized_and_all_descriptors_are_closed(self):
        real_open = os.open
        opened = []

        def tracking_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd

        with patch("harness.audit.os.open", side_effect=tracking_open), \
                patch("harness.audit.os.write", side_effect=OSError("sensitive-storage-detail")):
            with self.assertRaises(AuditError) as raised:
                self.record()
        self.assertNotIn("sensitive-storage-detail", str(raised.exception))
        self.assertEqual(len(opened), 3)
        for fd in opened:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_failed_sync_cannot_be_reported_as_a_successful_audit(self):
        with patch("harness.audit.os.fsync", side_effect=OSError("sync failed")):
            with self.assertRaises(AuditError):
                self.record()


if __name__ == "__main__":
    unittest.main()
