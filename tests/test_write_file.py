import json
import os
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from harness.permissions import PermissionPolicy
from harness.tools.executor import ToolError
from harness.tools.write_file import DEFINITION, MAX_FILE_BYTES, execute


class WriteFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def write(self, path, content="内容"):
        return execute({"path": str(path), "content": content}, self.workspace)

    def assert_error(self, path, code, content="内容"):
        with self.assertRaises(ToolError) as caught:
            self.write(path, content)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(str(caught.exception))
        self.assertNotIn("private error details", str(caught.exception))

    def test_definition_requires_local_confirmation_and_only_path_and_content(self):
        self.assertEqual(PermissionPolicy().check(DEFINITION.name), "ask")
        definition = DEFINITION.to_deepseek()
        self.assertEqual(definition["function"]["name"], "write_file")
        schema = definition["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"path", "content"})
        self.assertEqual(schema["required"], ["path", "content"])
        self.assertEqual(schema["properties"]["path"]["minLength"], 1)
        self.assertNotIn("minLength", schema["properties"]["content"])
        self.assertIs(schema["additionalProperties"], False)
        self.assertNotIn("requires_confirmation", json.dumps(definition))

    def test_creates_missing_parent_directories_and_preserves_utf8_newlines(self):
        content = "第一行\r\n第二行\n末行。"
        result = self.write("notes/subdir/说明.txt", content)
        self.assertEqual((self.workspace / "notes/subdir/说明.txt").read_bytes(), content.encode("utf-8"))
        self.assertEqual(result["path"], "notes/subdir/说明.txt")
        self.assertEqual(result["bytes_written"], len(content.encode("utf-8")))
        self.assertTrue(result["message"])
        self.assertNotIn("content", result)
        self.assertNotIn(content, result["message"])

    def test_absolute_path_within_workspace_is_supported(self):
        result = self.write(self.workspace / "has spaces.txt", "文本")
        self.assertEqual(result["path"], "has spaces.txt")
        self.assertEqual((self.workspace / "has spaces.txt").read_text(encoding="utf-8"), "文本")

    def test_overwrites_entire_file_and_preserves_permissions(self):
        path = self.workspace / "existing.txt"
        path.write_bytes(b"much longer existing content")
        path.chmod(0o640)
        self.write("existing.txt", "短")
        self.assertEqual(path.read_bytes(), "短".encode("utf-8"))
        self.assertEqual(S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual([item.name for item in self.workspace.iterdir()], ["existing.txt"])

    def test_empty_content_creates_and_truncates_files(self):
        self.write("empty.txt", "")
        self.assertEqual((self.workspace / "empty.txt").read_bytes(), b"")
        (self.workspace / "existing.txt").write_bytes(b"old")
        result = self.write("existing.txt", "")
        self.assertEqual((self.workspace / "existing.txt").read_bytes(), b"")
        self.assertEqual(result["bytes_written"], 0)

    def test_invalid_path_semantics_have_no_side_effects(self):
        for path in ("", " \t\n", "bad\x00path"):
            with self.subTest(path=path):
                self.assert_error(path, "invalid_arguments")
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_invalid_utf8_and_binary_text_do_not_create_directories(self):
        for content in ("before\x00after", "\ud800"):
            with self.subTest(content=repr(content)):
                self.assert_error("new/sub/file.txt", "invalid_encoding", content)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_exact_byte_limit_is_allowed(self):
        content = "a" * MAX_FILE_BYTES
        result = self.write("limit.txt", content)
        self.assertEqual(result["bytes_written"], MAX_FILE_BYTES)
        self.assertEqual((self.workspace / "limit.txt").stat().st_size, MAX_FILE_BYTES)

    def test_byte_limit_uses_encoded_size_and_has_no_side_effects(self):
        content = "中" * (MAX_FILE_BYTES // 3 + 1)
        self.assertLess(len(content), MAX_FILE_BYTES)
        self.assert_error("new/sub/file.txt", "file_too_large", content)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_sensitive_paths_are_rejected_without_creating_directories(self):
        for path in (".env", ".env.local", "nested/.env.production", ".git/config", ".git", ".ENV", ".GIT/config"):
            with self.subTest(path=path):
                self.assert_error(path, "access_denied")
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_parent_and_absolute_outside_paths_are_rejected(self):
        for path in ("../outside.txt", self.root / "outside.txt"):
            with self.subTest(path=path):
                self.assert_error(path, "access_denied")
        self.assertFalse((self.root / "outside.txt").exists())

    def test_directory_parent_file_and_fifo_are_rejected(self):
        (self.workspace / "regular.txt").write_bytes(b"original")
        os.mkfifo(self.workspace / "pipe")
        for path in (".", "regular.txt/child.txt", "pipe"):
            with self.subTest(path=path):
                self.assert_error(path, "not_a_file")
        self.assertEqual((self.workspace / "regular.txt").read_bytes(), b"original")

    def test_symlink_outside_workspace_is_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_bytes(b"original")
        (self.workspace / "outside-link").symlink_to(outside)
        self.assert_error("outside-link", "access_denied")
        self.assertEqual(outside.read_bytes(), b"original")

    def test_parent_symlink_outside_workspace_is_rejected(self):
        (self.workspace / "outside-dir").symlink_to(self.root, target_is_directory=True)
        self.assert_error("outside-dir/new/sub/file.txt", "access_denied")
        self.assertFalse((self.root / "new").exists())

    def test_symlink_to_sensitive_path_is_rejected(self):
        for index, target in enumerate((".env", ".git/config")):
            link = self.workspace / f"link-{index}"
            link.symlink_to(self.workspace / target)
            with self.subTest(target=target):
                self.assert_error(link, "access_denied")
                self.assertFalse((self.workspace / target).exists())

    def test_ordinary_internal_symlink_updates_target_and_keeps_link(self):
        target = self.workspace / "target.txt"
        target.write_bytes(b"original")
        link = self.workspace / "alias.txt"
        link.symlink_to(target)
        result = self.write("alias.txt", "更新")
        self.assertEqual(result["path"], "target.txt")
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "更新")

    def test_symlink_loop_returns_write_error(self):
        link = self.workspace / "loop"
        link.symlink_to(link)
        self.assert_error(link, "write_error")

    def test_permission_failure_does_not_overwrite_original(self):
        path = self.workspace / "existing.txt"
        path.write_bytes(b"original")
        with patch("harness.tools.write_file.NamedTemporaryFile", side_effect=PermissionError("private error details")):
            self.assert_error(path, "permission_denied")
        self.assertEqual(path.read_bytes(), b"original")

    def test_mkdir_permission_failure_returns_normalized_error(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("private error details")):
            self.assert_error("new/sub/file.txt", "permission_denied")
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_replace_failure_preserves_original_and_removes_temporary_file(self):
        path = self.workspace / "existing.txt"
        path.write_bytes(b"original")
        with patch.object(Path, "replace", side_effect=OSError("private error details")):
            self.assert_error(path, "write_error")
        self.assertEqual(path.read_bytes(), b"original")
        self.assertEqual([item.name for item in self.workspace.iterdir()], ["existing.txt"])


if __name__ == "__main__":
    unittest.main()
