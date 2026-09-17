import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from harness.tools import create_tool_executor, execute_tool, get_tool_definitions


class GrepTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def write(self, name, content):
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return path

    def search(self, keyword="needle", **arguments):
        return execute_tool("grep", {"keyword": keyword, **arguments}, workspace=self.workspace)

    def assert_success(self, result):
        self.assertEqual(result["status"], "success", result)
        self.assertIs(result["executed"], True)
        self.assertEqual(result["tool"], "grep")
        self.assertEqual(result["returned_count"], len(result["matches"]))
        self.assertIs(result["cancelled"], False)
        self.assertTrue(result["message"])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 6000)

    def assert_error(self, result, code):
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(result["code"], code)
        self.assertEqual(result["tool"], "grep")
        self.assertIs(result["executed"], False)
        self.assertTrue(result["message"])

    def test_schema_is_discovered_without_model_approval_fields(self):
        definition = next(item for item in get_tool_definitions() if item["function"]["name"] == "grep")
        self.assertEqual(definition["type"], "function")
        schema = definition["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"keyword", "path", "glob", "max_results"})
        self.assertEqual(schema["required"], ["keyword"])
        self.assertEqual(schema["properties"]["max_results"]["minimum"], 1)
        self.assertEqual(schema["properties"]["max_results"]["maximum"], 500)
        self.assertIs(schema["additionalProperties"], False)

    def test_matching_lines_include_relative_path_one_based_number_and_content(self):
        self.write("src/main.py", "first\nneedle = 1\n  needle again  \nlast")
        result = self.search()
        self.assert_success(result)
        self.assertEqual(result["matches"], [
            {"path": "src/main.py", "line_number": 2, "content": "needle = 1", "line_truncated": False},
            {"path": "src/main.py", "line_number": 3, "content": "  needle again  ", "line_truncated": False},
        ])
        self.assertIs(result["truncated"], False)

    def test_one_line_is_returned_once_even_with_repeated_occurrences(self):
        self.write("sample.txt", "needle needle needle\n")
        result = self.search()
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 1)

    def test_regex_characters_are_literal(self):
        self.write("sample.txt", "a.b[0]*\naxb000\na.b[0]* again\n")
        result = self.search("a.b[0]*")
        self.assert_success(result)
        self.assertEqual([item["line_number"] for item in result["matches"]], [1, 3])

    def test_unicode_keyword_and_case_sensitive_matching(self):
        self.write("说明.py", "关键字 Apple\n关键字 apple\n关键词 APPLE\n")
        result = self.search("关键字")
        self.assert_success(result)
        self.assertEqual([item["line_number"] for item in result["matches"]], [1, 2])
        self.assertEqual([item["line_number"] for item in self.search("apple")["matches"]], [2])

    def test_crlf_empty_lines_and_last_line_keep_correct_line_numbers(self):
        self.write("sample.txt", b"first\r\n\r\nneedle\r\nlast needle")
        result = self.search()
        self.assert_success(result)
        self.assertEqual([(item["line_number"], item["content"]) for item in result["matches"]],
                         [(3, "needle"), (4, "last needle")])

    def test_form_feed_is_content_and_does_not_increment_line_number(self):
        self.write("sample.py", b"\x0cneedle = 1\nneedle = 2\n")
        result = self.search()
        self.assert_success(result)
        self.assertEqual([(item["line_number"], item["content"]) for item in result["matches"]],
                         [(1, "\x0cneedle = 1"), (2, "needle = 2")])

    def test_glob_filters_file_names_in_nested_directories(self):
        self.write("one.py", "needle")
        self.write("src/deep/two.py", "needle")
        self.write("src/three.js", "needle")
        self.write("README.md", "needle")
        result = self.search(glob="*.py")
        self.assert_success(result)
        self.assertEqual({item["path"] for item in result["matches"]}, {"one.py", "src/deep/two.py"})

    def test_path_limits_search_to_requested_subdirectory(self):
        self.write("outside.py", "needle")
        self.write("src/inside.py", "needle")
        result = self.search(path="src")
        self.assert_success(result)
        self.assertEqual([item["path"] for item in result["matches"]], ["src/inside.py"])

    def test_relative_and_absolute_single_file_paths_are_supported(self):
        path = self.write("src/one.py", "needle")
        self.write("src/two.py", "needle")
        for requested in ("src/one.py", str(path)):
            with self.subTest(path=requested):
                result = self.search(path=requested)
                self.assert_success(result)
                self.assertEqual([item["path"] for item in result["matches"]], ["src/one.py"])

    def test_glob_also_filters_explicit_file(self):
        self.write("one.js", "needle")
        result = self.search(path="one.js", glob="*.py")
        self.assert_success(result)
        self.assertEqual(result["matches"], [])

    def test_files_and_directories_are_scanned_in_stable_order(self):
        for name in ("z/file.py", "b.py", "a/file.py", "a.py"):
            self.write(name, "needle")
        result = self.search()
        self.assert_success(result)
        self.assertEqual([item["path"] for item in result["matches"]],
                         ["a.py", "b.py", "a/file.py", "z/file.py"])

    def test_no_matches_and_empty_files_are_successful(self):
        self.write("empty.py", "")
        self.write("other.py", "different text")
        result = self.search()
        self.assert_success(result)
        self.assertEqual(result["matches"], [])
        self.assertIs(result["truncated"], False)
        self.assertEqual(result["skipped_files"], 0)

    def test_result_limit_returns_complete_entries_and_marks_extra_matches(self):
        self.write("many.py", "needle 1\nneedle 2\nneedle 3\n")
        result = self.search(max_results=2)
        self.assert_success(result)
        self.assertEqual([item["line_number"] for item in result["matches"]], [1, 2])
        self.assertIs(result["truncated"], True)

    def test_exact_result_limit_without_extra_match_is_not_truncated(self):
        self.write("exact.py", "needle 1\nother\nneedle 2\n")
        result = self.search(max_results=2)
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 2)
        self.assertIs(result["truncated"], False)

    def test_long_line_retains_late_keyword_instead_of_only_its_prefix(self):
        self.write("long.py", "x" * 20000 + "needle" + "y" * 20000)
        result = self.search()
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 1)
        match = result["matches"][0]
        self.assertIn("needle", match["content"])
        self.assertIs(match["line_truncated"], True)
        self.assertEqual(match["line_number"], 1)

    def test_json_escaping_is_included_in_result_budget(self):
        line = "needle " + '\\"\t' * 600
        self.write("escaped.py", (line + "\n") * 20)
        result = self.search(max_results=500)
        self.assert_success(result)
        self.assertGreater(result["returned_count"], 0)
        self.assertLess(result["returned_count"], 20)
        self.assertIs(result["truncated"], True)
        for match in result["matches"]:
            self.assertEqual(set(match), {"path", "line_number", "content", "line_truncated"})
            self.assertIn("needle", match["content"])

    def test_long_keyword_is_preserved_in_truncated_line(self):
        keyword = "K" * 1200
        self.write("long.py", "x" * 2000 + keyword + "y" * 2000)
        result = self.search(keyword)
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 1)
        self.assertIn(keyword, result["matches"][0]["content"])
        self.assertIs(result["matches"][0]["line_truncated"], True)

    def test_expensive_json_escapes_still_return_first_matching_line(self):
        self.write("escaped.py", "needle" + "\x01" * 994)
        result = self.search()
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 1)
        self.assertIn("needle", result["matches"][0]["content"])
        self.assertIs(result["matches"][0]["line_truncated"], True)

    def test_keyword_too_large_for_a_result_returns_actionable_error(self):
        result = self.search("K" * 10000)
        self.assert_error(result, "invalid_arguments")

    def test_invalid_arguments_are_rejected_before_reading(self):
        invalid = (
            None, [], {}, {"keyword": None}, {"keyword": 1}, {"keyword": ""},
            {"keyword": "a\nb"}, {"keyword": "a\rb"}, {"keyword": "a\x00b"},
            {"keyword": "x", "path": ""}, {"keyword": "x", "path": 1},
            {"keyword": "x", "glob": "src/*.py"}, {"keyword": "x", "glob": "*\x00"},
            {"keyword": "x", "glob": "*\n"}, {"keyword": "x", "glob": True},
            {"keyword": "x", "max_results": 0}, {"keyword": "x", "max_results": -1},
            {"keyword": "x", "max_results": 501}, {"keyword": "x", "max_results": True},
            {"keyword": "x", "max_results": 1.0}, {"keyword": "x", "max_results": "1"},
            {"keyword": "x", "confirmed": True},
        )
        with patch.object(Path, "open") as open_file:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    self.assert_error(execute_tool("grep", arguments, workspace=self.workspace), "invalid_arguments")
        open_file.assert_not_called()

    def test_missing_root_is_a_tool_error(self):
        self.assert_error(self.search(path="missing"), "not_found")

    def test_outside_and_sensitive_roots_are_rejected_without_opening(self):
        paths = ("..", str(self.root), ".env", ".env.local", "nested/.ENV.production", ".git", ".GIT/config")
        with patch.object(Path, "open") as open_file:
            for path in paths:
                with self.subTest(path=path):
                    self.assert_error(self.search(path=path), "access_denied")
        open_file.assert_not_called()

    def test_non_regular_explicit_root_is_rejected_before_opening(self):
        os.mkfifo(self.workspace / "pipe")
        with patch.object(Path, "open") as open_file:
            self.assert_error(self.search(path="pipe"), "not_a_file")
        open_file.assert_not_called()

    def test_symlink_to_outside_root_is_rejected(self):
        (self.workspace / "alias").symlink_to(self.root / "outside")
        self.assert_error(self.search(path="alias"), "access_denied")

    def test_symlink_to_sensitive_root_is_rejected(self):
        for index, name in enumerate((".env", ".git/config")):
            alias = self.workspace / f"alias-{index}"
            alias.symlink_to(self.workspace / name)
            with self.subTest(name=name), patch.object(Path, "open") as open_file:
                self.assert_error(self.search(path=alias.name), "access_denied")
                open_file.assert_not_called()

    def test_recursive_search_does_not_follow_directory_symlinks(self):
        self.write("real/file.py", "needle")
        (self.workspace / "alias").symlink_to(self.workspace / "real", target_is_directory=True)
        (self.workspace / "real/back").symlink_to(self.workspace, target_is_directory=True)
        result = self.search()
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["matches"][0]["path"], "real/file.py")
        self.assertGreaterEqual(result["skipped_directories"], 2)

    def test_safe_file_symlink_can_be_searched(self):
        self.write("real.txt", "needle")
        (self.workspace / "alias.txt").symlink_to(self.workspace / "real.txt")
        result = self.search(path="alias.txt")
        self.assert_success(result)
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["matches"][0]["content"], "needle")
        self.assertFalse(Path(result["matches"][0]["path"]).is_absolute())

    def test_recursive_search_skips_unsafe_file_symlinks(self):
        outside = self.root / "outside.txt"
        outside.write_text("needle outside", encoding="utf-8")
        (self.workspace / "outside-link.txt").symlink_to(outside)
        (self.workspace / "sensitive-link.txt").symlink_to(self.workspace / ".env")
        self.write("safe.py", "needle safe")
        result = self.search()
        self.assert_success(result)
        self.assertEqual([item["content"] for item in result["matches"]], ["needle safe"])
        self.assertGreaterEqual(result["skipped_files"], 2)

    def test_invalid_binary_and_oversized_files_are_skipped(self):
        self.write("invalid.txt", b"needle\n\xff")
        self.write("binary.txt", b"needle\x00binary")
        self.write("large.txt", b"needle\n" + b"x" * (1024 * 1024))
        self.write("valid.py", "needle valid")
        result = self.search()
        self.assert_success(result)
        self.assertEqual([item["path"] for item in result["matches"]], ["valid.py"])
        self.assertEqual(result["skipped_files"], 3)

    def test_permission_failure_skips_file_and_does_not_leak_exception(self):
        blocked = self.write("blocked.py", "needle blocked")
        self.write("valid.py", "needle valid")
        original_open = Path.open

        def open_file(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("private error details")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_file):
            result = self.search()
        self.assert_success(result)
        self.assertEqual([item["path"] for item in result["matches"]], ["valid.py"])
        self.assertEqual(result["skipped_files"], 1)
        self.assertNotIn("private error details", json.dumps(result))

    def test_unreadable_subdirectory_is_counted_and_other_files_are_searched(self):
        self.write("blocked/hidden.py", "needle hidden")
        self.write("visible.py", "needle visible")
        original_scandir = os.scandir

        def scandir(path):
            if Path(path) == self.workspace / "blocked":
                raise PermissionError("private directory details")
            return original_scandir(path)

        with patch.object(os, "scandir", scandir):
            result = self.search()
        self.assert_success(result)
        self.assertEqual([item["path"] for item in result["matches"]], ["visible.py"])
        self.assertEqual(result["skipped_directories"], 1)
        self.assertNotIn("private directory details", json.dumps(result))

    def test_read_only_search_never_requests_confirmation(self):
        self.write("sample.py", "needle")
        confirm = Mock(side_effect=AssertionError("read-only tools must not ask for approval"))
        executor = create_tool_executor(self.workspace, confirm=confirm)
        result = executor("grep", {"keyword": "needle"})
        self.assert_success(result)
        confirm.assert_not_called()

    def test_cancellation_before_search_does_not_open_files(self):
        abort = Event()
        abort.set()
        with patch.object(Path, "open") as open_file:
            result = execute_tool("grep", {"keyword": "needle"}, workspace=self.workspace, abort=abort)
        self.assert_error(result, "cancelled")
        open_file.assert_not_called()

    def test_cancellation_during_search_returns_partial_structured_result(self):
        self.write("a.py", "needle")
        self.write("b.py", "needle")
        abort = Event()
        original_open = Path.open

        def open_file(path, *args, **kwargs):
            abort.set()
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_file):
            result = execute_tool("grep", {"keyword": "needle"}, workspace=self.workspace, abort=abort)
        self.assertEqual(result["status"], "error")
        self.assertIs(result["cancelled"], True)
        self.assertEqual(result["returned_count"], len(result["matches"]))
        self.assertLessEqual(result["returned_count"], 1)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 6000)

    def test_executor_keeps_startup_workspace(self):
        self.write("sample.py", "needle initial")
        (self.root / "sample.py").write_text("needle elsewhere", encoding="utf-8")
        with patch.object(Path, "cwd", return_value=self.workspace):
            executor = create_tool_executor()
        with patch.object(Path, "cwd", return_value=self.root) as cwd:
            result = executor("grep", {"keyword": "needle"})
        cwd.assert_not_called()
        self.assert_success(result)
        self.assertEqual([item["content"] for item in result["matches"]], ["needle initial"])


if __name__ == "__main__":
    unittest.main()
