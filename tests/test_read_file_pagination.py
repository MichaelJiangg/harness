import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.orchestration import query_context
from harness.context import truncate_tool_result
from harness.tools import execute_tool
from harness.tools.read_file import MAX_FILE_BYTES


class ReadFilePaginationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.path = self.root / "sample.txt"

    def read(self, *, budget=None, **arguments):
        if budget is None:
            return execute_tool("read_file", {"path": self.path.name, **arguments}, workspace=self.root)
        with query_context(SimpleNamespace(tool_result_limit=budget)):
            return execute_tool("read_file", {"path": self.path.name, **arguments}, workspace=self.root)

    def read_all(self, *, budget, **arguments):
        pages = []
        cursor = (arguments.pop("offset", 0), arguments.pop("column", 0))
        for _ in range(1000):
            result = self.read(budget=budget, offset=cursor[0], column=cursor[1], **arguments)
            self.assertEqual(result["status"], "success")
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), budget)
            self.assertEqual(json.loads(truncate_tool_result(result, budget)), result)
            self.assertEqual((result["offset"], result["column"]), cursor)
            pages.append(result)
            if result["eof"]:
                return pages
            following = (result["next_offset"], result["next_column"])
            self.assertGreater(following, cursor)
            self.assertTrue(result["content"])
            cursor = following
        self.fail("分页游标未能在有限次数内到达文件末尾。")

    def assert_content(self, expected, **arguments):
        result = self.read(**arguments)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], expected)
        self.assertEqual(result["path"], self.path.name)
        self.assertTrue(result["executed"])

    def test_defaults_preserve_all_lines_and_original_newlines(self):
        content = "第一行\r\n第二行\n最后一行"
        self.path.write_bytes(content.encode("utf-8"))
        self.assert_content(content)
        self.assert_content(content, offset=0)

    def test_offset_is_zero_based_and_limit_counts_lines(self):
        self.path.write_bytes("一\n二\n三\n四\n".encode("utf-8"))
        self.assert_content("一\n", limit=1)
        self.assert_content("二\n三\n", offset=1, limit=2)
        self.assert_content("四\n", offset=3, limit=1)

    def test_omitted_limit_returns_remaining_lines(self):
        self.path.write_bytes(b"first\nsecond\nlast")
        self.assert_content("second\nlast", offset=1)

    def test_limit_beyond_eof_returns_available_lines(self):
        self.path.write_bytes(b"first\nsecond\nlast")
        self.assert_content("second\nlast", offset=1, limit=100)

    def test_offset_at_or_beyond_eof_returns_empty_content(self):
        self.path.write_bytes(b"first\nsecond\n")
        for offset in (2, 3, 10**30):
            with self.subTest(offset=offset):
                self.assert_content("", offset=offset)
                self.assert_content("", offset=offset, limit=1)

    def test_empty_file_has_no_lines(self):
        self.path.write_bytes(b"")
        self.assert_content("")
        self.assert_content("", offset=0, limit=1)
        self.assert_content("", offset=5)

    def test_blank_lines_and_crlf_are_preserved(self):
        self.path.write_bytes(b"one\r\n\r\nthree\r\nlast")
        self.assert_content("\r\nthree\r\n", offset=1, limit=2)
        self.assert_content("last", offset=3, limit=1)

    def test_single_line_without_terminator(self):
        self.path.write_bytes("中文与 emoji 🙂".encode("utf-8"))
        self.assert_content("中文与 emoji 🙂", offset=0, limit=1)
        self.assert_content("", offset=1)

    def test_invalid_pagination_parameters_do_not_open_files(self):
        invalid = (
            {"offset": -1}, {"offset": True}, {"offset": 1.0},
            {"offset": "1"}, {"offset": None}, {"offset": []},
            {"limit": 0}, {"limit": -1}, {"limit": False},
            {"limit": 1.0}, {"limit": "1"}, {"limit": None}, {"limit": {}},
            {"column": -1}, {"column": True}, {"column": 1.0},
            {"column": "1"}, {"column": None}, {"column": []},
        )
        with patch.object(Path, "open") as open_file:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    result = self.read(**arguments)
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["code"], "invalid_arguments")
                    self.assertNotIn("content", result)
        open_file.assert_not_called()

    def test_pagination_does_not_bypass_whole_file_size_limit(self):
        self.path.write_bytes(b"a\n" + b"b" * MAX_FILE_BYTES)
        result = self.read(offset=0, limit=1)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "file_too_large")

    def test_exact_file_size_limit_can_be_paginated(self):
        self.path.write_bytes(b"a\n" + b"b" * (MAX_FILE_BYTES - 2))
        self.assert_content("a\n", offset=0, limit=1)

    def test_invalid_utf8_outside_page_is_still_rejected(self):
        self.path.write_bytes(b"valid\n\xff")
        result = self.read(offset=0, limit=1)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "invalid_encoding")

    def test_default_pages_stop_at_200_lines_and_resume_to_eof(self):
        lines = [f"line {index}\n" for index in range(451)]
        self.path.write_bytes("".join(lines).encode("utf-8"))
        pages = self.read_all(budget=12000)
        self.assertEqual([len(page["content"].splitlines()) for page in pages], [200, 200, 51])
        self.assertEqual([page["next_offset"] for page in pages], [200, 400, 451])
        self.assertEqual([page["next_column"] for page in pages], [0, 0, 0])
        self.assertTrue(all(page["total_lines"] == 451 for page in pages))
        self.assertEqual("".join(page["content"] for page in pages), "".join(lines))

    def test_character_budget_preserves_whole_lines_and_all_json_metadata(self):
        lines = [f"中文🙂 {index} \"\\\t内容\n" for index in range(50)]
        content = "".join(lines)
        self.path.write_bytes(content.encode("utf-8"))
        pages = self.read_all(budget=480)
        self.assertGreater(len(pages), 1)
        self.assertTrue(all(page["content"].endswith("\n") for page in pages))
        self.assertTrue(all(page["next_column"] == 0 for page in pages))
        self.assertEqual("".join(page["content"] for page in pages), content)

    def test_single_long_line_uses_character_cursor_without_losing_middle(self):
        content = "中文🙂\"\\\t" * 300 + "\r\n尾行\n"
        self.path.write_bytes(content.encode("utf-8"))
        pages = self.read_all(budget=450)
        self.assertGreater(len(pages), 2)
        self.assertTrue(any(page["next_offset"] == 0 and page["next_column"] > 0 for page in pages))
        self.assertEqual("".join(page["content"] for page in pages), content)
        self.assertEqual((pages[-1]["next_offset"], pages[-1]["next_column"]), (2, 0))

    def test_long_following_line_is_left_for_next_page_before_character_splitting(self):
        content = "short\n" + "长🙂" * 500 + "\n"
        self.path.write_bytes(content.encode("utf-8"))
        pages = self.read_all(budget=420)
        self.assertEqual(pages[0]["content"], "short\n")
        self.assertEqual((pages[0]["next_offset"], pages[0]["next_column"]), (1, 0))
        self.assertEqual("".join(page["content"] for page in pages), content)

    def test_explicit_column_counts_unicode_characters_and_preserves_crlf_remainder(self):
        self.path.write_bytes("甲🙂\r\n乙\n".encode("utf-8"))
        first = self.read(offset=0, column=1, limit=1)
        self.assertEqual(first["content"], "🙂\r\n")
        self.assertEqual((first["next_offset"], first["next_column"]), (1, 0))
        self.assertFalse(first["eof"])
        second = self.read(offset=0, column=3)
        self.assertEqual(second["content"], "\n乙\n")
        self.assertTrue(second["eof"])

    def test_column_outside_line_or_nonzero_at_eof_is_invalid(self):
        self.path.write_bytes(b"abc\n")
        for arguments in ({"column": 4}, {"column": 99}, {"offset": 1, "column": 1},
                          {"offset": 9, "column": 3}):
            with self.subTest(arguments=arguments):
                result = self.read(**arguments)
                self.assertEqual(result["code"], "invalid_arguments")
                self.assertFalse(result["executed"])

    def test_empty_and_past_eof_pages_have_explicit_completion_and_no_content(self):
        for content, offset, total in ((b"", 0, 0), (b"abc\n", 1, 1), (b"abc\n", 20, 1)):
            with self.subTest(content=content, offset=offset):
                self.path.write_bytes(content)
                result = self.read(offset=offset)
                self.assertEqual(result["content"], "")
                self.assertTrue(result["eof"])
                self.assertEqual(result["total_lines"], total)
                self.assertEqual(result["offset"], offset)
                self.assertEqual(result["next_column"], 0)

    def test_explicit_line_limit_returns_requested_range_without_forcing_full_file(self):
        self.path.write_bytes(b"zero\none\ntwo\nthree\n")
        result = self.read(offset=1, limit=2)
        self.assertEqual(result["content"], "one\ntwo\n")
        self.assertEqual((result["next_offset"], result["next_column"]), (3, 0))
        self.assertFalse(result["eof"])

    def test_without_query_context_uses_configured_tool_result_budget(self):
        self.path.write_bytes(b"x" * 3000)
        with patch("harness.tools.read_file.get_settings", return_value={"context": {"tool_result_chars": 400}}):
            result = self.read()
        self.assertEqual(result["status"], "success")
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 400)
        self.assertFalse(result["eof"])
        self.assertGreater(result["next_column"], 0)

    def test_budget_too_small_for_metadata_returns_error_instead_of_stalled_cursor(self):
        self.path.write_bytes(b"x")
        result = self.read(budget=32)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["executed"])
        self.assertNotIn("content", result)
        self.assertNotIn("next_offset", result)
