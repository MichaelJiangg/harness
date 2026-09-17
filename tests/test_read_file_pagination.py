from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from harness.tools import execute_tool
from harness.tools.read_file import MAX_FILE_BYTES


class ReadFilePaginationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.path = self.root / "sample.txt"

    def read(self, **arguments):
        return execute_tool("read_file", {"path": self.path.name, **arguments}, workspace=self.root)

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
