import os
from pathlib import Path
import unittest
from unittest.mock import patch

from harness import __main__ as entrypoint
from harness import config


class ConfigTests(unittest.TestCase):
    def test_loads_last_key_without_expanding_or_mutating_environment(self):
        content = (
            "OTHER_SETTING=ignored\n"
            "DEEPSEEK_API_KEY=first-test-value\n"
            "DEEPSEEK_API_KEY='${OTHER_SETTING}$(echo ignored)'\n"
        )
        environment = {"OTHER_SETTING": "original"}
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(Path, "read_text", return_value=content):
                self.assertEqual(
                    config.load_api_key(), "${OTHER_SETTING}$(echo ignored)"
                )
            self.assertEqual(dict(os.environ), environment)

    def test_environment_takes_precedence_including_explicit_empty_value(self):
        for value in ("environment-test-value", ""):
            with self.subTest(value=value):
                with patch.object(Path, "read_text") as read_text:
                    self.assertEqual(
                        config.load_api_key(environ={"DEEPSEEK_API_KEY": value}),
                        value,
                    )
                read_text.assert_not_called()

    def test_missing_file_or_missing_key_returns_none(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(config.load_api_key(environ={}))
        with patch.object(Path, "read_text", return_value="# comment\nOTHER=value\n"):
            self.assertIsNone(config.load_api_key(environ={}))

    def test_supported_value_syntax_and_custom_file_path(self):
        examples = (
            ("DEEPSEEK_API_KEY=test-value", "test-value"),
            ("DEEPSEEK_API_KEY=", ""),
            ("DEEPSEEK_API_KEY= # optional value", ""),
            ("DEEPSEEK_API_KEY=''", ""),
            ("DEEPSEEK_API_KEY='test # value' # comment", "test # value"),
            ('DEEPSEEK_API_KEY="test value" # comment', "test value"),
            (" export DEEPSEEK_API_KEY = test-value # comment", "test-value"),
        )
        path = Path("/unused/harness-config-test")
        for content, expected in examples:
            with self.subTest(content=content):
                with patch.object(Path, "read_text", autospec=True, return_value=content) as read_text:
                    self.assertEqual(config.load_api_key(env_file=path, environ={}), expected)
                read_text.assert_called_once_with(path, encoding="utf-8-sig")

    def test_invalid_assignment_reports_line_without_disclosing_value(self):
        for line in (
            "DEEPSEEK_API_KEY='private-test-value",
            "DEEPSEEK_API_KEY=private-test-value unexpected",
            "DEEPSEEK_API_KEY",
        ):
            with self.subTest(line=line):
                with patch.object(Path, "read_text", return_value=f"# comment\n{line}"):
                    with self.assertRaises(ValueError) as raised:
                        config.load_api_key(environ={})
                message = str(raised.exception)
                self.assertIn("第 2 行", message)
                self.assertNotIn("private-test-value", message)

    def test_file_read_errors_are_sanitized(self):
        for error in (PermissionError("private-test-value"), UnicodeError("private-test-value")):
            with self.subTest(error_type=type(error).__name__):
                with patch.object(Path, "read_text", side_effect=error):
                    with self.assertRaises(ValueError) as raised:
                        config.load_api_key(environ={})
                self.assertIn("无法读取", str(raised.exception))
                self.assertNotIn("private-test-value", str(raised.exception))

    def test_default_file_is_relative_to_project_not_working_directory(self):
        expected_path = Path(config.__file__).resolve().parent.parent / ".env"
        with patch("os.getcwd", return_value="/unrelated/working/directory"):
            with patch.object(Path, "read_text", autospec=True, return_value="DEEPSEEK_API_KEY=test-value") as read_text:
                self.assertEqual(config.load_api_key(environ={}), "test-value")
        read_text.assert_called_once_with(expected_path, encoding="utf-8-sig")

    def test_main_passes_loaded_key_to_client(self):
        with patch("sys.argv", ["harness"]):
            with patch.object(entrypoint, "load_api_key", return_value="test-value") as load:
                with patch.object(entrypoint, "DeepSeekClient") as client:
                    with patch.object(entrypoint, "run_cli") as run_cli:
                        self.assertEqual(entrypoint.main(), 0)
        load.assert_called_once_with()
        client.assert_called_once_with("test-value")
        run_cli.assert_called_once_with(client.return_value)


if __name__ == "__main__":
    unittest.main()
