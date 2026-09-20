import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap
import unittest


PROJECT = Path(__file__).resolve().parent.parent


class SettingsIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        shutil.copytree(
            PROJECT / "harness",
            self.root / "harness",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        self.environ = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.environ.pop("DEEPSEEK_API_KEY", None)
        self.environ.pop("GLM_API_KEY", None)
        self.environ.pop("HARNESS_MODEL", None)
        self.environ.pop("HARNESS_MAX_TURNS", None)

    def write_config(self, content):
        path = self.root / ".harness" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def run_python(self, code, *args):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code), *args],
            cwd=self.root,
            env=self.environ,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_config_file_reaches_client_engine_and_tools(self):
        self.write_config('''
[model]
name = "integration-model"
endpoint = "https://example.invalid/chat/completions"
request_timeout = 17

[engine]
max_turns = 7
max_requests = 7
max_retries = 2
retry_initial_delay = 0.25
retry_backoff = 3.0

[context]
max_chars = 9000
summary_chars = 600
keep_recent_turns = 1
max_compactions = 1
tool_result_chars = 6500

[tools.bash]
default_timeout = 9
max_timeout = 15
''')
        result = self.run_python('''
            from harness.client import DeepSeekClient
            from harness.engine import QueryState
            from harness.usage import UsageLedger

            client = DeepSeekClient("fake-test-key")
            assert client.model == "integration-model"
            assert client.endpoint == "https://example.invalid/chat/completions"
            assert client.timeout == 17
            state = QueryState(client, UsageLedger(), tools=[])
            assert state.max_requests == 7
            assert state.max_retries == 2
            assert state.context_limit == 9000
            assert state.summary_limit == 600
            assert state.keep_recent_turns == 1
            assert state.max_compactions == 1
            assert state.tool_result_limit == 6500
            print("ok")
        ''')
        self.assertIn("ok", result.stdout)

    def test_cli_overrides_config_and_show_config_prints_final_values(self):
        self.write_config('[engine]\nmax_turns = 20\n')
        result = subprocess.run(
            [sys.executable, "-m", "harness", "--max-turns", "3", "--show-config"],
            cwd=self.root,
            env=self.environ,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("'max_turns': 3", result.stdout)

    def test_invalid_config_stops_before_api_key_loading(self):
        self.write_config('[engine]\nmax_turns = "invalid"\n')
        result = subprocess.run(
            [sys.executable, "-m", "harness"],
            cwd=self.root,
            env=self.environ,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("DEEPSEEK_API_KEY", result.stderr)

    def test_missing_config_still_requires_api_key(self):
        result = subprocess.run(
            [sys.executable, "-m", "harness"],
            cwd=self.root,
            env=self.environ,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("DEEPSEEK_API_KEY 或 GLM_API_KEY", result.stderr)


if __name__ == "__main__":
    unittest.main()
