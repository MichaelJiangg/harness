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
        shutil.copytree(PROJECT / "harness", self.root / "harness",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.config = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        self.environ = {**os.environ, "PYTHONPATH": str(self.root), "PYTHONDONTWRITEBYTECODE": "1"}
        self.environ.pop("DEEPSEEK_API_KEY", None)

    def write_config(self, replacements=()):
        content = self.config
        for old, new in replacements:
            self.assertIn(old, content)
            content = content.replace(old, new, 1)
        (self.root / "pyproject.toml").write_text(content, encoding="utf-8")

    def run_python(self, code):
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=self.root,
                                env=self.environ, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_changed_settings_reach_client_engine_retry_display_and_pricing(self):
        self.write_config([
            ('name = "deepseek-flash"', 'name = "integration-model"'),
            ('endpoint = "https://api.deepseek.com/chat/completions"', 'endpoint = "https://example.invalid/chat/completions"'),
            ('request_timeout = 120', 'request_timeout = 17'),
            ('character_delay = 0.02', 'character_delay = 0.0'),
            ('max_requests = 20', 'max_requests = 7'),
            ('max_retries = 3', 'max_retries = 2'),
            ('retry_initial_delay = 1.0', 'retry_initial_delay = 0.25'),
            ('retry_backoff = 2.0', 'retry_backoff = 3.0'),
            ('max_chars = 64000', 'max_chars = 9000'),
            ('summary_chars = 2000', 'summary_chars = 600'),
            ('keep_recent_turns = 4', 'keep_recent_turns = 1'),
            ('max_compactions = 6', 'max_compactions = 1'),
            ('tool_result_chars = 12000', 'tool_result_chars = 6500'),
            ('input_miss_per_million = 0.15', 'input_miss_per_million = 2.0'),
            ('peak_weekdays = [0, 1, 2, 3, 4]', 'peak_weekdays = [6]'),
            ('peak_hours_utc = [[1, 4], [6, 10]]', 'peak_hours_utc = [[12, 13]]'),
        ])
        self.run_python('''
            from datetime import datetime, timezone
            from io import StringIO
            from threading import Event
            from unittest.mock import Mock, patch
            from urllib.error import URLError
            from harness.client import APIError, DeepSeekClient, DEFAULT_MODEL
            from harness.cli import run_cli
            from harness.engine import QueryState, query_loop
            from harness.usage import UsageLedger, format_cost

            opener = Mock(side_effect=URLError("offline test"))
            client = DeepSeekClient("fake-test-key", opener=opener)
            try:
                client.complete(model=DEFAULT_MODEL, messages=[], tools=[])
            except APIError:
                pass
            assert opener.call_args.args[0].full_url == "https://example.invalid/chat/completions"
            assert opener.call_args.kwargs["timeout"] == 17

            response = {"model": DEFAULT_MODEL, "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "完成"
            }}], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
            class Abort:
                def __init__(self): self.waits = []
                def is_set(self): return False
                def wait(self, delay): self.waits.append(delay); return False
            abort = Abort()
            model = Mock(complete=Mock(side_effect=[APIError("稍后重试", retryable=True),
                                                    APIError("稍后重试", retryable=True), response]))
            state = QueryState(model, UsageLedger(), tools=[], abort=abort)
            assert (state.model, state.max_requests, state.max_retries) == ("integration-model", 7, 2)
            assert (state.context_limit, state.summary_limit, state.keep_recent_turns,
                    state.max_compactions, state.tool_result_limit) == (9000, 600, 1, 1, 6500)
            assert (state.swarm_max_requests, state.swarm_max_role_requests) == (120, 40)
            state.messages.append({"role": "user", "content": "继续"})
            assert query_loop(state) == "完成"
            assert abort.waits == [0.25, 0.75]
            assert model.complete.call_count == 3
            assert state.ledger.summary()["requests"] == 3

            ledger = UsageLedger()
            usage = {"prompt_tokens": 1000000, "completion_tokens": 0, "total_tokens": 1000000}
            weekday = datetime(2026, 9, 18, 2, tzinfo=timezone.utc).timestamp()
            sunday = datetime(2026, 9, 20, 12, tzinfo=timezone.utc).timestamp()
            assert ledger.record(model=DEFAULT_MODEL, turn=1, usage=usage, created=weekday)["estimated_cost"] == 2.0
            assert ledger.record(model=DEFAULT_MODEL, turn=2, usage=usage, created=sunday)["rate_period"] == "peak"
            assert "周日 12:00–13:00" in format_cost(ledger)

            class Terminal(StringIO):
                def isatty(self): return True
            stop = Event()
            def stream(**request):
                request["on_text"]("完成")
                return response
            with patch("harness.cli.Event", return_value=stop), patch.object(stop, "wait", wraps=stop.wait) as wait:
                run_cli(Mock(complete=Mock(side_effect=stream)), input_stream=Terminal("你好\\n"),
                        output=Terminal(), error_output=StringIO())
                wait.assert_not_called()
        ''')

    def test_invalid_or_missing_config_stops_cli_before_api_key_loading(self):
        for content in (None, "[tool.harness\n", self.config.replace('max_retries = 3', 'max_retries = -1')):
            with self.subTest(content="missing" if content is None else "invalid"):
                path = self.root / "pyproject.toml"
                # 缺失场景必须先跑，不需要删除任何项目文件。
                if content is not None:
                    path.write_text(content, encoding="utf-8")
                result = subprocess.run([sys.executable, "-m", "harness"], cwd=self.root,
                                        env=self.environ, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1)
                self.assertIn("pyproject.toml", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("DEEPSEEK_API_KEY", result.stderr)

    def test_help_uses_configuration_without_loading_real_credentials(self):
        self.write_config([('[tool.harness.tools.bash]\ndefault_timeout = 30',
                           '[tool.harness.tools.bash]\ndefault_timeout = 9'),
                           ('max_timeout = 120', 'max_timeout = 15')])
        result = subprocess.run([sys.executable, "-m", "harness", "--help"], cwd=self.root,
                                env=self.environ, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("默认超时 9 秒，最多 15 秒", result.stdout)


if __name__ == "__main__":
    unittest.main()
