from copy import deepcopy
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from harness import config


class ProjectConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_file = Path(config.__file__).resolve().parent.parent / "pyproject.toml"
        cls.source = cls.project_file.read_text(encoding="utf-8")
        cls.defaults = config.load_settings(cls.project_file)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "pyproject.toml"
        self.path.write_text(self.source, encoding="utf-8")

    def modified(self, *changes):
        source = self.source
        for old, new in changes:
            self.assertIn(old, source)
            source = source.replace(old, new, 1)
        self.path.write_text(source, encoding="utf-8")
        return self.path

    def load_document(self, settings):
        with patch.object(config.tomllib, "load", return_value={"tool": {"harness": settings}}):
            return config.load_settings(self.path)

    def test_metadata_has_python_version_and_no_dependencies(self):
        document = config.tomllib.loads(self.source)
        project = document["project"]
        self.assertEqual(project["name"], "harness")
        self.assertEqual(project["version"], "0.6.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["urls"]["Repository"], "https://github.com/MichaelJiangg/harness")

    def test_initial_defaults_preserve_existing_runtime_values(self):
        settings = config.load_settings(self.path)
        self.assertEqual(settings["model"], {
            "name": "deepseek-flash", "endpoint": "https://api.deepseek.com/chat/completions",
            "request_timeout": 120,
        })
        self.assertEqual(settings["display"], {"character_delay": 0.02})
        self.assertEqual(settings["engine"], {
            "max_requests": 20, "max_retries": 3, "retry_initial_delay": 1.0, "retry_backoff": 2.0,
        })
        self.assertEqual(settings["background"], {"max_concurrent": 5, "default_timeout": 300})
        self.assertEqual(settings["security"], {"mode": "ask", "auto_directories": []})
        self.assertEqual(settings["swarm"], {"max_requests": 120, "max_role_requests": 40})
        self.assertEqual(settings["context"], {
            "max_chars": 64000, "summary_chars": 2000, "keep_recent_turns": 4,
            "max_compactions": 6, "tool_result_chars": 12000,
        })
        self.assertEqual(settings["tools"], {
            "file_max_bytes": 1048576,
            "read_file": {"page_lines": 200},
            "bash": {"default_timeout": 30, "max_timeout": 120, "max_output_bytes": 65536},
            "grep": {"default_max_results": 100, "max_results": 500,
                     "max_line_chars": 500, "max_result_chars": 5500},
        })
        self.assertEqual(settings["permissions"], {
            "allow": ["read_file", "grep", "delegate", "notes_append"],
            "ask": ["write_file", "notes_replace"], "deny": [], "rules": [],
        })

    def test_initial_rates_and_peak_periods_are_preserved(self):
        self.assertEqual(self.defaults["pricing"], {
            "input_hit_per_million": 0.003, "input_miss_per_million": 0.15,
            "output_per_million": 0.6,
            "peak": {"input_hit_per_million": 0.006, "input_miss_per_million": 0.30,
                     "output_per_million": 1.2},
            "currency": "USD", "source": "https://api-docs.deepseek.com/quick_start/pricing",
            "checked_at": "2026-09-18", "peak_weekdays": [0, 1, 2, 3, 4],
            "peak_hours_utc": [[1, 4], [6, 10]],
        })

    def test_changed_toml_values_are_loaded_without_hardcoded_fallbacks(self):
        path = self.modified(
            ('name = "deepseek-flash"', 'name = "custom-model"'),
            ("character_delay = 0.02", "character_delay = 0.04"),
            ("max_requests = 20", "max_requests = 7"),
            ("max_concurrent = 5", "max_concurrent = 2"),
            ("[tool.harness.tools.bash]\ndefault_timeout = 30",
             "[tool.harness.tools.bash]\ndefault_timeout = 10"),
            ("default_max_results = 100", "default_max_results = 5"),
            ("deny = []", 'deny = ["bash"]'),
        )
        settings = config.load_settings(path)
        self.assertEqual(settings["model"]["name"], "custom-model")
        self.assertEqual(settings["display"]["character_delay"], 0.04)
        self.assertEqual(settings["engine"]["max_requests"], 7)
        self.assertEqual(settings["background"]["max_concurrent"], 2)
        self.assertEqual(settings["tools"]["bash"]["default_timeout"], 10)
        self.assertEqual(settings["tools"]["grep"]["default_max_results"], 5)
        self.assertEqual(settings["permissions"]["deny"], ["bash"])

    def test_missing_file_has_actionable_sanitized_error(self):
        with self.assertRaisesRegex(ValueError, "未找到项目 pyproject.toml"):
            config.load_settings(Path(self.directory.name) / "missing.toml")

    def test_file_read_errors_do_not_disclose_details(self):
        for error in (PermissionError("private-test-value"), UnicodeError("private-test-value")):
            with self.subTest(error=type(error).__name__):
                with patch.object(Path, "open", side_effect=error):
                    with self.assertRaises(ValueError) as raised:
                        config.load_settings(self.path)
                self.assertIn("pyproject.toml", str(raised.exception))
                self.assertNotIn("private-test-value", str(raised.exception))

    def test_invalid_encoding_and_toml_do_not_echo_source(self):
        for source in (b"\xffprivate-test-value", b'[tool.harness]\nprivate-test-value = "unterminated'):
            with self.subTest(source=source):
                self.path.write_bytes(source)
                with self.assertRaises(ValueError) as raised:
                    config.load_settings(self.path)
                self.assertIn("TOML", str(raised.exception))
                self.assertNotIn("private-test-value", str(raised.exception))

    def test_missing_harness_table_is_not_replaced_by_defaults(self):
        for source in ('[project]\nname = "harness"\n', '[tool]\nharness = "invalid"\n'):
            with self.subTest(source=source):
                self.path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "tool.harness"):
                    config.load_settings(self.path)

    def test_every_missing_section_is_rejected(self):
        for section in self.defaults:
            with self.subTest(section=section):
                settings = deepcopy(self.defaults)
                del settings[section]
                with self.assertRaises(ValueError):
                    self.load_document(settings)

    def test_missing_nested_field_is_rejected(self):
        for section, field in (("bash", "max_timeout"), ("read_file", "page_lines")):
            with self.subTest(section=section):
                settings = deepcopy(self.defaults)
                del settings["tools"][section][field]
                with self.assertRaisesRegex(ValueError, f"tool.harness.tools.{section}"):
                    self.load_document(settings)

    def test_unknown_fields_and_secret_fields_are_rejected_without_echoing_keys(self):
        for section in ((), ("model",), ("tools", "bash"), ("tools", "read_file"), ("permissions",)):
            with self.subTest(section=section):
                settings = deepcopy(self.defaults)
                target = settings
                for key in section:
                    target = target[key]
                target["private-test-value"] = "not-allowed"
                with self.assertRaises(ValueError) as raised:
                    self.load_document(settings)
                self.assertNotIn("private-test-value", str(raised.exception))
        with self.assertRaises(ValueError):
            config.load_settings(self.modified(("request_timeout = 120", 'request_timeout = 120\napi_key = "private-test-value"')))

    def test_unrelated_project_and_tool_tables_do_not_affect_harness(self):
        with self.path.open("a", encoding="utf-8") as file:
            file.write('\n[tool.other]\nsetting = "other"\n')
        self.assertEqual(config.load_settings(self.path), self.defaults)

    def test_numeric_fields_reject_booleans_non_numbers_nan_and_infinity(self):
        paths = (
            ("model", "request_timeout"), ("display", "character_delay"),
            ("engine", "max_requests"), ("engine", "max_retries"),
            ("engine", "retry_initial_delay"), ("engine", "retry_backoff"),
            ("background", "max_concurrent"), ("background", "default_timeout"),
            ("swarm", "max_requests"), ("swarm", "max_role_requests"),
            ("context", "max_chars"), ("context", "summary_chars"),
            ("context", "keep_recent_turns"), ("context", "max_compactions"),
            ("context", "tool_result_chars"), ("tools", "file_max_bytes"),
            ("tools", "read_file", "page_lines"),
            ("tools", "bash", "default_timeout"), ("tools", "bash", "max_timeout"),
            ("tools", "bash", "max_output_bytes"), ("tools", "grep", "default_max_results"),
            ("tools", "grep", "max_results"), ("tools", "grep", "max_line_chars"),
            ("tools", "grep", "max_result_chars"), ("pricing", "input_hit_per_million"),
            ("pricing", "input_miss_per_million"), ("pricing", "output_per_million"),
            ("pricing", "peak", "input_hit_per_million"),
        )
        for path in paths:
            for value in (True, False, "1", None, float("nan"), float("inf"), -float("inf"), -1):
                with self.subTest(path=path, value=value):
                    settings = deepcopy(self.defaults)
                    target = settings
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    with self.assertRaises(ValueError):
                        self.load_document(settings)

    def test_toml_non_finite_values_are_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "character_delay"):
                    config.load_settings(self.modified(("character_delay = 0.02", f"character_delay = {value}")))

    def test_integer_counts_reject_floats(self):
        for old, new in (("max_requests = 20", "max_requests = 20.0"),
                         ("max_retries = 3", "max_retries = 3.0"),
                         ("page_lines = 200", "page_lines = 200.0"),
                         ("max_results = 500", "max_results = 500.0")):
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    config.load_settings(self.modified((old, new)))

    def test_supported_zero_values_can_disable_optional_behavior(self):
        settings = config.load_settings(self.modified(
            ("character_delay = 0.02", "character_delay = 0"),
            ("max_retries = 3", "max_retries = 0"),
            ("retry_initial_delay = 1.0", "retry_initial_delay = 0"),
            ("keep_recent_turns = 4", "keep_recent_turns = 0"),
            ("max_compactions = 6", "max_compactions = 0"),
            ("input_hit_per_million = 0.003", "input_hit_per_million = 0"),
        ))
        self.assertEqual(settings["context"]["keep_recent_turns"], 0)
        self.assertEqual(settings["engine"]["max_retries"], 0)

    def test_timeouts_and_result_counts_obey_limits(self):
        for old, new in (("[tool.harness.tools.bash]\ndefault_timeout = 30",
                          "[tool.harness.tools.bash]\ndefault_timeout = 121"),
                         ("[tool.harness.background]\nmax_concurrent = 5\ndefault_timeout = 300",
                          "[tool.harness.background]\nmax_concurrent = 5\ndefault_timeout = 301"),
                         ("max_role_requests = 40", "max_role_requests = 121"),
                         ('mode = "ask"', 'mode = "invalid"'),
                         ("auto_directories = []", 'auto_directories = ["/etc"]'),
                         ("default_max_results = 100", "default_max_results = 501"),
                         ("page_lines = 200", "page_lines = 0"),
                         ("request_timeout = 120", "request_timeout = 0"),
                         ("max_requests = 20", "max_requests = 0"),
                         ("retry_backoff = 2.0", "retry_backoff = 0.5")):
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    config.load_settings(self.modified((old, new)))

    def test_summary_must_leave_room_in_context(self):
        for value in (64000, 64001):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "摘要长度"):
                    config.load_settings(self.modified(("summary_chars = 2000", f"summary_chars = {value}")))

    def test_result_budgets_are_compatible(self):
        for old, new in (("max_result_chars = 5500", "max_result_chars = 999"),
                         ("max_result_chars = 5500", "max_result_chars = 1000"),
                         ("max_result_chars = 5500", "max_result_chars = 11501"),
                         ("max_output_bytes = 65536", "max_output_bytes = 127"),
                         ("max_line_chars = 500", "max_line_chars = 0")):
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    config.load_settings(self.modified((old, new)))
        settings = config.load_settings(self.modified(
            ("max_result_chars = 5500", "max_result_chars = 1000"),
            ("tool_result_chars = 12000", "tool_result_chars = 1500"),
            ("max_output_bytes = 65536", "max_output_bytes = 128"),
            ("max_line_chars = 500", "max_line_chars = 100"),
        ))
        self.assertEqual(settings["tools"]["grep"]["max_line_chars"], 100)

    def test_grep_budget_reserves_space_for_json_escaping_and_metadata(self):
        with self.assertRaisesRegex(ValueError, "6 倍加 400"):
            config.load_settings(self.modified(
                ("max_result_chars = 5500", "max_result_chars = 1000"),
                ("max_line_chars = 500", "max_line_chars = 101"),
            ))

    def test_endpoint_requires_https_hostname_and_no_userinfo(self):
        for endpoint in ("http://example.com", "https:///missing", "https://user:private-test-value@example.com",
                         "https://user@example.com", "https://example.com:99999", "https://bad host.com",
                         "https://example.com\x00/private-test-value", "https://[broken", ""):
            with self.subTest(endpoint=endpoint):
                settings = deepcopy(self.defaults)
                settings["model"]["endpoint"] = endpoint
                with self.assertRaises(ValueError) as raised:
                    self.load_document(settings)
                self.assertNotIn("private-test-value", str(raised.exception))
        settings = deepcopy(self.defaults)
        settings["model"]["endpoint"] = "https://localhost:8443/chat/completions"
        self.assertEqual(self.load_document(settings)["model"]["endpoint"], settings["model"]["endpoint"])

    def test_peak_weekdays_and_hours_are_validated(self):
        for field, values in (
            ("peak_weekdays", ([True], [-1], [7], [1.0], "Monday")),
            ("peak_hours_utc", ([[4, 4]], [[4, 3]], [[-1, 3]], [[1, 25]], [[False, 2]], [[1, 2.0]], [[1]], "1-4")),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    settings = deepcopy(self.defaults)
                    settings["pricing"][field] = value
                    with self.assertRaises(ValueError):
                        self.load_document(settings)
        settings = deepcopy(self.defaults)
        settings["pricing"]["peak_weekdays"] = [0, 6]
        settings["pricing"]["peak_hours_utc"] = [[0, 24]]
        self.assertEqual(self.load_document(settings)["pricing"]["peak_hours_utc"], [[0, 24]])
        settings["pricing"]["peak_weekdays"] = []
        settings["pricing"]["peak_hours_utc"] = []
        self.load_document(settings)

    def test_permission_lists_require_nonempty_names_and_allow_overlap(self):
        for rule in ("allow", "ask", "deny"):
            for value in ("read_file", [True], [1], [""], [" "], ["read file"], [" read_file"]):
                with self.subTest(rule=rule, value=value):
                    settings = deepcopy(self.defaults)
                    settings["permissions"][rule] = value
                    with self.assertRaises(ValueError):
                        self.load_document(settings)
        settings = deepcopy(self.defaults)
        settings["permissions"] = {rule: ["bash", "new_tool"] for rule in ("allow", "ask", "deny")}
        settings["permissions"]["rules"] = []
        self.assertEqual(self.load_document(settings)["permissions"], settings["permissions"])

    def test_permission_rules_load_from_toml_array_tables_as_original_dicts(self):
        source = self.source.replace("rules = []", "") + r'''
[[tool.harness.permissions.rules]]
tool = "write_file"
action = "allow"
directory = "tests"
priority = 100

[[tool.harness.permissions.rules]]
tool = "bash"
action = "deny"
command_pattern = '\brm\s+-(?:rf|fr)\b'
priority = 200
'''
        self.path.write_text(source, encoding="utf-8")
        rules = config.load_settings(self.path)["permissions"]["rules"]
        self.assertEqual(rules, [
            {"tool": "write_file", "action": "allow", "directory": "tests", "priority": 100},
            {"tool": "bash", "action": "deny", "command_pattern": r"\brm\s+-(?:rf|fr)\b", "priority": 200},
        ])

    def test_permission_rules_require_a_list_and_valid_records(self):
        invalid = (
            None, (), {}, "deny", [None], ["bash"], [True], [{}],
            [{"tool": "bash"}], [{"action": "deny"}],
            [{"tool": "bash", "action": "deny", "private-test-value": True}],
            [{"tool": "", "action": "deny"}], [{"tool": "read file", "action": "ask"}],
            [{"tool": "bash", "action": "unknown"}],
            [{"tool": "bash", "action": "deny", "priority": True}],
            [{"tool": "bash", "action": "deny", "priority": 1.5}],
            [{"tool": "bash", "action": "allow"}],
            [{"tool": "write_file", "action": "allow"}],
            [{"tool": "write_file", "action": "allow", "directory": "../outside"}],
            [{"tool": "write_file", "action": "allow", "directory": "/tmp/tests"}],
            [{"tool": "write_file", "action": "allow", "directory": ""}],
            [{"tool": "write_file", "action": "allow", "directory": "tests\x00"}],
            [{"tool": "bash", "action": "deny", "directory": "tests"}],
            [{"tool": "grep", "action": "deny", "directory": "tests"}],
            [{"tool": "new_tool", "action": "ask", "directory": "tests"}],
            [{"tool": "write_file", "action": "ask", "command_pattern": ".*"}],
            [{"tool": "bash", "action": "deny", "command_pattern": ""}],
            [{"tool": "bash", "action": "deny", "command_pattern": "private-test-value["}],
            [{"tool": "write_file", "action": "ask", "directory": "tests", "command_pattern": ".*"}],
        )
        for rules in invalid:
            with self.subTest(rules=rules):
                settings = deepcopy(self.defaults)
                settings["permissions"]["rules"] = rules
                with self.assertRaisesRegex(ValueError, r"tool\.harness\.permissions\.rules") as raised:
                    self.load_document(settings)
                self.assertNotIn("private-test-value", str(raised.exception))

    def test_permission_rules_field_is_required(self):
        with self.assertRaisesRegex(ValueError, r"tool\.harness\.permissions"):
            config.load_settings(self.modified(("rules = []", "")))

    def test_invalid_rule_regex_stops_startup_before_api_key_loading(self):
        from harness import __main__

        source = self.source.replace("rules = []", "") + '''
[[tool.harness.permissions.rules]]
tool = "bash"
action = "deny"
command_pattern = 'private-test-value['
'''
        self.path.write_text(source, encoding="utf-8")
        config._cached_settings.cache_clear()
        self.addCleanup(config._cached_settings.cache_clear)
        error = StringIO()
        with patch.object(config, "PROJECT_FILE", self.path), patch.object(__main__, "load_api_key") as load_key:
            with patch("sys.stderr", error):
                self.assertEqual(__main__.main(), 1)
        load_key.assert_not_called()
        self.assertIn("tool.harness.permissions.rules", error.getvalue())
        self.assertNotIn("private-test-value", error.getvalue())

    def test_permission_rule_data_is_cached_and_independent_of_returned_copies(self):
        self.modified(("rules = []", 'rules = [{ tool = "write_file", action = "allow", directory = "tests" }]'))
        config._cached_settings.cache_clear()
        self.addCleanup(config._cached_settings.cache_clear)
        with patch.object(config, "PROJECT_FILE", self.path):
            first = config.get_settings()
            first["permissions"]["rules"][0]["directory"] = "."
            self.modified(("rules = []", 'rules = [{ tool = "bash", action = "deny" }]'))
            second = config.get_settings()
            self.assertEqual(second["permissions"]["rules"], [
                {"tool": "write_file", "action": "allow", "directory": "tests"},
            ])
            self.assertEqual(config.load_settings()["permissions"]["rules"], [
                {"tool": "bash", "action": "deny"},
            ])

    def test_text_fields_cannot_be_empty_or_non_string(self):
        for section, key in (("model", "name"), ("pricing", "currency"),
                             ("pricing", "source"), ("pricing", "checked_at")):
            for value in ("", " ", 123):
                with self.subTest(section=section, key=key, value=value):
                    settings = deepcopy(self.defaults)
                    settings[section][key] = value
                    with self.assertRaises(ValueError):
                        self.load_document(settings)

    def test_default_path_does_not_follow_working_directory(self):
        self.modified(("max_requests = 20", "max_requests = 1"))
        with patch("os.getcwd", return_value=self.directory.name):
            self.assertEqual(config.load_settings()["engine"]["max_requests"], 20)
        self.assertEqual(config.PROJECT_FILE, self.project_file)

    def test_each_load_returns_independent_data(self):
        first = config.load_settings(self.path)
        first["permissions"]["deny"].append("read_file")
        first["pricing"]["peak_hours_utc"][0][0] = 0
        self.assertEqual(config.load_settings(self.path), self.defaults)

    def test_get_settings_caches_file_and_returns_isolated_deep_copies(self):
        config._cached_settings.cache_clear()
        self.addCleanup(config._cached_settings.cache_clear)
        with patch.object(config, "PROJECT_FILE", self.path):
            first = config.get_settings()
            first["permissions"]["allow"].clear()
            first["pricing"]["peak_hours_utc"][0][0] = 0
            self.modified(("max_requests = 20", "max_requests = 1"))
            with patch.object(Path, "open", side_effect=AssertionError("cached configuration reread")):
                second = config.get_settings()
            self.assertEqual(second, self.defaults)
            self.assertEqual(config.load_settings()["engine"]["max_requests"], 1)

    def test_settings_loading_never_reads_dotenv_or_environment(self):
        with patch.object(config, "load_api_key", side_effect=AssertionError("unexpected key read")):
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-value"}):
                settings = config.load_settings(self.path)
        self.assertEqual(settings, self.defaults)


if __name__ == "__main__":
    unittest.main()
