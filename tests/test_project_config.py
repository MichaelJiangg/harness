from pathlib import Path
import io
import json
import os
from contextlib import redirect_stdout
import tempfile
import unittest

from harness import config


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config_file = self.root / ".harness" / "config.toml"

    def write_config(self, text):
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(text, encoding="utf-8")
        return self.config_file

    def test_missing_file_uses_builtin_defaults(self):
        settings = config.resolve_settings(config_file=self.root / "missing.toml", environ={})
        self.assertEqual(settings["provider"], "auto")
        self.assertEqual(settings["model"]["name"], "deepseek-flash")
        self.assertEqual(settings["engine"]["max_turns"], 100)
        self.assertEqual(settings["context"]["max_chars"], 64000)
        self.assertEqual(settings["tools"]["bash"]["default_timeout"], 30)
        self.assertEqual(settings["mcp"]["servers"][0]["name"], "tavily")

    def test_installed_runtime_reads_local_dotenv_first(self):
        (self.root / ".env").write_text(
            'DEEPSEEK_API_KEY="cwd-test-key"\n',
            encoding="utf-8",
        )
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.assertEqual(
            config.load_env_key("DEEPSEEK_API_KEY", environ={}),
            "cwd-test-key",
        )

    def test_partial_config_is_merged_over_defaults(self):
        self.write_config('''
[model]
name = "custom-model"

[engine]
max_turns = 12
context_window = 8000

[tools]
timeout = 7
''')
        settings = config.resolve_settings(config_file=self.config_file, environ={})
        self.assertEqual(settings["model"]["name"], "custom-model")
        self.assertEqual(settings["engine"]["max_turns"], 12)
        self.assertEqual(settings["context"]["max_chars"], 8000)
        self.assertEqual(settings["tools"]["bash"]["default_timeout"], 7)

    def test_precedence_is_cli_env_file_defaults(self):
        self.write_config('''
[model]
name = "file-model"

[engine]
max_turns = 20

[tools]
timeout = 20
''')
        settings = config.resolve_settings(
            config_file=self.config_file,
            environ={"HARNESS_MODEL": "env-model", "HARNESS_MAX_TURNS": "30"},
            cli_overrides={"model": {"name": "cli-model"}, "tools": {"bash": {"default_timeout": 5}}},
        )
        self.assertEqual(settings["model"]["name"], "cli-model")
        self.assertEqual(settings["engine"]["max_turns"], 30)
        self.assertEqual(settings["tools"]["bash"]["default_timeout"], 5)

    def test_mcp_servers_can_come_from_env_or_cli(self):
        settings = config.resolve_settings(
            environ={
                "HARNESS_MCP_SERVERS": '[{"name":"env-server","command":"env-command"}]',
            },
        )
        self.assertEqual(settings["mcp"]["servers"][0]["name"], "env-server")
        settings = config.resolve_settings(
            environ={},
            cli_overrides={
                "mcp": {"servers": [{"name": "cli-server", "command": "cli-command", "args": [], "env": []}]},
            },
        )
        self.assertEqual(settings["mcp"]["servers"][0]["name"], "cli-server")

    def test_invalid_config_does_not_disclose_source_values(self):
        self.write_config('[engine]\nmax_turns = "private-test-value"\n')
        with self.assertRaises(ValueError) as raised:
            config.resolve_settings(config_file=self.config_file, environ={})
        self.assertNotIn("private-test-value", str(raised.exception))

    def test_cli_parser_and_override_builder(self):
        args = config.parse_cli_args([
            "--provider", "glm", "--model", "glm-model", "--max-turns", "9",
            "--context-window", "5000", "--tool-timeout", "4",
            "--mcp-server", "search=search-command",
        ])
        overrides = config.cli_overrides_from_args(args)
        self.assertEqual(overrides["model"]["name"], "glm-model")
        self.assertEqual(overrides["engine"]["max_turns"], 9)
        self.assertEqual(overrides["context"]["max_chars"], 5000)
        self.assertEqual(overrides["tools"]["bash"]["default_timeout"], 4)
        self.assertEqual(overrides["mcp"]["servers"][0]["command"], "search-command")

    def test_process_settings_snapshot_is_isolated(self):
        config.clear_process_settings()
        self.addCleanup(config.clear_process_settings)
        settings = config.resolve_settings(cli_overrides={"engine": {"max_turns": 3}})
        config.set_process_settings(settings)
        snapshot = config.get_settings()
        snapshot["engine"]["max_turns"] = 999
        self.assertEqual(config.get_settings()["engine"]["max_turns"], 3)

    def test_first_run_creates_commented_valid_default_config(self):
        path = self.root / ".harness" / "config.toml"
        self.assertTrue(config.ensure_default_config(path))
        self.assertFalse(config.ensure_default_config(path))
        content = path.read_text(encoding="utf-8")
        self.assertIn("API 密钥不要写在这里", content)
        self.assertIn("HARNESS_PROVIDER", content)
        settings = config.resolve_settings(config_file=path, environ={})
        self.assertEqual(settings["engine"]["max_turns"], 100)
        self.assertEqual(settings["tools"]["bash"]["default_timeout"], 30)
        for name in ("web_search", "web_fetch", "mcp_tavily_tavily_search"):
            self.assertIn(name, settings["permissions"]["allow"])
            self.assertNotIn(name, settings["permissions"]["ask"])
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_default_tavily_search_permission_is_inherited_without_permissions_config(self):
        for text in (None, '[model]\nname = "custom-model"\n'):
            with self.subTest(config=text):
                path = self.root / "missing.toml" if text is None else self.write_config(text)
                settings = config.resolve_settings(config_file=path, environ={})
                self.assertIn("mcp_tavily_tavily_search", settings["permissions"]["allow"])
                self.assertNotIn("mcp_tavily_tavily_search", settings["permissions"]["ask"])
                if text is not None:
                    self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_explicit_tavily_permission_lists_are_not_rewritten(self):
        for field, names in (("allow", ["read_file"]), ("allow", []),
                             ("ask", ["mcp_tavily_tavily_search"]),
                             ("deny", ["mcp_tavily_tavily_search"])):
            with self.subTest(field=field, names=names):
                text = f'[permissions]\n{field} = {json.dumps(names)}\n'
                path = self.write_config(text)
                settings = config.resolve_settings(config_file=path, environ={})
                self.assertEqual(settings["permissions"][field], names)
                self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_explicit_web_permissions_are_preserved_when_loading_existing_config(self):
        path = self.write_config('[permissions]\nask = ["web_search"]\ndeny = ["web_fetch"]\n')
        settings = config.resolve_settings(config_file=path, environ={})
        self.assertEqual(settings["permissions"]["ask"], ["web_search"])
        self.assertEqual(settings["permissions"]["deny"], ["web_fetch"])
        self.assertEqual(path.read_text(encoding="utf-8"),
                         '[permissions]\nask = ["web_search"]\ndeny = ["web_fetch"]\n')

    def test_version_argument_prints_package_version(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                config.parse_cli_args(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), "Harness v1.0.4")

    def test_pyproject_keeps_only_package_metadata(self):
        path = Path(config.__file__).resolve().parent.parent / "pyproject.toml"
        document = config.tomllib.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["project"]["name"], "michael-harness")
        self.assertEqual(document["project"]["version"], "1.0.4")
        self.assertEqual(document["project"]["authors"], [{"name": "Michael Jiang"}])
        self.assertEqual(
            document["project"]["scripts"],
            {"harness": "harness.__main__:main"},
        )
        self.assertEqual(
            document["build-system"]["requires"],
            ["setuptools>=68"],
        )
        self.assertNotIn("harness", document.get("tool", {}))


if __name__ == "__main__":
    unittest.main()
