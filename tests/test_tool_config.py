import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ToolConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name) / "project"
        shutil.copytree(PROJECT_ROOT / "harness", self.project / "harness",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def write_runtime_config(self, changes):
        lines = []
        for section, values in changes.items():
            if "." in section:
                table, _, name = section.partition(".")
                lines.append(f"[{table}.{name}]")
            else:
                lines.append(f"[{section}]")
            for name, value in values.items():
                lines.append(f"{name} = {value}")
        path = self.project / ".harness" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_tools(self, changes, script):
        self.write_runtime_config(changes)
        environment = os.environ.copy()
        environment.pop("DEEPSEEK_API_KEY", None)
        environment.pop("GLM_API_KEY", None)
        prelude = """
from pathlib import Path
import json
from harness.tools import execute_tool, get_tool_definitions
workspace = Path("workspace")
workspace.mkdir()
def execute(name, arguments):
    return execute_tool(name, arguments, workspace=workspace, confirm=lambda *args: True)
definitions = {item["function"]["name"]: item["function"] for item in get_tool_definitions()}
"""
        completed = subprocess.run(
            [sys.executable, "-c", dedent(prelude) + dedent(script)], cwd=self.project,
            env=environment, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_shared_file_limit_controls_read_write_search_and_descriptions(self):
        result = self.run_tools({"tools": {"file_max_bytes": 32}}, """
            (workspace / "small.py").write_text("needle" + "x" * 26, encoding="utf-8")
            (workspace / "large.py").write_text("needle" + "x" * 27, encoding="utf-8")
            print(json.dumps({
                "read": execute("read_file", {"path": "small.py"}),
                "large_read": execute("read_file", {"path": "large.py"}),
                "write": execute("write_file", {"path": "new.txt", "content": "x" * 32}),
                "large_write": execute("write_file", {"path": "too-large.txt", "content": "x" * 33}),
                "search": execute("grep", {"keyword": "needle", "glob": "*.py"}),
                "descriptions": {name: definitions[name]["description"] for name in ("read_file", "write_file", "grep")},
                "rejected_file_exists": (workspace / "too-large.txt").exists(),
            }))
        """)
        self.assertEqual(result["read"]["status"], "success")
        self.assertEqual(len(result["read"]["content"]), 32)
        self.assertEqual(result["write"]["bytes_written"], 32)
        for name in ("large_read", "large_write"):
            self.assertEqual(result[name]["code"], "file_too_large")
            self.assertIn("32 字节", result[name]["message"])
        self.assertFalse(result["rejected_file_exists"])
        self.assertEqual(result["search"]["skipped_files"], 1)
        self.assertEqual([match["path"] for match in result["search"]["matches"]], ["small.py"])
        for description in result["descriptions"].values():
            self.assertIn("32 字节", description)
            self.assertNotIn("1 MiB", description)

    def test_read_file_default_page_lines_follow_configuration(self):
        result = self.run_tools({"tools.read_file": {"page_lines": 2}}, r'''
            (workspace / "notes.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")
            first = execute("read_file", {"path": "notes.txt"})
            second = execute("read_file", {"path": "notes.txt", "offset": first["next_offset"],
                                           "column": first["next_column"]})
            print(json.dumps({
                "definition": definitions["read_file"], "first": first, "second": second,
                "explicit": execute("read_file", {"path": "notes.txt", "limit": 3}),
            }))
        ''')
        self.assertEqual(result["definition"]["parameters"]["properties"]["limit"]["default"], 2)
        self.assertEqual(result["first"]["content"], "first\nsecond\n")
        self.assertEqual(result["first"]["next_offset"], 2)
        self.assertEqual(result["first"]["next_column"], 0)
        self.assertFalse(result["first"]["eof"])
        self.assertEqual(result["second"]["content"], "third\n")
        self.assertTrue(result["second"]["eof"])
        self.assertEqual(result["explicit"]["content"], "first\nsecond\nthird\n")
        self.assertTrue(result["explicit"]["eof"])

    def test_bash_schema_errors_and_output_limit_follow_configuration(self):
        result = self.run_tools({"tools.bash": {
            "default_timeout": 1, "max_timeout": 2, "max_output_bytes": 128,
        }}, """
            print(json.dumps({
                "definition": definitions["bash"],
                "invalid": execute("bash", {"command": "printf unused", "timeout": 3}),
                "output": execute("bash", {"command": "printf 'x%0.s' {1..400}"}),
            }))
        """)
        timeout = result["definition"]["parameters"]["properties"]["timeout"]
        self.assertEqual(timeout["default"], 1)
        self.assertEqual(timeout["maximum"], 2)
        self.assertIn("1～2", timeout["description"])
        self.assertIn("默认 1 秒，最多 2 秒", result["definition"]["description"])
        self.assertIn("128 字节", result["definition"]["description"])
        self.assertEqual(result["invalid"]["code"], "invalid_arguments")
        self.assertIn("2", result["invalid"]["message"])
        self.assertEqual(result["output"]["exit_code"], 0)
        self.assertTrue(result["output"]["stdout_truncated"])
        self.assertEqual(result["output"]["stdout_bytes"], 400)
        self.assertEqual(len(result["output"]["stdout"].encode("utf-8")), 128)

    def test_bash_executes_with_configured_default_timeout(self):
        result = self.run_tools({"tools.bash": {"default_timeout": 1, "max_timeout": 2}}, """
            from time import monotonic
            start = monotonic()
            output = execute("bash", {"command": "printf started; sleep 5"})
            print(json.dumps({"output": output, "elapsed": monotonic() - start}))
        """)
        self.assertTrue(result["output"]["timed_out"])
        self.assertEqual(result["output"]["stdout"], "started")
        self.assertNotEqual(result["output"]["exit_code"], 0)
        self.assertIn("1 秒", result["output"]["message"])
        self.assertLess(result["elapsed"], 4)

    def test_grep_schema_default_count_and_excerpts_follow_configuration(self):
        result = self.run_tools({"tools.grep": {
            "default_max_results": 2, "max_results": 3, "max_line_chars": 100,
        }}, r'''
            (workspace / "many.py").write_text("needle\n" * 4, encoding="utf-8")
            (workspace / "long.py").write_text("a" * 300 + "needle" + "z" * 300, encoding="utf-8")
            print(json.dumps({
                "definition": definitions["grep"],
                "default": execute("grep", {"keyword": "needle", "path": "many.py"}),
                "maximum": execute("grep", {"keyword": "needle", "path": "many.py", "max_results": 3}),
                "invalid": execute("grep", {"keyword": "needle", "max_results": 4}),
                "excerpt": execute("grep", {"keyword": "needle", "path": "long.py"}),
            }))
        ''')
        schema = result["definition"]["parameters"]["properties"]["max_results"]
        self.assertEqual(schema["default"], 2)
        self.assertEqual(schema["maximum"], 3)
        self.assertIn("默认 2，最大 3", schema["description"])
        self.assertIn("默认 2，最多 3", result["definition"]["description"])
        self.assertEqual(result["default"]["returned_count"], 2)
        self.assertTrue(result["default"]["truncated"])
        self.assertEqual(result["maximum"]["returned_count"], 3)
        self.assertEqual(result["invalid"]["code"], "invalid_arguments")
        match = result["excerpt"]["matches"][0]
        self.assertTrue(match["line_truncated"])
        self.assertIn("needle", match["content"])
        self.assertEqual(len(match["content"]), 102)

    def test_grep_budget_keeps_complete_entries_and_counts_json_escaping(self):
        result = self.run_tools({"tools.grep": {
            "max_line_chars": 100, "max_result_chars": 1000,
        }, "context": {"tool_result_chars": 1500}}, r'''
            (workspace / "many.py").write_text(("needle" + "\x01" * 200 + "\n") * 40, encoding="utf-8")
            print(json.dumps({
                "search": execute("grep", {"keyword": "needle"}),
                "long_keyword": execute("grep", {"keyword": "x" * 500}),
            }))
        ''')
        search = result["search"]
        self.assertEqual(search["status"], "success")
        self.assertTrue(search["truncated"])
        self.assertGreater(search["returned_count"], 0)
        self.assertLess(search["returned_count"], 40)
        self.assertLessEqual(len(json.dumps(search, ensure_ascii=False)), 1500)
        for match in search["matches"]:
            self.assertEqual(set(match), {"path", "line_number", "content", "line_truncated"})
            self.assertIn("needle", match["content"])
        self.assertEqual(result["long_keyword"]["code"], "invalid_arguments")


if __name__ == "__main__":
    unittest.main()
