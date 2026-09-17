from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.cli import run_cli
from harness.engine import QueryState, query_loop
from harness.tools import create_tool_executor
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


def grep_call(identifier, **arguments):
    return tool_call(identifier, "grep", json.dumps(arguments, ensure_ascii=False))


class GrepIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name).resolve()

    def state(self, responses):
        return QueryState(FakeClient(responses), UsageLedger(),
                          messages=[{"role": "user", "content": "搜索代码中的 needle"}],
                          tool_executor=create_tool_executor(self.workspace))

    def test_model_receives_matching_paths_lines_and_content_without_confirmation(self):
        (self.workspace / "main.py").write_text("# header\nneedle = 1\n", encoding="utf-8")
        (self.workspace / "notes.md").write_text("needle excluded by glob", encoding="utf-8")
        (self.workspace / "pkg").mkdir()
        (self.workspace / "pkg/helper.py").write_text("# needle\n", encoding="utf-8")
        state = self.state([
            reply(None, [grep_call("search-1", keyword="needle", glob="*.py")]),
            reply("main.py 第 2 行与 pkg/helper.py 第 1 行包含 needle。"),
        ])
        self.assertIn("main.py 第 2 行", query_loop(state))
        definitions = state.client.requests[0]["tools"]
        self.assertIn("grep", [item["function"]["name"] for item in definitions])
        response_messages = state.client.requests[1]["messages"]
        self.assertEqual(response_messages[-2]["tool_calls"][0]["id"], "search-1")
        self.assertEqual(response_messages[-1]["tool_call_id"], "search-1")
        result = json.loads(response_messages[-1]["content"])
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["executed"])
        self.assertEqual([(hit["path"], hit["line_number"], hit["content"]) for hit in result["matches"]],
                         [("main.py", 2, "needle = 1"), ("pkg/helper.py", 1, "# needle")])
        self.assertEqual(state.ledger.summary()["requests"], 2)
        self.assertEqual(state.ledger.summary()["total_tokens"], 240)

    def test_invalid_root_error_allows_model_to_correct_search_scope(self):
        (self.workspace / "main.py").write_text("needle", encoding="utf-8")
        state = self.state([
            reply(None, [grep_call("wrong-path", keyword="needle", path="missing")]),
            reply(None, [grep_call("corrected", keyword="needle", path="main.py")]),
            reply("已找到 main.py 第 1 行。"),
        ])
        self.assertEqual(query_loop(state), "已找到 main.py 第 1 行。")
        failed = json.loads(state.client.requests[1]["messages"][-1]["content"])
        self.assertEqual(failed["code"], "not_found")
        self.assertFalse(failed["executed"])
        succeeded_message = state.client.requests[2]["messages"][-1]
        self.assertEqual(succeeded_message["tool_call_id"], "corrected")
        self.assertEqual(json.loads(succeeded_message["content"])["matches"][0]["line_number"], 1)
        self.assertEqual(state.ledger.summary()["requests"], 3)

    def test_many_matches_remain_structured_inside_model_context_budget(self):
        line = "needle " + "长内容" * 40
        (self.workspace / "many.py").write_text((line + "\n") * 1000, encoding="utf-8")
        state = self.state([
            reply(None, [grep_call("many-matches", keyword="needle", max_results=500)]),
            reply("匹配较多，结果已截断。"),
        ])
        self.assertEqual(query_loop(state), "匹配较多，结果已截断。")
        serialized = state.client.requests[1]["messages"][-1]["content"]
        self.assertLessEqual(len(serialized), 6000)
        result = json.loads(serialized)
        self.assertNotIn("head", result)
        self.assertTrue(result["truncated"])
        self.assertGreater(result["returned_count"], 0)
        self.assertLess(result["returned_count"], 500)
        for number, hit in enumerate(result["matches"], 1):
            self.assertEqual(hit["path"], "many.py")
            self.assertEqual(hit["line_number"], number)
            self.assertEqual(hit["content"], line)

    def test_cli_can_search_in_non_interactive_mode_without_approval(self):
        (self.workspace / "sample.py").write_text("needle\n", encoding="utf-8")
        client = Mock(complete=Mock(side_effect=[
            reply(None, [grep_call("pipe-search", keyword="needle", glob="*.py")]),
            reply("sample.py 第 1 行包含 needle。"),
        ]))
        output, errors = StringIO(), StringIO()
        with patch("harness.tools.executor.Path.cwd", return_value=self.workspace):
            run_cli(client, input_stream=StringIO("搜索 needle\n"), output=output, error_output=errors)
        self.assertIn("[工具] grep：返回 1 条匹配。", output.getvalue())
        self.assertIn("DeepSeek > sample.py 第 1 行包含 needle。", output.getvalue())
        self.assertNotIn("[确认]", output.getvalue())
        self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
