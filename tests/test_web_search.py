import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from harness.context import truncate_tool_result
from harness.orchestration import query_context
from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.tools.executor import ToolError
from harness.tools.web_search import DEFINITION, execute
from test_cli import CLISession, reply
from test_engine import tool_call


class FakeResponse(io.BytesIO):
    def __init__(self, data, *, status=200):
        super().__init__(data)
        self.status = status

    def getcode(self):
        return self.status


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_definition_and_request_are_registered_without_leaking_key(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("web_search", names)
        self.assertIs(REGISTRY.get("web_search")[0], DEFINITION)
        response = FakeResponse(json.dumps({
            "answer": "",
            "results": [{
                "title": "MUSE 产品页",
                "url": "https://example.com/muse",
                "content": "Personal Agent 竞品信息。",
                "score": 0.9,
                "published_date": "2026-09-19",
            }],
        }, ensure_ascii=False).encode())
        opener = Mock(return_value=response)
        result = execute(
            {"query": "Personal Agent MUSE Today", "max_results": 5},
            self.root,
            opener=opener,
            api_key="test-tavily-key",
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["results"][0]["title"], "MUSE 产品页")
        request = opener.call_args.args[0]
        self.assertIn("Bearer test-tavily-key", request.headers["Authorization"])
        self.assertNotIn("test-tavily-key", result["message"])

    def test_missing_key_returns_configuration_error(self):
        with patch("harness.tools.web_search.load_tavily_api_key", return_value=None):
            with self.assertRaises(ToolError) as error:
                execute({"query": "MUSE"}, self.root)
        self.assertEqual(error.exception.code, "web_search_unconfigured")
        self.assertIn("TAVILY_API_KEY", str(error.exception))

    def test_http_and_malformed_responses_are_safe_errors(self):
        response = FakeResponse(b"", status=401)
        with self.assertRaises(ToolError) as error:
            execute({"query": "MUSE"}, self.root, opener=Mock(return_value=response),
                    api_key="test-key")
        self.assertEqual(error.exception.code, "web_search_error")
        self.assertNotIn("test-key", str(error.exception))

        response = FakeResponse(b"{broken")
        with self.assertRaises(ToolError) as error:
            execute({"query": "MUSE"}, self.root, opener=Mock(return_value=response),
                    api_key="test-key")
        self.assertEqual(error.exception.code, "web_search_error")

    def test_default_permission_allows_repeated_searches_without_confirmation(self):
        confirm = Mock(return_value=False)
        executor = create_tool_executor(self.root, confirm=confirm)
        opener = Mock(side_effect=lambda *args, **kwargs: FakeResponse(b'{"results": []}'))
        with patch("harness.tools.web_search.load_tavily_api_key", return_value="test-key"), \
                patch("harness.tools.web_search.urlopen", opener):
            for query in ("奇瑞 开阳实验室", "奇瑞 星火计划", "开阳实验室 最新进展"):
                with self.subTest(query=query):
                    result = executor("web_search", {"query": query})
                    self.assertEqual(result["status"], "success")
                    self.assertTrue(result["executed"])
        self.assertEqual(opener.call_count, 3)
        confirm.assert_not_called()

    def test_cli_research_completes_in_both_modes_without_confirmation_input(self):
        for mode in ("ask", "auto"):
            with self.subTest(mode=mode):
                calls = [tool_call(f"search-{index}", "web_search", json.dumps({"query": query}))
                         for index, query in enumerate(("奇瑞 开阳实验室", "奇瑞 星火计划"))]
                client = Mock(complete=Mock(side_effect=[
                    reply(None, tool_calls=calls), reply("调研完成。"),
                ]))
                opener = Mock(side_effect=lambda *args, **kwargs: FakeResponse(b'{"results": []}'))
                with patch("harness.tools.executor.Path.cwd", return_value=self.root), \
                        patch("harness.tools.web_search.load_tavily_api_key", return_value="test-key"), \
                        patch("harness.tools.web_search.urlopen", opener):
                    session = CLISession(client, lines=(f"/mode {mode}\n", "调研开阳实验室\n"),
                                         terminal=True, character_delay=0)
                    self.addCleanup(session.close)
                    self.assertTrue(session.output.wait_for("调研完成。"))
                    session.close()
                self.assertNotIn("[确认]", session.output.getvalue())
                self.assertEqual(opener.call_count, 2)
                results = [json.loads(message["content"])
                           for message in client.complete.call_args_list[1].kwargs["messages"]
                           if message["role"] == "tool"]
                self.assertEqual(len(results), 2)
                self.assertTrue(all(result["status"] == "success" for result in results))
                self.assertEqual(session.errors.getvalue(), "")

    def search_with_budget(self, results, budget, *, answer=""):
        response = FakeResponse(json.dumps({"results": results, "answer": answer},
                                           ensure_ascii=False).encode())
        with query_context(SimpleNamespace(tool_result_limit=budget)):
            return execute({"query": "奇瑞 开阳实验室", "max_results": 10}, self.root,
                           opener=Mock(return_value=response), api_key="test-key")

    def test_large_search_keeps_complete_sources_within_serialized_budget(self):
        results = [{"title": "完整标题" + "标" * 350,
                    "url": f"https://example.com/source/{index}?query=%E4%B8%AD",
                    "content": '中文😀"\\\n\t' * 1000, "score": 0.9,
                    "published_date": "2026-10-14"} for index in range(10)]
        result = self.search_with_budget(results, 12000)
        complete = {**result, "executed": True, "tool": "web_search"}
        self.assertLessEqual(len(json.dumps(complete, ensure_ascii=False)), 12000)
        self.assertEqual(result["returned_count"], 10)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["omitted_count"], 0)
        for actual, original in zip(result["results"], results):
            for key in ("title", "url", "score", "published_date"):
                self.assertEqual(actual[key], original[key])
            self.assertTrue(actual["content_truncated"])
            self.assertTrue(original["content"].startswith(actual["content"]))
            self.assertTrue(actual["content"])

    def test_small_query_budget_omits_whole_items_instead_of_cutting_urls(self):
        results = [{"title": f"来源 {index}", "url": f"https://example.com/{index}/" + "x" * 200,
                    "content": "正文" * 1000, "score": 0.8} for index in range(10)]
        result = self.search_with_budget(results, 1200)
        complete = {**result, "executed": True, "tool": "web_search"}
        self.assertLessEqual(len(json.dumps(complete, ensure_ascii=False)), 1200)
        self.assertGreater(result["returned_count"], 0)
        self.assertLess(result["returned_count"], 10)
        self.assertEqual(result["omitted_count"], 10 - result["returned_count"])
        self.assertTrue(result["truncated"])
        self.assertEqual([item["url"] for item in result["results"]],
                         [item["url"] for item in results[:result["returned_count"]]])

    def test_overlong_url_is_omitted_as_a_whole_source(self):
        results = [{"title": "过长链接", "url": "https://example.com/" + "x" * 2048,
                    "content": "不应返回截断链接"},
                   {"title": "有效来源", "url": "https://example.com/valid", "content": "正文"}]
        result = self.search_with_budget(results, 12000)
        self.assertEqual([item["url"] for item in result["results"]], [results[1]["url"]])
        self.assertEqual(result["omitted_count"], 1)
        self.assertTrue(result["truncated"])

    def test_source_with_oversized_metadata_does_not_crowd_out_later_sources(self):
        results = [{"title": "标题" * 1000, "url": "https://example.com/large", "content": "正文"},
                   {"title": "可容纳来源", "url": "https://example.com/small", "content": "正文"}]
        result = self.search_with_budget(results, 700)
        self.assertEqual([item["url"] for item in result["results"]], [results[1]["url"]])
        self.assertEqual(result["omitted_count"], 1)
        self.assertTrue(result["truncated"])

    def test_budget_too_small_for_search_metadata_returns_explicit_error(self):
        opener = Mock()
        with query_context(SimpleNamespace(tool_result_limit=40)), self.assertRaises(ToolError) as error:
            execute({"query": "奇瑞 开阳实验室"}, self.root, opener=opener, api_key="test-key")
        self.assertEqual(error.exception.code, "search_budget_too_small")
        opener.assert_not_called()

    def test_default_budget_counts_escaped_answer_and_executor_metadata(self):
        source = {"title": "完整来源", "url": "https://example.com/source",
                  "content": '中文😀"\\\n\t' * 1000}
        response = FakeResponse(json.dumps({"results": [source], "answer": "总结" * 1000},
                                           ensure_ascii=False).encode())
        with patch("harness.tools.web_search.get_settings",
                   return_value={"context": {"tool_result_chars": 800}}):
            result = execute({"query": "奇瑞 开阳实验室"}, self.root,
                             opener=Mock(return_value=response), api_key="test-key")
        complete = {**result, "executed": True, "tool": "web_search"}
        encoded = json.dumps(complete, ensure_ascii=False)
        self.assertLessEqual(len(encoded), 800)
        self.assertTrue(result["answer_truncated"])
        self.assertTrue(result["results"][0]["content_truncated"])
        self.assertEqual(result["results"][0]["url"], source["url"])
        self.assertEqual(truncate_tool_result(complete, 800), encoded)

    def test_exact_budget_preserves_full_search_without_extra_truncation(self):
        source = {"title": "完整来源", "url": "https://example.com/source",
                  "content": "完整摘要😀\n"}
        expected = self.search_with_budget([source], 12000)
        exact = len(json.dumps({**expected, "executed": True, "tool": "web_search"},
                               ensure_ascii=False))
        self.assertEqual(self.search_with_budget([source], exact), expected)
        smaller = self.search_with_budget([source], exact - 1)
        self.assertLessEqual(len(json.dumps({**smaller, "executed": True, "tool": "web_search"},
                                           ensure_ascii=False)), exact - 1)
        self.assertEqual(smaller["results"][0]["url"], source["url"])
        self.assertTrue(smaller["results"][0]["content_truncated"])

    def test_executor_result_keeps_sources_without_generic_head_tail_wrapper(self):
        source = {"title": "来源", "url": "https://example.com/" + "x" * (2048 - 20),
                  "content": "正文" * 10000}
        self.assertEqual(len(source["url"]), 2048)
        opener = Mock(return_value=FakeResponse(json.dumps({"results": [source]},
                                                         ensure_ascii=False).encode()))
        executor = create_tool_executor(self.root)
        with query_context(SimpleNamespace(tool_result_limit=3000)), \
                patch("harness.tools.web_search.load_tavily_api_key", return_value="test-key"), \
                patch("harness.tools.web_search.urlopen", opener):
            result = executor("web_search", {"query": "奇瑞 开阳实验室"})
        encoded = truncate_tool_result(result, 3000)
        parsed = json.loads(encoded)
        self.assertLessEqual(len(encoded), 3000)
        self.assertEqual(parsed["status"], "success")
        self.assertTrue(parsed["executed"])
        self.assertEqual(parsed["tool"], "web_search")
        self.assertEqual(parsed["results"][0]["url"], source["url"])
        self.assertNotIn("head", parsed)


if __name__ == "__main__":
    unittest.main()
