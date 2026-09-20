import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.tools.executor import ToolError
from harness.tools.web_search import DEFINITION, execute


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

    def test_default_permission_asks_and_confirm_approves(self):
        executor = create_tool_executor(self.root)
        result = executor("web_search", {"query": "MUSE"})
        self.assertEqual(result["code"], "confirmation_required")


if __name__ == "__main__":
    unittest.main()
