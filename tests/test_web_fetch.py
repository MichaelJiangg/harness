import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.permissions import PermissionPolicy
from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.tools.executor import ToolError
from harness.tools.web_fetch import DEFINITION, execute


class FakeResponse(io.BytesIO):
    def __init__(self, data, *, status=200, content_type="text/html", url="https://example.com"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(data))}
        self.final_url = url

    def geturl(self):
        return self.final_url


class WebFetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fetch(self, url, response, *, opener=None):
        patcher = patch("harness.tools.web_fetch.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ])
        patcher.start()
        self.addCleanup(patcher.stop)
        return execute({"url": url}, self.root, opener=opener or Mock(return_value=response))

    def test_definition_is_registered_and_reads_public_html(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("web_fetch", names)
        self.assertIs(REGISTRY.get("web_fetch")[0], DEFINITION)
        response = FakeResponse(
            "<html><head><title>产品页</title><script>secret()</script></head>"
            "<body><h1>Personal Agent</h1><p>竞品信息。</p></body></html>".encode(),
        )
        result = self.fetch("https://example.com", response)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["title"], "产品页")
        self.assertIn("Personal Agent", result["text"])
        self.assertNotIn("secret()", result["text"])

    def test_private_loopback_non_standard_ports_and_bad_schemes_are_rejected(self):
        for url in (
            "http://127.0.0.1/", "http://localhost/", "http://192.168.1.1/",
            "http://example.local/", "ftp://example.com/file", "https://user:pass@example.com/",
            "https://example.com:8443/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ToolError):
                    execute({"url": url}, self.root)

    def test_http_status_content_type_and_size_limits_are_errors(self):
        with self.assertRaises(ToolError) as error:
            self.fetch("https://example.com", FakeResponse(b"", status=404))
        self.assertEqual(error.exception.code, "http_error")
        with self.assertRaises(ToolError) as error:
            self.fetch("https://example.com", FakeResponse(b"x", content_type="image/png"))
        self.assertEqual(error.exception.code, "unsupported_content")
        with self.assertRaises(ToolError) as error:
            self.fetch("https://example.com", FakeResponse(b"x" * (1_048_576 + 1)))
        self.assertIn(error.exception.code, {"too_large"})

    def test_default_permission_asks_and_confirm_approves(self):
        response = FakeResponse("<html><body>公开内容</body></html>".encode())
        opener = Mock(return_value=response)
        patcher = patch("harness.tools.web_fetch.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ])
        patcher.start()
        self.addCleanup(patcher.stop)
        executor = create_tool_executor(self.root)
        result = executor("web_fetch", {"url": "https://example.com"})
        self.assertEqual(result["code"], "confirmation_required")
        executor = create_tool_executor(self.root, confirm=Mock(return_value=True))
        with patch("harness.tools.web_fetch.build_opener", return_value=opener):
            result = executor("web_fetch", {"url": "https://example.com"})
        self.assertEqual(result["status"], "success")
        self.assertIn("公开内容", result["text"])


if __name__ == "__main__":
    unittest.main()
