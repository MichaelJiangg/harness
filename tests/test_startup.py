import unittest
from unittest.mock import Mock, patch

from harness import startup


class StartupTests(unittest.TestCase):
    def test_required_tool_failure_raises(self):
        with patch.object(startup, "get_tool_definitions", side_effect=RuntimeError("tool failure")):
            with self.assertRaises(startup.StartupFailure):
                startup.check_startup(client=Mock())

    def test_optional_module_failures_are_reported_as_warnings(self):
        hooks = Mock(load_error="hooks failed", hooks=[])
        memory = Mock(records=Mock(side_effect=RuntimeError("memory failed")))
        with patch.object(startup, "get_tool_definitions", return_value=[{}, {}]), \
                patch.object(startup, "load_server_configs", return_value=([], ".harness/config.toml")), \
                patch.object(startup, "HookManager", return_value=hooks), \
                patch.object(startup, "MemoryStore", return_value=memory), \
                patch("sys.stdin.isatty", return_value=True), \
                patch("sys.stdout.isatty", return_value=True):
            items, warnings = startup.check_startup(client=Mock(provider="deepseek"))
        self.assertEqual(warnings, 2)
        self.assertEqual(
            [item.name for item in items if item.status == "WARN"],
            ["Hooks", "Memory"],
        )
        output = startup.format_startup_report(items, warnings, check_mode=True)
        self.assertIn("hooks failed", output)
        self.assertIn("memory failed", output)

    def test_mcp_check_reports_failed_server(self):
        statuses = [
            {"name": "good", "state": "connected", "last_error": None},
            {"name": "bad", "state": "failed", "last_error": "refused"},
        ]
        manager = Mock(status=Mock(return_value=statuses))
        with patch.object(startup, "get_tool_definitions", return_value=[{}]), \
                patch.object(startup, "load_server_configs", return_value=([{}, {}], "config")), \
                patch.object(startup, "MCPManager", return_value=manager):
            items, warnings = startup.check_startup(
                client=Mock(provider="deepseek"), check_mcp=True,
            )
        self.assertEqual(warnings, 1)
        mcp = next(item for item in items if item.name == "MCP servers")
        self.assertEqual(mcp.status, "WARN")
        self.assertIn("bad", mcp.error)
        manager.connect_all.assert_called_once()
        manager.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
