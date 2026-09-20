from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

from harness.config import get_settings
from harness.mcp import (
    MCPConfigError, MCPError, MCPManager, MCPServer, MCPServerConfig,
    load_server_configs,
)
from harness.permissions import PermissionPolicy
from harness.tools import create_tool_executor
from harness.tools.definition import ToolDefinition
from test_cli import CLISession, reply


class PersistentLineStream:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.closed = Event()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.lines)
        except StopIteration:
            while not self.closed.wait(0.01):
                pass
            raise StopIteration


class FakePopen:
    def __init__(self, responses, *, persistent=False):
        lines = [json.dumps(item, ensure_ascii=False) for item in responses]
        self.stdin = StringIO()
        self.stdout = (
            PersistentLineStream(lines)
            if persistent
            else StringIO("\n".join(lines) + "\n")
        )
        self.stderr = StringIO()

    def terminate(self):
        if isinstance(self.stdout, PersistentLineStream):
            self.stdout.closed.set()
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None

    def kill(self):
        pass


class MCPTests(unittest.TestCase):
    def test_load_server_configs_reads_json_file_before_pyproject(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / ".harness"
            config_dir.mkdir()
            (config_dir / "mcp.json").write_text(json.dumps({
                "servers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@anthropic-ai/mcp-filesystem", "."],
                        "env": {"ROOT_DIR": "/tmp/projects"},
                    },
                    "database": {
                        "command": "python3",
                        "args": ["server.py"],
                        "env": {"DATABASE_URL": "${DATABASE_URL}"},
                    },
                },
            }), encoding="utf-8")
            configs, source = load_server_configs(root)
        self.assertEqual(source, ".harness/mcp.json")
        self.assertEqual([config.name for config in configs], ["filesystem", "database"])
        self.assertEqual(configs[0].command, "npx")
        self.assertEqual(configs[0].args, ("-y", "@anthropic-ai/mcp-filesystem", "."))
        self.assertEqual(configs[0].env, ("ROOT_DIR=/tmp/projects",))
        self.assertEqual(configs[1].env, ("DATABASE_URL=${DATABASE_URL}",))

    def test_invalid_mcp_json_config_is_rejected_without_fallback(self):
        invalid_documents = (
            None,
            [],
            {},
            {"servers": []},
            {"servers": {"bad name": {"command": "npx"}}},
            {"servers": {"filesystem": {"args": ["npx"]}}},
            {"servers": {"filesystem": {"command": "npx", "unknown": True}}},
            {"servers": {"filesystem": {"command": "npx", "args": "missing"}}},
            {"servers": {"filesystem": {"command": "npx", "env": {"BAD NAME": "x"}}}},
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    config_dir = root / ".harness"
                    config_dir.mkdir()
                    (config_dir / "mcp.json").write_text(
                        json.dumps(document) if document is not None else "{",
                        encoding="utf-8",
                    )
                    with self.assertRaises(MCPConfigError):
                        load_server_configs(root)

    def test_manager_discovers_and_calls_tools(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {
                "protocolVersion": "2024-11-05",
            }},
            {"jsonrpc": "2.0", "id": 2, "result": {
                "tools": [{
                    "name": "search",
                    "description": "搜索文件",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }],
            }},
            {"jsonrpc": "2.0", "id": 3, "result": {
                "content": [{"type": "text", "text": "found.pdf"}],
            }},
        ]
        with patch("harness.mcp.subprocess.Popen", return_value=FakePopen(responses)):
            manager = MCPManager([
                MCPServerConfig("filesystem", "fake-server", ("--demo",)),
            ])
            self.addCleanup(manager.close)
        self.assertEqual(len(manager.tools), 1)
        definition = manager.tools[0]
        self.assertEqual(definition.name, "mcp_filesystem_search")
        self.assertRegex(definition.name, r"^[A-Za-z0-9_-]+$")
        result = manager.execute(
            {"query": "*.pdf"}, None, internal_name=definition.name,
        )
        self.assertEqual(result["content"], "found.pdf")
        self.assertEqual(result["status"], "success")

    def test_manager_can_defer_initial_connections(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
        ]
        with patch("harness.mcp.subprocess.Popen",
                   return_value=FakePopen(responses)) as launch:
            manager = MCPManager(
                [MCPServerConfig("filesystem", "fake-server", ())],
                connect=False, reconnect=False,
            )
            self.addCleanup(manager.close)
            self.assertEqual(manager.servers, [])
            launch.assert_not_called()
            manager.connect_all()
        self.assertEqual(len(manager.servers), 1)
        self.assertEqual(manager.status()[0]["state"], "connected")

    def test_external_schema_skips_local_json_schema_validation(self):
        definition, _ = MCPManager._build_definition("filesystem", {
            "name": "search",
            "description": "搜索文件",
            "inputSchema": {
                "type": "object",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "properties": {
                    "query": {"type": "string", "enum": ["*.pdf", "*.txt"]},
                    "limit": {"type": "number", "minimum": 1},
                    "dry_run": {"type": "boolean"},
                },
                "required": ["query"],
            },
        }, 0, internal_name="mcp_filesystem_search")
        self.assertFalse(definition.validate_arguments)
        handler = Mock(return_value={"status": "success", "content": "found.pdf"})
        with TemporaryDirectory() as directory:
            executor = create_tool_executor(
                directory, extra_tools=[(definition, handler)],
                permissions=PermissionPolicy(allow=[definition.name]),
            )
            result = executor(
                definition.name,
                {"query": "*.pdf", "limit": 1.5, "dry_run": True},
            )
        self.assertEqual(result["status"], "success")
        handler.assert_called_once_with(
            {"query": "*.pdf", "limit": 1.5, "dry_run": True},
            Path(directory).resolve(),
        )

    def test_server_environment_resolves_env_file_reference_and_keeps_explicit_values(self):
        captured = []

        def launch(*args, **kwargs):
            captured.append(kwargs)
            return FakePopen([])

        environment = {
            "HOME": "/tmp", "PATH": "/usr/bin:/bin", "KEEP": "value",
            "DEEPSEEK_API_KEY": "deepseek-secret", "TAVILY_API_KEY": "tavily-secret",
            "BASH_ENV": "/tmp/bash-env", "ENV": "/tmp/env",
            "LD_PRELOAD": "/tmp/loader", "DYLD_INSERT_LIBRARIES": "/tmp/loader",
        }
        with patch.dict(os.environ, environment, clear=True), \
                patch("harness.mcp.load_env_key", return_value="tvly-from-env-file"), \
                patch("harness.mcp.subprocess.Popen", side_effect=launch):
            server = MCPServer(MCPServerConfig(
                "filesystem", "fake-server", (),
                env=(
                    "TAVILY_API_KEY=${TAVILY_API_KEY}",
                    "DEEPSEEK_API_KEY=forced",
                    "CUSTOM_TOKEN=token",
                ),
            ))
            server.start()
            self.addCleanup(server.close)
        child_environment = captured[0]["env"]
        self.assertTrue(captured[0]["start_new_session"])
        self.assertEqual(child_environment["HOME"], "/tmp")
        self.assertEqual(child_environment["PATH"], "/usr/bin:/bin")
        self.assertEqual(child_environment["CUSTOM_TOKEN"], "token")
        self.assertEqual(child_environment["TAVILY_API_KEY"], "tvly-from-env-file")
        self.assertEqual(child_environment["DEEPSEEK_API_KEY"], "forced")
        self.assertNotIn("BASH_ENV", child_environment)
        self.assertNotIn("ENV", child_environment)
        self.assertNotIn("LD_PRELOAD", child_environment)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", child_environment)

    def test_missing_env_reference_fails_before_launching_server(self):
        with patch("harness.mcp.load_env_key", return_value=None), \
                patch("harness.mcp.subprocess.Popen") as launch:
            server = MCPServer(MCPServerConfig(
                "tavily", "npx", (),
                env=("TAVILY_API_KEY=${TAVILY_API_KEY}",),
            ))
            with self.assertRaisesRegex(MCPError, "TAVILY_API_KEY"):
                server.start()
        launch.assert_not_called()

    def test_reader_ignores_non_object_json_messages(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
            ["not", "an", "object"],
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
        ]
        with patch("harness.mcp.subprocess.Popen", return_value=FakePopen(responses)):
            manager = MCPManager([MCPServerConfig("filesystem", "fake-server", ())])
            self.addCleanup(manager.close)
        self.assertEqual(manager.tools, [])
        self.assertEqual(manager.load_errors, [])

    def test_manager_reconnects_after_server_exits(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{
                "name": "search",
                "description": "搜索文件",
                "inputSchema": {"type": "object", "properties": {}},
            }]}},
        ]
        first = FakePopen(responses)
        second = FakePopen(responses, persistent=True)
        with patch("harness.mcp.subprocess.Popen", side_effect=[first, second]), \
                patch("harness.mcp.RECONNECT_BASE_DELAY", 0.01), \
                patch("harness.mcp.RECONNECT_MAX_DELAY", 0.1):
            manager = MCPManager([
                MCPServerConfig("filesystem", "fake-server", ())
            ])
            self.addCleanup(manager.close)
            deadline = monotonic() + 2
            while monotonic() < deadline and manager.servers[0].process is not second:
                sleep(0.01)
        self.assertIs(manager.servers[0].process, second)
        status = manager.status()[0]
        self.assertEqual(status["state"], "connected")
        self.assertEqual(status["tool_count"], 1)
        self.assertGreaterEqual(status["attempts"], 2)
        self.assertEqual(manager.tools[0].name, "mcp_filesystem_search")


class MCPCLIIntegrationTests(unittest.TestCase):
    def test_external_tool_is_merged_and_forwarded(self):
        definition = ToolDefinition(
            "mcp_filesystem_search", "搜索文件。", {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            validate_arguments=False,
        )
        manager = Mock()
        manager.tool_definitions.return_value = [definition]
        manager.load_errors = []
        manager.tool_names.return_value = ["search"]
        manager.status.return_value = [{
            "name": "filesystem",
            "state": "connected",
            "tool_count": 1,
            "uptime_seconds": 1,
            "last_error": None,
            "attempts": 1,
        }]
        manager.execute.return_value = {
            "status": "success", "content": "found.pdf", "message": "外部工具已调用。",
        }
        settings = deepcopy(get_settings())
        settings["mcp"] = {
            "enabled": True,
            "servers": [{
                "name": "filesystem", "command": "fake-server",
                "args": ["--demo"], "env": [],
            }],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cwd = patch("harness.tools.executor.Path.cwd", return_value=root)
            cwd.start()
            self.addCleanup(cwd.stop)
            tool_call = {
                "id": "mcp_call_1", "type": "function", "function": {
                    "name": definition.name,
                    "arguments": json.dumps({"query": "*.pdf"}, ensure_ascii=False),
                },
            }
            client = Mock(complete=Mock(side_effect=[
                reply(None, tool_calls=[tool_call]),
                reply("找到 1 个 PDF 文件。"),
            ]))
            with patch("harness.cli.get_settings", return_value=settings), \
                    patch("harness.cli.MCPManager", return_value=manager):
                session = CLISession(
                    client, lines=("列出 PDF 文件\n",),
                    terminal=True, character_delay=0,
                    run_cli_kwargs={"mcp_enabled": True},
                )
                self.addCleanup(session.close)
                self.assertTrue(session.output.wait_for(
                    "[确认] 输入 y 批准本次外部 MCP 工具"
                ))
                session.input.send("y\n")
                self.assertTrue(session.output.wait_for("找到 1 个 PDF 文件。"))
                session.close()
        self.assertEqual(client.complete.call_count, 2)
        tool_names = [
            item["function"]["name"]
            for item in client.complete.call_args_list[0].kwargs["tools"]
        ]
        self.assertIn(definition.name, tool_names)
        manager.execute.assert_called_once_with(
            {"query": "*.pdf"}, root, internal_name=definition.name,
        )
        manager.close.assert_called_once()
        output = session.output.getvalue()
        self.assertIn("[mcp] Connecting to filesystem... OK (1 tools)", output)
        self.assertIn("[mcp] Tools merged: search", output)

    def test_mcp_command_reports_server_status(self):
        manager = Mock()
        manager.tool_definitions.return_value = []
        manager.load_errors = []
        manager.tool_names.return_value = []
        manager.status.return_value = [{
            "name": "database",
            "state": "connected",
            "tool_count": 3,
            "uptime_seconds": 120,
            "last_error": None,
            "attempts": 1,
        }]
        client = Mock()
        with patch("harness.cli.load_server_configs", return_value=([], ".harness/mcp.json")), \
                patch("harness.cli.MCPManager", return_value=manager):
            session = CLISession(
                client, lines=("/mcp\n", "/exit\n"),
                terminal=True, character_delay=0,
                run_cli_kwargs={"mcp_enabled": True},
            )
            self.addCleanup(session.close)
            self.assertTrue(session.output.wait_for("MCP Server Status:"))
            session.close()
        client.complete.assert_not_called()
        manager.close.assert_called_once()
        output = session.output.getvalue()
        self.assertIn("database", output)
        self.assertIn("connected", output)
        self.assertIn("3 tools", output)
        self.assertIn("uptime: 2m 0s", output)
        self.assertIn("0 external tools", output)

    def test_first_query_waits_for_background_connection_and_refreshes_tools(self):
        definition = ToolDefinition(
            "mcp_filesystem_search", "搜索文件。", {
                "type": "object", "properties": {},
            },
            validate_arguments=False,
        )
        manager = Mock()
        manager.tool_definitions.side_effect = [[], [definition]]
        manager.load_errors = []
        manager.tool_names.return_value = []
        manager.status.return_value = [{
            "name": "filesystem",
            "state": "connected",
            "tool_count": 1,
            "uptime_seconds": 1,
            "last_error": None,
            "attempts": 1,
        }]
        connect_started = Event()
        release_connection = Event()

        def manager_factory(configs, *, on_change=None, connect=False):
            manager.on_change = on_change
            return manager

        def connect_all():
            connect_started.set()
            if not release_connection.wait(3):
                raise AssertionError("Test did not release MCP connection")
            manager.on_change("filesystem", manager.status.return_value[0])

        manager.connect_all.side_effect = connect_all
        client = Mock(complete=Mock(return_value=reply("回答完成。")))
        with patch("harness.cli.load_server_configs", return_value=([], ".harness/mcp.json")), \
                patch("harness.cli.MCPManager", side_effect=manager_factory):
            session = CLISession(
                client, lines=("你好\n",),
                terminal=False, character_delay=0,
                run_cli_kwargs={"mcp_enabled": True},
            )
            self.addCleanup(session.close)
            self.assertTrue(connect_started.wait(3))
            self.assertFalse(client.complete.called)
            release_connection.set()
            self.assertTrue(session.output.wait_for("回答完成。"))
            session.close()
        tool_names = [
            item["function"]["name"]
            for item in client.complete.call_args_list[0].kwargs["tools"]
        ]
        self.assertIn(definition.name, tool_names)


if __name__ == "__main__":
    unittest.main()
