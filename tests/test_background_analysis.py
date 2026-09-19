import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from harness.engine import QueryState, SYSTEM_PROMPT
from harness.orchestration import run_background_analysis
from harness.tools import REGISTRY, ToolRegistry, create_tool_executor
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call
from functools import partial


class BackgroundAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.root.joinpath("note.txt").write_text("needle = 1\n", encoding="utf-8")
        self.registry = ToolRegistry()
        self.registry.register(*REGISTRY.get("read_file"))

    def parent(self, responses, **options):
        tools = [REGISTRY.get("read_file")[0].to_deepseek()]
        settings = {
            "tools": tools,
            "tool_executor": create_tool_executor(self.root, registry=self.registry),
            "max_requests": 5,
        }
        settings.update(options)
        state = QueryState(FakeClient(responses), UsageLedger(), turn=4, **settings)
        state.messages.append({"role": "user", "content": "父会话不应进入后台分析。"})
        return state

    def test_background_analysis_uses_isolated_query_and_returns_report(self):
        events = []
        state = self.parent([
            reply(None, [tool_call("read-1", "read_file", '{"path":"note.txt"}')]),
            reply("note.txt 中定义了 needle。"),
        ])
        result = run_background_analysis(
            parent=state, task="读取 note.txt 并报告", tools=self.registry.definitions(),
            executor_factory=partial(create_tool_executor, self.root, registry=self.registry),
            stop_event=Event(), on_event=events.append,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "note.txt 中定义了 needle。")
        child_messages = state.client.requests[0]["messages"]
        self.assertEqual([message["role"] for message in child_messages], ["system", "user"])
        self.assertEqual(child_messages[0]["content"].startswith(SYSTEM_PROMPT), True)
        self.assertIn("后台独立分析助手", child_messages[0]["content"])
        self.assertNotIn("父会话不应进入后台分析", json.dumps(child_messages, ensure_ascii=False))
        self.assertEqual(state.client.requests[0]["tools"], self.registry.definitions())

    def test_stop_event_cancels_before_background_model_request(self):
        stop = Event()
        stop.set()
        state = self.parent([])
        result = run_background_analysis(
            parent=state, task="不应执行的读取任务", tools=self.registry.definitions(),
            executor_factory=partial(create_tool_executor, self.root, registry=self.registry),
            stop_event=stop, on_event=lambda event: None,
        )
        self.assertEqual(result["code"], "background_cancelled")
        self.assertEqual(state.client.requests, [])


if __name__ == "__main__":
    unittest.main()
