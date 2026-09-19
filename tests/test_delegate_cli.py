from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from harness.engine import query_loop
from harness.usage import UsageLedger
from test_cli import CLISession, reply, visible_text


def tool_call(name, arguments, call_id):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
    }}


class DelegateCLITests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.file_content = "测试文件资料，仅供子任务分析。"
        (self.root / "sample.txt").write_text(self.file_content, encoding="utf-8")
        cwd = patch("harness.tools.executor.Path.cwd", return_value=self.root)
        cwd.start()
        self.addCleanup(cwd.stop)
        self.requests = []
        self.ledger = UsageLedger(pricing={
            "input_hit_per_million": 1, "input_miss_per_million": 2,
            "output_per_million": 3, "currency": "USD",
        })
        self.delegate = tool_call("delegate", {
            "description": "检查测试文件", "task": "读取 sample.txt 并报告文件内容。",
        }, "delegate_1")
        self.read = tool_call("read_file", {"path": "sample.txt"}, "child_read_1")

    def start(self, responses):
        steps = iter(responses)

        def complete(**request):
            self.requests.append(deepcopy({key: value for key, value in request.items() if key != "on_text"}))
            response = next(steps)
            return response(request) if callable(response) else response

        client = Mock(complete=Mock(side_effect=complete))
        session = CLISession(client, ledger=self.ledger, lines=("请委托检查文件并总结。\n",),
                             terminal=True, character_delay=0)
        self.addCleanup(session.close)
        return session, client

    def streamed(self, text):
        def respond(request):
            request["on_text"](text)
            return reply(text)
        return respond

    def test_child_reads_file_reports_to_parent_and_shares_exact_usage(self):
        child_report = "子任务私有报告：文件检查完成。"
        session, client = self.start([
            reply(None, tool_calls=[self.delegate]), reply(None, tool_calls=[self.read]),
            self.streamed(child_report), self.streamed("主 AI 汇总完成。"),
        ])
        self.assertTrue(session.output.wait_for("主 AI 汇总完成。"))
        session.input.send("/cost\n")
        self.assertTrue(session.output.wait_for("模型请求：4 次。"))
        session.close()
        output = visible_text(session.output.getvalue())
        self.assertIn("[delegate] 启动子任务：检查测试文件", output)
        self.assertIn("[delegate] 子任务完成，返回结果：检查测试文件", output)
        self.assertIn("read_file", output)
        self.assertIn("已读取文件（15 字符）。 本页已到文件末尾。", output)
        self.assertIn("[delegate] [请求 #2／对话 1]", output)
        self.assertIn("[delegate] [请求 #3／对话 1]", output)
        self.assertEqual(output.count("主 AI 汇总完成。"), 1)
        self.assertNotIn(child_report, output)
        self.assertNotIn(self.file_content, output)
        self.assertEqual(client.complete.call_count, 4)
        self.assertEqual(self.ledger.summary()["total_tokens"], 60)
        self.assertAlmostEqual(self.ledger.summary()["estimated_cost"], 0.000132)
        self.assertIn("Token 合计：输入 40，输出 20，合计 60。", output)
        self.assertIn("预估费用合计：USD 0.00013200", output)
        self.assertEqual([record["turn"] for record in self.ledger.records], [1] * 4)
        child_result = next(message for message in self.requests[2]["messages"] if message["role"] == "tool")
        self.assertEqual(child_result["tool_call_id"], "child_read_1")
        self.assertEqual(json.loads(child_result["content"])["content"], self.file_content)
        parent_result = next(message for message in self.requests[3]["messages"] if message["role"] == "tool")
        self.assertEqual(parent_result["tool_call_id"], "delegate_1")
        self.assertEqual(json.loads(parent_result["content"])["content"], child_report)
        self.assertNotIn("delegate", [tool["function"]["name"] for tool in self.requests[1]["tools"]])
        self.assertEqual(session.errors.getvalue(), "")

    def test_cost_stays_available_while_child_request_is_waiting(self):
        started, release = Event(), Event()

        def child_report(request):
            started.set()
            if not release.wait(3):
                raise AssertionError("Child request was not released")
            return self.streamed("子任务等待后的报告。")(request)

        session, client = self.start([
            reply(None, tool_calls=[self.delegate]), reply(None, tool_calls=[self.read]),
            child_report, self.streamed("等待后主回答完成。"),
        ])
        try:
            self.assertTrue(started.wait(3))
            session.input.send("/cost\n")
            self.assertTrue(session.output.wait_for("模型请求：2 次。"))
            self.assertTrue(session.output.wait_for("尚未返回的请求用量将在返回后记录。"))
            self.assertEqual(client.complete.call_count, 3)
            self.assertNotIn("子任务完成，返回结果", session.output.getvalue())
            self.assertNotIn("DeepSeek >", session.output.getvalue())
            release.set()
            self.assertTrue(session.output.wait_for("等待后主回答完成。"))
            self.assertEqual(self.ledger.summary()["requests"], 4)
            self.assertEqual(session.errors.getvalue(), "")
        finally:
            release.set()
            session.close()

    def test_exit_cancels_parent_and_child_before_late_tools_can_run(self):
        started, release, finished = Event(), Event(), Event()
        late_write = tool_call("write_file", {"path": "cancelled/file.txt", "content": "不应写入"}, "late_write")

        def child_reply(_):
            started.set()
            if not release.wait(3):
                raise AssertionError("Child request was not released")
            return reply(None, tool_calls=[late_write])

        def tracked_query(state):
            try:
                return query_loop(state)
            finally:
                finished.set()

        with patch("harness.cli.query_loop", side_effect=tracked_query):
            session, client = self.start([
                reply(None, tool_calls=[self.delegate]), reply(None, tool_calls=[self.read]), child_reply,
            ])
            try:
                self.assertTrue(started.wait(3))
                session.input.send("/exit\n")
                session.join()
                release.set()
                self.assertTrue(finished.wait(3))
                self.assertEqual(client.complete.call_count, 3)
                self.assertEqual(self.ledger.summary()["requests"], 3)
                self.assertFalse((self.root / "cancelled").exists())
                self.assertNotIn("[写入确认]", session.output.getvalue())
                self.assertNotIn("子任务完成，返回结果", session.output.getvalue())
                self.assertNotIn("DeepSeek >", session.output.getvalue())
                self.assertEqual(session.errors.getvalue(), "")
            finally:
                release.set()
                session.close()
                self.assertTrue(finished.wait(3))

    def test_child_failure_is_reported_and_parent_can_finish(self):
        def failed_child(_):
            raise RuntimeError("private child failure")

        session, _ = self.start([
            reply(None, tool_calls=[self.delegate]), failed_child, self.streamed("主 AI 已说明子任务失败。"),
        ])
        self.assertTrue(session.output.wait_for("主 AI 已说明子任务失败。"))
        session.close()
        self.assertIn("[delegate] 子任务失败：检查测试文件；", session.output.getvalue())
        self.assertNotIn("private child failure", session.output.getvalue())
        self.assertNotIn("子任务完成，返回结果", session.output.getvalue())
        result = next(message for message in self.requests[2]["messages"] if message["role"] == "tool")
        self.assertEqual(json.loads(result["content"])["status"], "error")
        self.assertEqual(session.errors.getvalue(), "")

    def test_delegate_event_labels_escape_controls_and_preserve_parent_stream_state(self):
        description = "检查\x1b[2J\n标题"
        message = "失败\r说明\u202e"

        def emit_events(state):
            state.on_event({"type": "response_start"})
            state.on_event({"type": "text", "text": "主流回答。"})
            state.on_event({"type": "delegate_start", "description": description})
            state.on_event({"type": "response_start", "agent": "delegate"})
            state.on_event({"type": "text", "agent": "delegate", "text": "隐藏的子流"})
            for event in (
                {"type": "compact_start", "attempt": 1},
                {"type": "compact_done", "before": 100, "after": 10},
                {"type": "compact_skipped"},
                {"type": "retry", "partial": True, "message": "暂时失败", "delay": 1, "attempt": 1},
            ):
                state.on_event({**event, "agent": "delegate"})
            state.on_event({"type": "delegate_complete", "description": description})
            state.on_event({"type": "delegate_failed", "description": description, "message": message})
            return "主流回答。"

        with patch("harness.cli.query_loop", side_effect=emit_events):
            session, _ = self.start([])
            self.assertTrue(session.output.wait_for("[delegate] 子任务失败："))
            session.close()
        output = visible_text(session.output.getvalue())
        self.assertEqual(output.count("主流回答。"), 1)
        self.assertNotIn("隐藏的子流", output)
        for control in ("\x1b", "\r", "\u202e"):
            self.assertNotIn(control, output)
        self.assertIn("检查\\u001b[2J\\u000a标题", output)
        self.assertIn("失败\\u000d说明\\u202e", output)
        self.assertIn("[delegate] [压缩] 正在总结旧对话", output)
        self.assertIn("[delegate] [压缩] 已将上下文", output)
        self.assertIn("[delegate] [压缩] 历史较短", output)
        self.assertIn("[delegate] [重试] 上次输出未完成", output)
        self.assertIn("[delegate] [重试] 暂时失败", output)
        self.assertEqual(session.errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
