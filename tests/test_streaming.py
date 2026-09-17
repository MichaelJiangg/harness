import io
import json
from http.client import IncompleteRead
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

from harness.client import DEFAULT_MODEL, DeepSeekClient
from harness.engine import QueryState, query_loop
from harness.usage import UsageLedger
from test_cli import CLISession, ObservableOutput


def sse(*chunks):
    return b"".join(
        ("data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk, ensure_ascii=False))
         + "\n\n").encode("utf-8")
        for chunk in chunks
    )


def delta(content=None, *, calls=None, reason=None):
    value = {"content": content}
    if calls is not None:
        value["tool_calls"] = calls
    return {
        "model": DEFAULT_MODEL,
        "choices": [{"index": 0, "delta": value, "finish_reason": reason}],
    }


def usage(prompt=10, completion=5):
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": prompt,
        },
    }


class PausedResponse(io.BytesIO):
    """在第一段 SSE 已读取后暂停，尾段只能由测试显式放行。"""

    def __init__(self, first, rest):
        super().__init__(first + rest)
        self.boundary = len(first)
        self.waiting = Event()
        self.release = Event()

    def __next__(self):
        if self.tell() == self.boundary and not self.waiting.is_set():
            self.waiting.set()
            if not self.release.wait(3):
                raise AssertionError("Test did not release the remaining SSE response")
        return super().__next__()


class StreamingIntegrationTests(unittest.TestCase):
    def test_chinese_text_is_flushed_before_tail_and_final_answer_is_not_repeated(self):
        response = PausedResponse(
            sse(delta("第一段中文")),
            sse(delta("，随后完成。", reason="stop"), usage(), "[DONE]"),
        )
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: response)
        flushed = Event()

        def observe_flush(output):
            if "DeepSeek > 第一段中文" in output.getvalue():
                flushed.set()

        with patch.object(ObservableOutput, "flush", observe_flush):
            session = CLISession(client, lines=("你好\n", ""))
            try:
                self.assertTrue(response.waiting.wait(3))
                self.assertTrue(flushed.wait(3))
                self.assertFalse(session.finished.is_set())
                self.assertNotIn("随后完成", session.output.getvalue())
                response.release.set()
                session.join()
                output = session.output.getvalue()
                self.assertEqual(output.count("DeepSeek >"), 1)
                self.assertEqual(output.count("第一段中文，随后完成。"), 1)
                self.assertIn("合计 15 token", output)
                self.assertEqual(session.errors.getvalue(), "")
            finally:
                response.release.set()
                session.close()

    def test_fragmented_tools_are_assembled_and_results_and_usage_survive_the_loop(self):
        first = sse(
            delta("开始检查。", calls=[
                {"index": 0, "id": "call_read", "type": "function",
                 "function": {"name": "test_read", "arguments": '{"path":"'}},
                {"index": 1, "id": "call_command", "type": "function",
                 "function": {"name": "test_command", "arguments": '{"command":"'}},
            ]),
            delta(calls=[
                {"index": 1, "function": {"arguments": 'pwd"}'}},
                {"index": 0, "function": {"arguments": '说明.md"}'}},
            ], reason="tool_calls"),
            usage(), "[DONE]",
        )
        responses = [first, sse(delta("检查完成。", reason="stop"), usage(20, 3), "[DONE]")]
        requests = []
        events = []

        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return io.BytesIO(responses.pop(0))

        execute = Mock(return_value={"status": "success", "executed": True, "message": "测试工具完成。"})
        state = QueryState(DeepSeekClient("test-key", opener=opener), UsageLedger(),
                           on_event=events.append, tool_executor=execute)
        state.messages.append({"role": "user", "content": "检查项目"})
        self.assertEqual(query_loop(state), "检查完成。")
        self.assertEqual(len(requests), 2)
        messages = requests[1]["messages"]
        calls = messages[2]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_read", "call_command"])
        self.assertEqual([json.loads(call["function"]["arguments"]) for call in calls], [
            {"path": "说明.md"}, {"command": "pwd"},
        ])
        self.assertEqual([message["tool_call_id"] for message in messages[3:]], ["call_read", "call_command"])
        for message in messages[3:]:
            result = json.loads(message["content"])
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["executed"])
        self.assertEqual([call.args for call in execute.call_args_list], [
            ("test_read", {"path": "说明.md"}), ("test_command", {"command": "pwd"}),
        ])
        self.assertEqual([event["text"] for event in events if event["type"] == "text"], [
            "开始检查。", "检查完成。",
        ])
        self.assertEqual([record["total_tokens"] for record in state.ledger.records], [15, 23])
        self.assertEqual([record["turn"] for record in state.ledger.records], [1, 1])
        self.assertEqual(state.ledger.summary()["total_tokens"], 38)

    def test_cost_remains_available_between_stream_fragments_and_counts_finished_requests(self):
        first = io.BytesIO(sse(delta(calls=[{
            "index": 0, "id": "call_read", "type": "function",
            "function": {"name": "test_read", "arguments": '{"path":"example.txt"}'},
        }], reason="tool_calls"), usage(), "[DONE]"))
        second = PausedResponse(
            sse(delta("正在说明")),
            sse(delta("占位结果。", reason="stop"), usage(20, 3), "[DONE]"),
        )
        responses = [first, second]
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: responses.pop(0))
        ledger = UsageLedger()
        execute = Mock(return_value={"status": "success", "executed": True, "message": "测试工具完成。"})
        factory = patch("harness.cli.create_tool_executor", return_value=execute)
        factory.start()
        self.addCleanup(factory.stop)
        session = CLISession(client, ledger=ledger, lines=("读取文件\n",))
        try:
            self.assertTrue(second.waiting.wait(3))
            session.input.send("/cost\n")
            self.assertTrue(session.output.wait_for("尚未返回的请求用量将在返回后记录。"))
            output = session.output.getvalue()
            self.assertIn("DeepSeek > 正在说明\n本次会话用量与费用", output)
            self.assertIn("模型请求：1 次。", output)
            self.assertIn("Token 合计：输入 10，输出 5，合计 15。", output)
            self.assertEqual(ledger.summary()["requests"], 1)
            second.release.set()
            session.close()
            self.assertIn("DeepSeek > 占位结果。", session.output.getvalue())
            self.assertEqual(session.output.getvalue().count("正在说明"), 1)
            self.assertEqual(ledger.summary()["requests"], 2)
            self.assertEqual(ledger.summary()["total_tokens"], 38)
            self.assertEqual(session.errors.getvalue(), "")
        finally:
            second.release.set()
            session.close()

    def test_disconnect_keeps_partial_display_but_does_not_commit_failed_conversation(self):
        class BrokenResponse(io.BytesIO):
            def __next__(self):
                try:
                    return super().__next__()
                except StopIteration:
                    raise IncompleteRead(b"private-response-body", 100) from None

        responses = [BrokenResponse(sse(delta("仅收到部分回答"))) for _ in range(4)]
        responses.append(io.BytesIO(sse(delta("重试成功。", reason="stop"), usage(), "[DONE]")))
        requests = []
        workers = []

        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return responses.pop(0)

        def create_worker(*args, **kwargs):
            worker = Thread(*args, **kwargs)
            workers.append(worker)
            return worker

        def query_without_retry_delay(state):
            with patch.object(state.abort, "wait", return_value=False):
                return query_loop(state)

        ledger = UsageLedger()
        with patch("harness.cli.Thread", side_effect=create_worker), \
                patch("harness.cli.query_loop", side_effect=query_without_retry_delay):
            session = CLISession(DeepSeekClient("test-key", opener=opener), ledger=ledger, lines=("失败的问题\n",))
            try:
                self.assertTrue(session.output.wait_for("仅收到部分回答"))
                workers[0].join(timeout=3)
                self.assertFalse(workers[0].is_alive())
                self.assertIn("连接中断", session.errors.getvalue())
                self.assertNotIn("private-response-body", session.errors.getvalue())
                self.assertEqual(ledger.summary()["missing_usage_requests"], 4)
                session.input.send("新的问题\n")
                self.assertTrue(session.output.wait_for("DeepSeek > 重试成功。"))
                session.close()
                self.assertEqual(requests[4]["messages"][1:], [{"role": "user", "content": "新的问题"}])
                self.assertEqual(session.output.getvalue().count("仅收到部分回答"), 4)
                self.assertEqual(ledger.summary()["requests"], 5)
                self.assertEqual(ledger.summary()["missing_usage_requests"], 4)
                self.assertEqual(ledger.summary()["total_tokens"], 15)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
