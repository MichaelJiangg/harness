import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from harness.client import DeepSeekClient
from harness.engine import QueryState, query_loop
from harness.tools import create_tool_executor
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call
from test_streaming import delta, sse, usage


def read_call(identifier, path, **options):
    return tool_call(identifier, "read_file", json.dumps({"path": path, **options}, ensure_ascii=False))


class ReadFileIntegrationTests(unittest.TestCase):
    def test_default_executor_keeps_initial_workspace_and_returns_file_to_model(self):
        with TemporaryDirectory() as initial, TemporaryDirectory() as later:
            content = "查询引擎测试文件。\n第二行。"
            Path(initial, "说明.txt").write_text(content, encoding="utf-8")
            Path(later, "说明.txt").write_text("不应读到另一个目录。", encoding="utf-8")
            client = FakeClient([reply(None, [read_call("read-1", "说明.txt")]), reply("文件包含两行。")])
            with patch("harness.tools.executor.Path.cwd", return_value=Path(initial)):
                state = QueryState(client, UsageLedger())
            state.messages.append({"role": "user", "content": "请读说明.txt"})
            with patch("harness.tools.executor.Path.cwd", return_value=Path(later)):
                self.assertEqual(query_loop(state), "文件包含两行。")

            definitions = client.requests[0]["tools"]
            self.assertEqual([item["function"]["name"] for item in definitions],
                             ["background_check", "background_submit", "bash", "delegate",
                              "grep", "notes_append", "notes_read", "notes_replace",
                              "read_file", "run_verify", "swarm", "web_fetch",
                              "web_search", "write_file"])
            definition = next(item for item in definitions if item["function"]["name"] == "read_file")
            self.assertEqual(definition["type"], "function")
            function = definition["function"]
            self.assertEqual(function["name"], "read_file")
            self.assertEqual(function["parameters"]["required"], ["path"])
            self.assertEqual(function["parameters"]["properties"]["path"]["type"], "string")
            self.assertFalse(function["parameters"]["additionalProperties"])
            messages = client.requests[1]["messages"]
            self.assertEqual(messages[-2]["tool_calls"][0]["id"], "read-1")
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(messages[-1]["tool_call_id"], "read-1")
            result = json.loads(messages[-1]["content"])
            self.assertEqual(result["content"], content)
            self.assertEqual(result["path"], "说明.txt")
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["executed"])
            self.assertEqual(state.ledger.summary()["total_tokens"], 240)

    def test_model_can_correct_pagination_arguments_and_receive_selected_lines(self):
        with TemporaryDirectory() as workspace:
            Path(workspace, "lines.txt").write_bytes("第一行\r\n第二行\n第三行".encode("utf-8"))
            client = FakeClient([
                reply(None, [read_call("invalid", "lines.txt", offset=-1, limit=1)]),
                reply(None, [read_call("corrected", "lines.txt", offset=1, limit=1)]),
                reply("第二行已读取。"),
            ])
            state = QueryState(client, UsageLedger(), tool_executor=create_tool_executor(workspace))
            state.messages.append({"role": "user", "content": "读取第二行"})
            self.assertEqual(query_loop(state), "第二行已读取。")
            failure = client.requests[1]["messages"][-1]
            self.assertEqual(failure["tool_call_id"], "invalid")
            self.assertEqual(json.loads(failure["content"])["code"], "invalid_arguments")
            success = client.requests[2]["messages"][-1]
            self.assertEqual(success["tool_call_id"], "corrected")
            self.assertEqual(json.loads(success["content"])["content"], "第二行\n")
            self.assertEqual(state.ledger.summary()["requests"], 3)

    def test_missing_file_error_allows_model_to_correct_path(self):
        with TemporaryDirectory() as workspace:
            Path(workspace, "correct.txt").write_text("修正路径后读取成功。", encoding="utf-8")
            client = FakeClient([
                reply(None, [read_call("missing", "missing.txt")]),
                reply(None, [read_call("corrected", "correct.txt")]),
                reply("已经找到正确文件。"),
            ])
            state = QueryState(client, UsageLedger(), tool_executor=create_tool_executor(workspace))
            state.messages.append({"role": "user", "content": "请读文件"})
            self.assertEqual(query_loop(state), "已经找到正确文件。")
            failure = client.requests[1]["messages"][-1]
            self.assertEqual(failure["tool_call_id"], "missing")
            failure_result = json.loads(failure["content"])
            self.assertEqual(failure_result["status"], "error")
            self.assertEqual(failure_result["code"], "not_found")
            self.assertFalse(failure_result["executed"])
            self.assertIn("不存在", failure_result["message"])
            success = client.requests[2]["messages"][-1]
            self.assertEqual(success["tool_call_id"], "corrected")
            self.assertEqual(json.loads(success["content"])["content"], "修正路径后读取成功。")
            self.assertEqual(state.ledger.summary()["requests"], 3)

    def test_large_file_returns_a_bounded_page_with_a_usable_continuation_cursor(self):
        with TemporaryDirectory() as workspace:
            content = "头部标记" + "长" * 10000 + "尾部标记"
            Path(workspace, "long.txt").write_text(content, encoding="utf-8")
            client = FakeClient([reply(None, [read_call("long", "long.txt")]), reply("已返回第一页。")])
            events = []
            state = QueryState(client, UsageLedger(), tool_executor=create_tool_executor(workspace),
                               on_event=events.append, tool_result_limit=1500)
            state.messages.append({"role": "user", "content": "读取长文件"})
            self.assertEqual(query_loop(state), "已返回第一页。")
            message = client.requests[1]["messages"][-1]
            self.assertEqual(message["tool_call_id"], "long")
            self.assertLessEqual(len(message["content"]), state.tool_result_limit)
            page = json.loads(message["content"])
            self.assertEqual(page["status"], "success")
            self.assertTrue(page["executed"])
            self.assertEqual(page["content"], content[:len(page["content"])])
            self.assertTrue(page["content"].startswith("头部标记"))
            self.assertNotIn("尾部标记", page["content"])
            self.assertNotIn("head", page)
            self.assertNotIn("tail", page)
            self.assertFalse(page["eof"])
            self.assertEqual((page["offset"], page["column"]), (0, 0))
            self.assertEqual((page["next_offset"], page["next_column"]), (0, len(page["content"])))
            self.assertGreater(page["next_column"], 0)
            self.assertEqual(page["total_lines"], 1)
            result_event = next(event for event in events if event["type"] == "tool")
            self.assertEqual(result_event["result"]["content"], page["content"])
            self.assertEqual(result_event["result"]["next_column"], page["next_column"])

    def test_streaming_tool_fragments_execute_real_read_and_preserve_usage(self):
        with TemporaryDirectory() as workspace:
            Path(workspace, "stream.txt").write_text("流式读取内容。", encoding="utf-8")
            responses = [
                sse(delta(calls=[{
                    "index": 0, "id": "stream-read", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"'},
                }]), delta(calls=[{
                    "index": 0, "function": {"arguments": 'stream.txt"}'},
                }], reason="tool_calls"), usage(), "[DONE]"),
                sse(delta("读取"), delta("完成。", reason="stop"), usage(20, 3), "[DONE]"),
            ]
            requests = []
            events = []

            def opener(request, **kwargs):
                requests.append(json.loads(request.data))
                return io.BytesIO(responses.pop(0))

            state = QueryState(DeepSeekClient("test-key", opener=opener), UsageLedger(),
                               tool_executor=create_tool_executor(workspace), on_event=events.append)
            state.messages.append({"role": "user", "content": "读取 stream.txt"})
            self.assertEqual(query_loop(state), "读取完成。")
            result = requests[1]["messages"][-1]
            self.assertEqual(result["tool_call_id"], "stream-read")
            self.assertEqual(json.loads(result["content"])["content"], "流式读取内容。")
            self.assertEqual([event["text"] for event in events if event["type"] == "text"], ["读取", "完成。"])
            self.assertEqual([record["total_tokens"] for record in state.ledger.records], [15, 23])


if __name__ == "__main__":
    unittest.main()
