from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from harness.context import context_size
from harness.engine import ContextTooLong, QueryAborted, QueryState, SYSTEM_PROMPT, query_loop
from harness.tools import create_tool_executor, get_tool_definitions
from harness.usage import UsageLedger
from test_engine import reply, tool_call


class PagingClient:
    """模拟模型只依据上一页返回的游标续读，不预先知道文件内容。"""

    def __init__(self, path):
        self.path = path
        self.requests = []
        self.pages = []
        self.summary_inputs = []
        self._seen = set()
        self._call_count = 0

    def complete(self, **request):
        self.requests.append(deepcopy(request))
        if request["tools"] == []:
            self.summary_inputs.append(deepcopy(request["messages"]))
            return reply("已分析早期页面，保留原任务，继续从最后一页的下一游标读取。")
        latest = next((message for message in reversed(request["messages"])
                       if message["role"] == "tool"), None)
        arguments = {"path": self.path}
        if latest is not None:
            page = json.loads(latest["content"])
            if latest["tool_call_id"] not in self._seen:
                self.pages.append(page)
                self._seen.add(latest["tool_call_id"])
            if page["eof"]:
                return reply("文件各页已读取完毕。")
            arguments.update(offset=page["next_offset"], column=page["next_column"])
        self._call_count += 1
        return reply(None, [tool_call(f"page-{self._call_count}", "read_file",
                                     json.dumps(arguments, ensure_ascii=False))])


class DelegatingPagingClient:
    def __init__(self, path):
        self.child = PagingClient(path)
        self.parent_requests = []
        self.result = None

    def complete(self, **request):
        if not any(tool["function"]["name"] == "delegate" for tool in request["tools"]):
            return self.child.complete(**request)
        self.parent_requests.append(deepcopy(request))
        if len(self.parent_requests) == 1:
            return reply(None, [tool_call("delegate-pages", "delegate", json.dumps({
                "description": "完整分析分页文件", "task": f"请从头到尾读取 {self.child.path} 并报告，不遗漏中间页面。",
            }, ensure_ascii=False))])
        self.result = json.loads(request["messages"][-1]["content"])
        return reply("主 AI 已整理子任务结果。")


class PageManagementTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = "pages.txt"
        self.tools = [tool for tool in get_tool_definitions() if tool["function"]["name"] == "read_file"]

    def state(self, content, **options):
        (self.root / self.path).write_bytes(content.encode("utf-8"))
        client = PagingClient(self.path)
        settings = {
            "tools": self.tools, "tool_executor": create_tool_executor(self.root),
            "tool_result_limit": 1400, "context_limit": 100000, "max_requests": 20,
        }
        settings.update(options)
        state = QueryState(client, UsageLedger(), **settings)
        state.messages.append({"role": "user", "content": "请从头到尾读取 pages.txt，分析完整文件，不遗漏中间内容。"})
        return state

    def assert_complete_tool_batches(self, messages):
        pending = set()
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                self.assertFalse(pending)
                pending = {call["id"] for call in message["tool_calls"]}
            elif message["role"] == "tool":
                self.assertIn(message["tool_call_id"], pending)
                pending.remove(message["tool_call_id"])
            else:
                self.assertFalse(pending)
        self.assertFalse(pending)

    def assert_resume_cursor(self, error, page):
        message = str(error)
        self.assertIn(self.path, message)
        for name in ("next_offset", "next_column"):
            self.assertRegex(message, rf'{name}["\x27]?\s*[:=]\s*{page[name]}\b')

    def test_real_file_pages_reconstruct_full_content_without_generic_truncation(self):
        content = "首行" + "长" * 2300 + "\r\n" + "短行\n" * 320 + "尾行"
        state = self.state(content)
        self.assertEqual(query_loop(state), "文件各页已读取完毕。")
        pages = state.client.pages
        self.assertGreater(len(pages), 2)
        self.assertEqual("".join(page["content"] for page in pages), content)
        self.assertFalse(pages[0]["eof"])
        self.assertTrue(pages[-1]["eof"])
        self.assertTrue(any(page["next_column"] > 0 for page in pages[:-1]))
        self.assertEqual(pages[0]["total_lines"], 322)
        cursor = (0, 0)
        for page in pages:
            self.assertEqual((page["offset"], page["column"]), cursor)
            self.assertEqual(page["status"], "success")
            self.assertTrue(page["executed"])
            self.assertNotIn("head", page)
            self.assertNotIn("tail", page)
            self.assertLessEqual(len(json.dumps(page, ensure_ascii=False)), state.tool_result_limit)
            cursor = (page["next_offset"], page["next_column"])
        self.assertEqual(state.ledger.summary()["requests"], len(pages) + 1)
        for request in state.client.requests:
            self.assert_complete_tool_batches(request["messages"])

    def test_main_current_turn_compacts_old_pages_and_keeps_recent_user_turn(self):
        content = "".join(f"第 {index} 行：" + "文" * 290 + "\n" for index in range(20))
        events = []
        state = self.state(content, keep_recent_turns=1, summary_limit=200, on_event=events.append)
        previous = [
            {"role": "user", "content": "之前的要求：保留中文行号。"},
            {"role": "assistant", "content": "会保留中文行号。"},
        ]
        state.messages[1:1] = previous
        original_task = deepcopy(state.messages[-1])
        state.context_limit = context_size(state.messages, state.tools) + 3 * state.tool_result_limit + 300
        self.assertEqual(query_loop(state), "文件各页已读取完毕。")
        self.assertTrue(state.client.summary_inputs)
        self.assertGreater(state.compaction_count, 0)
        self.assertLessEqual(state.compaction_count, state.max_compactions)
        self.assertEqual("".join(page["content"] for page in state.client.pages), content)
        self.assertTrue(state.client.pages[-1]["eof"])
        self.assertTrue(any(event["type"] == "compact_done" for event in events))
        summarized = [request for request in state.client.requests if request["tools"]
                      and any("[历史对话摘要]" in (message.get("content") or "")
                              for message in request["messages"])]
        self.assertTrue(summarized)
        for request in summarized:
            self.assertEqual(request["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
            self.assertIn(original_task, request["messages"])
            for message in previous:
                self.assertIn(message, request["messages"])
            self.assertLessEqual(context_size(request["messages"], request["tools"]), state.context_limit)
            self.assert_complete_tool_batches(request["messages"])
        for request in state.client.summary_inputs:
            summary_source = json.loads(request[-1]["content"].split("\n", 1)[1])
            self.assert_complete_tool_batches(summary_source)
            for message in previous + [original_task]:
                self.assertNotIn(message, summary_source)

    def test_single_current_tool_batch_over_budget_can_grow_before_compaction(self):
        content = "首行" + "长" * 2300 + "\r\n" + "短行\n" * 320 + "尾行"
        state = self.state(content)
        state.context_limit = (context_size(state.messages, state.tools)
                               + state.tool_result_limit - 200)
        self.assertEqual(query_loop(state), "文件各页已读取完毕。")
        self.assertTrue(state.client.summary_inputs)
        self.assertEqual("".join(page["content"] for page in state.client.pages), content)
        self.assertTrue(state.client.pages[-1]["eof"])

    def test_request_limit_error_reports_the_latest_unread_cursor(self):
        state = self.state("很长的行" * 2500, max_requests=2)
        with self.assertRaises(RuntimeError) as raised:
            query_loop(state)
        self.assertIn("模型请求上限", str(raised.exception))
        latest = json.loads(state.messages[-1]["content"])
        self.assertFalse(latest["eof"])
        self.assertGreater(latest["next_column"], 0)
        self.assert_resume_cursor(raised.exception, latest)
        self.assertEqual(len(state.client.requests), 2)
        self.assertEqual(state.ledger.summary()["requests"], 2)

    def test_next_query_clears_previous_incomplete_read_progress(self):
        state = self.state("长" * 9000, max_requests=1)
        with self.assertRaises(RuntimeError) as first:
            query_loop(state)
        self.assertIn(self.path, str(first.exception))
        state.messages.append({"role": "user", "content": "这是一个新的问题。"})
        state.max_requests = 0
        with self.assertRaises(RuntimeError) as second:
            query_loop(state)
        self.assertNotIn(self.path, str(second.exception))
        self.assertNotIn("next_offset", str(second.exception))
        self.assertEqual(state.read_progress, {})
        self.assertEqual(state.ledger.summary()["requests"], 1)

    def test_delegate_request_exhaustion_returns_latest_cursor_to_parent(self):
        events = []
        state = self.state("长" * 9000, max_requests=4, on_event=events.append)
        state.tools = [tool for tool in get_tool_definitions()
                       if tool["function"]["name"] in {"delegate", "read_file"}]
        state.client = DelegatingPagingClient(self.path)
        self.assertEqual(query_loop(state), "主 AI 已整理子任务结果。")
        result = state.client.result
        self.assertEqual(result["code"], "delegate_request_limit")
        self.assertEqual(result["status"], "error")
        reads = [event["result"] for event in events
                 if event["type"] == "tool" and event.get("agent") == "delegate"]
        self.assertEqual(len(reads), 2)
        latest = reads[-1]
        self.assertFalse(latest["eof"])
        self.assertEqual(result["progress"], [{
            "path": self.path, "next_offset": latest["next_offset"],
            "next_column": latest["next_column"], "eof": False,
        }])
        self.assertEqual(state.ledger.summary()["requests"], 4)

    def test_delegate_compacts_earlier_pages_and_continues_to_eof(self):
        content = "".join(f"第 {index} 行：" + "文" * 290 + "\n" for index in range(24))
        events = []
        state = self.state(content, summary_limit=200, on_event=events.append)
        state.tools = [tool for tool in get_tool_definitions()
                       if tool["function"]["name"] in {"delegate", "read_file"}]
        state.context_limit = context_size(state.messages, state.tools) + 3 * state.tool_result_limit + 300
        state.client = DelegatingPagingClient(self.path)
        self.assertEqual(query_loop(state), "主 AI 已整理子任务结果。")
        child = state.client.child
        self.assertEqual(state.client.result["status"], "success")
        self.assertTrue(child.summary_inputs)
        self.assertEqual("".join(page["content"] for page in child.pages), content)
        self.assertTrue(child.pages[-1]["eof"])
        self.assertTrue(any(event["type"] == "compact_done" and event.get("agent") == "delegate"
                            for event in events))
        self.assertEqual(state.ledger.summary()["requests"], len(child.requests) + 2)
        for request in child.requests:
            if request["tools"]:
                self.assert_complete_tool_batches(request["messages"])

    def test_exhausted_compaction_reports_cursor_without_claiming_file_completion(self):
        state = self.state("文" * 8000, max_compactions=0)
        state.context_limit = context_size(state.messages, state.tools) + 2 * state.tool_result_limit + 100
        with self.assertRaises(ContextTooLong) as raised:
            query_loop(state)
        latest = json.loads(state.messages[-1]["content"])
        self.assertFalse(latest["eof"])
        self.assertIn("建议开个新会话", str(raised.exception))
        self.assert_resume_cursor(raised.exception, latest)
        self.assertFalse(state.client.summary_inputs)

    def test_cancellation_after_first_page_stops_model_and_read_continuation(self):
        events = []
        state = self.state("长" * 9000)

        def cancel_after_read(event):
            events.append(deepcopy(event))
            if event["type"] == "tool":
                state.abort.set()

        state.on_event = cancel_after_read
        with self.assertRaises(QueryAborted):
            query_loop(state)
        reads = [event for event in events if event["type"] == "tool"]
        self.assertEqual(len(reads), 1)
        self.assertFalse(reads[0]["result"]["eof"])
        self.assertEqual(len(state.client.requests), 1)
        self.assertEqual(state.ledger.summary()["requests"], 1)


if __name__ == "__main__":
    unittest.main()
