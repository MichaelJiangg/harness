import copy
import unittest

from harness.client import APIError
from harness.engine import QueryAborted, QueryState, SYSTEM_PROMPT, query_loop
from harness.usage import UsageLedger


def tool_call(call_id="call-1", name="test_read", arguments='{"path":"example.txt"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(content, calls=None, reason=None):
    return {
        "model": "deepseek-flash", "created": 1789696800,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                  "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60},
        "choices": [{
            "finish_reason": reason or ("tool_calls" if calls is not None else "stop"),
            "message": {"role": "assistant", "content": content,
                        **({"tool_calls": calls} if calls is not None else {})},
        }],
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, **request):
        self.requests.append(copy.deepcopy(request))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class QueryLoopTests(unittest.TestCase):
    def make_state(self, responses, **kwargs):
        state = QueryState(FakeClient(responses), UsageLedger(), **kwargs)
        state.messages.append({"role": "user", "content": "帮我查看项目"})
        return state

    def test_text_answer_finishes_and_records_usage(self):
        events = []
        state = self.make_state([reply("最终回答")], on_event=events.append)
        self.assertEqual(query_loop(state), "最终回答")
        self.assertEqual(len(state.client.requests), 1)
        self.assertEqual(state.ledger.summary()["total_tokens"], 120)
        self.assertEqual(state.messages[-1]["content"], "最终回答")
        self.assertEqual([event["type"] for event in events], ["response_start", "usage"])

    def test_system_prompt_requires_delegate_for_directory_analysis(self):
        self.assertIn("目录级分析", SYSTEM_PROMPT)
        self.assertIn("代码质量审查", SYSTEM_PROMPT)
        self.assertIn("必须优先调用 delegate", SYSTEM_PROMPT)
        self.assertIn("不直接在主会话逐文件读取", SYSTEM_PROMPT)
        self.assertIn("background_submit", SYSTEM_PROMPT)
        self.assertIn("background_check", SYSTEM_PROMPT)
        self.assertIn("swarm", SYSTEM_PROMPT)

    def test_multiple_tools_and_rounds_preserve_assistant_and_results(self):
        import json
        calls = [tool_call("first"), tool_call("second", "test_command", '{"command":"pwd"}')]
        executed = []

        def execute(name, arguments):
            executed.append((name, arguments))
            return {"status": "success", "executed": True, "tool": name, "message": "测试工具完成。"}

        state = self.make_state([
            reply("我先查看", calls), reply(None, [tool_call("third")]), reply("检查完成。"),
        ], tool_executor=execute)
        self.assertEqual(query_loop(state), "检查完成。")
        self.assertEqual(len(state.client.requests), 3)
        second_request = state.client.requests[1]["messages"]
        self.assertEqual(second_request[2]["tool_calls"], calls)
        self.assertEqual([item["tool_call_id"] for item in second_request[3:]], ["first", "second"])
        for item in second_request[3:]:
            self.assertEqual(item["role"], "tool")
            self.assertEqual(json.loads(item["content"])["status"], "success")
            self.assertTrue(json.loads(item["content"])["executed"])
        self.assertEqual(executed, [
            ("test_read", {"path": "example.txt"}),
            ("test_command", {"command": "pwd"}),
            ("test_read", {"path": "example.txt"}),
        ])
        self.assertEqual(state.ledger.summary()["total_tokens"], 360)
        self.assertEqual([record["turn"] for record in state.ledger.records], [1, 1, 1])

    def test_next_turn_has_previous_conversation(self):
        state = self.make_state([reply("第一答"), reply("第二答")])
        query_loop(state)
        state.messages.append({"role": "user", "content": "继续解释"})
        state.turn += 1
        self.assertEqual(query_loop(state), "第二答")
        self.assertEqual(state.client.requests[1]["messages"][-2:], [
            {"role": "assistant", "content": "第一答"}, {"role": "user", "content": "继续解释"},
        ])
        self.assertEqual(state.ledger.records[-1]["turn"], 2)

    def test_invalid_json_and_unknown_tools_are_returned_as_errors(self):
        import json
        state = self.make_state([
            reply(None, [tool_call("bad", arguments="{"), tool_call("unknown", "unknown", "{}")]),
            reply("参数无效"),
        ])
        query_loop(state)
        results = state.client.requests[1]["messages"][-2:]
        self.assertEqual([json.loads(item["content"])["status"] for item in results], ["error", "error"])

    def test_tool_exception_is_returned_without_leaking_details(self):
        def execute(name, arguments):
            raise RuntimeError("private-tool-details")
        state = self.make_state([reply(None, [tool_call()]), reply("工具失败")], tool_executor=execute)
        query_loop(state)
        result = state.client.requests[1]["messages"][-1]["content"]
        self.assertIn('"status": "error"', result)
        self.assertNotIn("private-tool-details", result)

    def test_loop_limit_preserves_token_cost(self):
        state = self.make_state([reply(None, [tool_call()]), reply(None, [tool_call()])], max_requests=2)
        with self.assertRaisesRegex(RuntimeError, "2 次模型请求上限"):
            query_loop(state)
        self.assertEqual(len(state.client.requests), 2)
        self.assertEqual(state.ledger.summary()["total_tokens"], 240)

    def test_api_failure_is_unknown_usage(self):
        state = self.make_state([reply(None, [tool_call()]), APIError("网络失败")])
        with self.assertRaises(APIError):
            query_loop(state)
        self.assertEqual(state.ledger.summary()["missing_usage_requests"], 1)
        self.assertEqual(state.ledger.summary()["total_tokens"], 120)

    def test_non_success_reasons_and_malformed_responses_do_not_finish(self):
        for response in [reply("部分回答", reason=reason) for reason in
                         ["length", "content_filter", "insufficient_system_resource", "aborted"]] + [
            reply(""), reply(None, []), reply(None, [tool_call("")]),
            reply(None, [tool_call("same"), tool_call("same")]), {"choices": []}, [],
        ]:
            with self.subTest(response=response):
                state = self.make_state([response])
                with self.assertRaises(RuntimeError):
                    query_loop(state)

    def test_missing_usage_does_not_prevent_answer(self):
        response = reply("正常答案")
        response.pop("usage")
        state = self.make_state([response])
        self.assertEqual(query_loop(state), "正常答案")
        self.assertTrue(state.ledger.records[0]["usage_missing"])
        self.assertIsNone(state.ledger.records[0]["estimated_cost"])

    def test_created_reaches_ledger_for_peak_price(self):
        from datetime import datetime, timezone
        response = reply("峰段回答")
        response["created"] = datetime(2026, 9, 18, 2, tzinfo=timezone.utc).timestamp()
        state = self.make_state([response])
        query_loop(state)
        self.assertEqual(state.ledger.records[0]["rate_period"], "peak")
        self.assertEqual(state.ledger.records[0]["created"], response["created"])

    def test_abort_before_request_never_calls_model(self):
        state = self.make_state([])
        state.abort.set()
        with self.assertRaises(QueryAborted):
            query_loop(state)
        self.assertEqual(state.client.requests, [])

    def test_abort_after_response_records_usage_but_never_executes_tool(self):
        state = self.make_state([reply(None, [tool_call()])])
        state.on_event = lambda event: state.abort.set() if event["type"] == "usage" else None
        state.tool_executor = lambda *args: self.fail("中止后不得执行工具")
        with self.assertRaises(QueryAborted):
            query_loop(state)
        self.assertEqual(state.ledger.summary()["total_tokens"], 120)


if __name__ == "__main__":
    unittest.main()
