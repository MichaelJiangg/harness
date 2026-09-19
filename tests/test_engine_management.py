from copy import deepcopy
import json
from threading import Event
import unittest
from unittest.mock import Mock, call

from harness.client import APIError
from harness.context import context_size
from harness.engine import ContextTooLong, QueryAborted, QueryState, compact_history, query_loop
from harness.usage import UsageLedger


def reply(text, calls=None, reason=None):
    return {
        "model": "deepseek-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                  "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10},
        "choices": [{"finish_reason": reason or ("tool_calls" if calls else "stop"),
                     "message": {"role": "assistant", "content": text,
                                 **({"tool_calls": calls} if calls else {})}}],
    }


def tool_call(identifier="read-1"):
    return {"id": identifier, "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'}}


def history(count=6):
    messages = [{"role": "system", "content": "系统约束保持不变"}]
    for index in range(count):
        messages.extend([{"role": "user", "content": f"旧问题{index}"},
                         {"role": "assistant", "content": f"旧回答{index}" + "资料" * 200}])
    return messages


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, **request):
        self.requests.append(deepcopy({key: value for key, value in request.items() if key != "on_text"}))
        response = self.responses.pop(0)
        if callable(response):
            return response(request)
        if isinstance(response, Exception):
            raise response
        message = response["choices"][0]["message"]
        if request.get("on_text") and message.get("content"):
            request["on_text"](message["content"])
        return response


class EngineManagementTests(unittest.TestCase):
    def state(self, responses, **kwargs):
        abort = Mock(wraps=Event())
        abort.wait.return_value = False
        return QueryState(ScriptedClient(responses), UsageLedger(), tools=[], abort=abort, **kwargs)

    def test_automatic_compaction_preserves_recent_round_and_current_question(self):
        events = []
        state = self.state([reply("目标与待办已总结"), reply("继续回答")],
                           context_limit=1200, summary_limit=100, keep_recent_turns=1, on_event=events.append)
        state.messages = history() + [{"role": "user", "content": "当前问题"}]
        recent = deepcopy(state.messages[-3:])
        self.assertEqual(query_loop(state), "继续回答")
        self.assertEqual(state.messages[-4:-1], recent)
        summary_request, normal_request = state.client.requests
        self.assertEqual(summary_request["tools"], [])
        self.assertEqual(summary_request["max_tokens"], 200)
        self.assertNotIn("当前问题", json.dumps(summary_request["messages"], ensure_ascii=False))
        self.assertLessEqual(context_size(normal_request["messages"], []), 1200)
        self.assertEqual(state.ledger.summary()["requests"], 2)
        self.assertEqual(state.ledger.summary()["total_tokens"], 30)
        self.assertEqual([event["text"] for event in events if event["type"] == "text"], ["继续回答"])

    def test_manual_compaction_works_below_automatic_threshold(self):
        state = self.state([reply("历史摘要")], keep_recent_turns=1)
        state.messages = history()
        original = deepcopy(state.messages)
        self.assertTrue(compact_history(state, force=True))
        self.assertLess(context_size(state.messages, []), context_size(original, []))
        self.assertEqual(state.messages[-2:], original[-2:])
        self.assertEqual(state.compaction_count, 1)

    def test_short_history_does_not_pay_for_a_summary(self):
        events = []
        state = self.state([], on_event=events.append)
        state.messages = history(2)
        self.assertFalse(compact_history(state, force=True))
        self.assertEqual(state.client.requests, [])
        self.assertEqual(events[-1]["type"], "compact_skipped")

    def test_overlong_complete_summary_is_truncated_to_budget_and_committed(self):
        state = self.state([reply("超" * 101)],
                           context_limit=1200, summary_limit=100, keep_recent_turns=1)
        state.messages = history()
        original = deepcopy(state.messages)
        self.assertTrue(compact_history(state, force=True))
        self.assertEqual(state.compaction_count, 1)
        self.assertLess(context_size(state.messages, []), context_size(original, []))
        summary = next(message for message in state.messages
                       if message.get("role") == "assistant"
                       and (message.get("content") or "").startswith("[历史对话摘要]"))
        self.assertEqual(len(summary["content"].removeprefix("[历史对话摘要]\n")), 100)
        self.assertEqual(len(state.client.requests), 1)

    def test_unfinished_model_summary_preserves_history_and_consumes_compaction_limit(self):
        state = self.state([reply("不完整摘要", reason="length")],
                           context_limit=1200, summary_limit=100, keep_recent_turns=1,
                           max_compactions=1)
        state.messages = history()
        original = deepcopy(state.messages)
        with self.assertRaisesRegex(ContextTooLong, "太长了，建议开个新会话"):
            compact_history(state, force=True)
        self.assertEqual(state.messages, original)
        self.assertEqual(state.compaction_count, 1)
        self.assertEqual(len(state.client.requests), 1)

    def test_summary_network_failure_keeps_original_history(self):
        state = self.state([APIError("无法生成摘要")], keep_recent_turns=1)
        state.messages = history()
        original = deepcopy(state.messages)
        with self.assertRaises(APIError):
            compact_history(state, force=True)
        self.assertEqual(state.messages, original)
        self.assertEqual(state.ledger.summary()["missing_usage_requests"], 1)

    def test_retained_round_alone_exceeds_budget_without_any_model_request(self):
        state = self.state([], context_limit=300, keep_recent_turns=1)
        state.messages = history(1) + [{"role": "user", "content": "继续"}]
        with self.assertRaises(ContextTooLong):
            query_loop(state)
        self.assertEqual(state.client.requests, [])

    def test_compaction_limit_is_shared_across_the_entire_tool_loop(self):
        state = self.state([
            reply("摘" * 180), reply(None, [tool_call("first")]),
            reply("摘要"), reply(None, [tool_call("second")]),
        ], context_limit=620, summary_limit=200, keep_recent_turns=0, max_compactions=2,
            tool_executor=lambda *args: {"message": "结果" * 60})
        state.messages = history() + [{"role": "user", "content": "当前问题"}]
        with self.assertRaises(ContextTooLong):
            query_loop(state)
        self.assertEqual(state.compaction_count, 2)
        self.assertEqual(len(state.client.requests), 4)
        self.assertEqual(state.ledger.summary()["requests"], 4)

    def test_transient_retries_wait_progressively_and_count_every_attempt(self):
        events = []
        state = self.state([APIError("暂时失败", retryable=True), APIError("暂时失败", retryable=True),
                            reply("已恢复")], on_event=events.append)
        state.messages.append({"role": "user", "content": "问题"})
        self.assertEqual(query_loop(state), "已恢复")
        self.assertEqual(state.abort.wait.call_args_list, [call(1), call(2)])
        self.assertEqual(state.ledger.summary()["requests"], 3)
        self.assertEqual(state.ledger.summary()["missing_usage_requests"], 2)
        self.assertEqual([event["attempt"] for event in events if event["type"] == "retry"], [1, 2])

    def test_retries_stop_after_three_retries_and_permanent_errors_stop_immediately(self):
        for retryable, count in [(True, 4), (False, 1)]:
            with self.subTest(retryable=retryable):
                state = self.state([APIError("失败", retryable=retryable) for _ in range(count)])
                with self.assertRaises(APIError):
                    query_loop(state)
                self.assertEqual(len(state.client.requests), count)
                self.assertEqual(state.ledger.summary()["requests"], count)
                self.assertEqual(state.abort.wait.call_args_list, [call(1), call(2), call(4)] if retryable else [])

    def test_exit_during_retry_wait_stops_without_another_request(self):
        state = self.state([APIError("网络中断", retryable=True)])
        state.abort.wait.return_value = True
        with self.assertRaises(QueryAborted):
            query_loop(state)
        self.assertEqual(len(state.client.requests), 1)

    def test_partial_retry_restarts_answer_without_duplicating_tool_execution(self):
        events = []
        executed = Mock(return_value={"message": "占位结果"})

        def interrupted(request):
            request["on_text"]("未完成草稿")
            raise APIError("连接中断", retryable=True)

        state = self.state([interrupted, reply(None, [tool_call()]), reply("完成")],
                           tool_executor=executed, on_event=events.append)
        state.messages.append({"role": "user", "content": "问题"})
        self.assertEqual(query_loop(state), "完成")
        executed.assert_called_once()
        self.assertNotIn("未完成草稿", json.dumps(state.messages, ensure_ascii=False))
        self.assertTrue(next(event for event in events if event["type"] == "retry")["partial"])

    def test_summary_and_retries_share_actual_request_limit(self):
        state = self.state([reply("摘要"), APIError("网络中断", retryable=True)],
                           keep_recent_turns=1, context_limit=1200, max_requests=2)
        state.messages = history() + [{"role": "user", "content": "继续"}]
        with self.assertRaises(APIError):
            query_loop(state)
        self.assertEqual(state.request_count, 2)
        self.assertEqual(state.ledger.summary()["requests"], 2)
        state.abort.wait.assert_not_called()

    def test_large_tool_result_is_truncated_before_next_request_and_display(self):
        events = []
        state = self.state([reply(None, [tool_call()]), reply("完成")], tool_result_limit=250,
                           tool_executor=lambda *args: {"message": "头" + "很长" * 1000 + "尾"},
                           on_event=events.append)
        query_loop(state)
        content = state.client.requests[1]["messages"][-1]["content"]
        self.assertLessEqual(len(content), 250)
        self.assertTrue(json.loads(content)["truncated"])
        displayed = next(event for event in events if event["type"] == "tool")["result"]["message"]
        self.assertIn("截断", displayed)
        self.assertLess(len(displayed), 100)


if __name__ == "__main__":
    unittest.main()
