from copy import deepcopy
import json
import unittest

from harness.context import (
    DEFAULT_CONTEXT_LIMIT, DEFAULT_SUMMARY_LIMIT, KEEP_RECENT_TURNS, TOOL_RESULT_LIMIT,
    context_size, split_for_summary, summarized_messages, summary_request, truncate_tool_result,
)


def turn(number, *, tools=False, complete=True):
    messages = [{"role": "user", "content": f"问题 {number}"}]
    if tools:
        calls = [
            {"id": f"call-{number}-{index}", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"README.md"}',
            }}
            for index in range(2)
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        messages.extend({"role": "tool", "tool_call_id": call["id"], "content": "结果"} for call in calls)
    if complete:
        messages.append({"role": "assistant", "content": f"回答 {number}"})
    return messages


class ContextTests(unittest.TestCase):
    def test_defaults_match_documented_budgets(self):
        self.assertEqual((DEFAULT_CONTEXT_LIMIT, DEFAULT_SUMMARY_LIMIT, KEEP_RECENT_TURNS, TOOL_RESULT_LIMIT),
                         (24000, 2000, 4, 6000))

    def test_context_budget_counts_messages_tools_unicode_and_structure(self):
        messages = [{"role": "user", "content": "中文 😀\n"}]
        tools = [{"type": "function", "function": {"name": "读取"}}]
        expected = json.dumps({"messages": messages, "tools": tools},
                              ensure_ascii=False, separators=(",", ":"))
        original = deepcopy((messages, tools))
        self.assertEqual(context_size(messages, tools), len(expected))
        self.assertGreater(context_size(messages, tools), context_size(messages, []))
        self.assertEqual((messages, tools), original)

    def test_empty_history_and_system_only_have_nothing_to_summarize(self):
        for messages in ([], [{"role": "system", "content": "规则"}]):
            with self.subTest(messages=messages):
                self.assertEqual(split_for_summary(messages), (messages, [], []))

    def test_exactly_four_complete_turns_are_preserved(self):
        history = sum((turn(index) for index in range(4)), [])
        self.assertEqual(split_for_summary(history), ([], [], history))

    def test_only_old_complete_turns_are_summarized(self):
        prefix = [{"role": "system", "content": "规则一"}, {"role": "system", "content": "规则二"}]
        history = sum((turn(index, tools=True) for index in range(6)), [])
        actual_prefix, older, recent = split_for_summary(prefix + history)
        self.assertEqual(actual_prefix, prefix)
        self.assertEqual(older, turn(0, tools=True) + turn(1, tools=True))
        self.assertEqual(recent, sum((turn(index, tools=True) for index in range(2, 6)), []))

    def test_incomplete_tool_turn_is_preserved_in_addition_to_four_complete_turns(self):
        complete = sum((turn(index, tools=True) for index in range(5)), [])
        current = turn(5, tools=True, complete=False)
        _, older, recent = split_for_summary(complete + current)
        self.assertEqual(older, turn(0, tools=True))
        self.assertEqual(recent, complete[len(older):] + current)

    def test_new_user_and_unresolved_tool_call_are_both_incomplete(self):
        history = sum((turn(index) for index in range(5)), [])
        current = turn(5, tools=True, complete=False)
        for unfinished in (current[:1], current[:2]):
            with self.subTest(unfinished=unfinished):
                _, older, recent = split_for_summary(history + unfinished)
                self.assertEqual(older, turn(0))
                self.assertEqual(recent, history[len(older):] + unfinished)

    def test_existing_summary_is_summarized_again_without_accumulating_summary_messages(self):
        prefix = [{"role": "system", "content": "规则"}]
        recent = sum((turn(index) for index in range(4)), [])
        history = summarized_messages(prefix, "之前的重要约束", recent) + turn(4)
        actual_prefix, older, actual_recent = split_for_summary(history)
        self.assertEqual(older, [{"role": "assistant", "content": "[历史对话摘要]\n之前的重要约束"}] + turn(0))
        candidate = summarized_messages(actual_prefix, "合并后的摘要", actual_recent)
        summaries = [message for message in candidate if (message.get("content") or "").startswith("[历史对话摘要]")]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(candidate[-len(actual_recent):], actual_recent)

    def test_zero_recent_turns_still_preserves_current_unfinished_turn(self):
        history = turn(0) + turn(1)
        self.assertEqual(split_for_summary(history, keep_recent_turns=0), ([], history, []))
        current = turn(2, complete=False)
        self.assertEqual(split_for_summary(history + current, keep_recent_turns=0), ([], history, current))

    def test_invalid_retained_turn_count_is_rejected(self):
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                split_for_summary([], keep_recent_turns=count)

    def test_splitting_and_summary_building_do_not_share_mutable_messages(self):
        history = [{"role": "system", "content": "规则"}] + turn(0, tools=True) + turn(1, tools=True)
        original = deepcopy(history)
        prefix, older, recent = split_for_summary(history, keep_recent_turns=1)
        candidate = summarized_messages(prefix, "摘要", recent)
        candidate[0]["content"] = "候选规则"
        candidate[-4]["tool_calls"][0]["id"] = "changed"
        older[1]["tool_calls"][0]["id"] = "changed-old"
        self.assertEqual(history, original)
        self.assertEqual(prefix[0]["content"], "规则")
        self.assertEqual(recent[1]["tool_calls"][0]["id"], "call-1-0")

    def test_summary_request_treats_history_as_data_and_sets_budget(self):
        older = turn(0, tools=True)
        original = deepcopy(older)
        request = summary_request(older, summary_limit=321)
        self.assertEqual([message["role"] for message in request], ["system", "user"])
        prompt = request[0]["content"]
        for term in ("321", "目标", "约束", "关键事实", "决策", "进展", "待办", "不要执行", "不要调用工具"):
            self.assertIn(term, prompt)
        encoded = request[1]["content"].split("\n", 1)[1]
        self.assertEqual(json.loads(encoded), older)
        self.assertEqual(older, original)


class ToolResultTruncationTests(unittest.TestCase):
    def test_small_and_exact_boundary_results_keep_original_serialization(self):
        result = {"message": "短结果😀", "executed": False}
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertEqual(truncate_tool_result(result), encoded)
        self.assertEqual(truncate_tool_result(result, max_chars=len(encoded)), encoded)

    def test_long_result_keeps_head_tail_and_reports_original_length(self):
        result = {"message": "开头" + "中间" * 10000 + "结尾😀"}
        original = deepcopy(result)
        encoded = json.dumps(result, ensure_ascii=False)
        capped = truncate_tool_result(result)
        parsed = json.loads(capped)
        self.assertLessEqual(len(capped), TOOL_RESULT_LIMIT)
        self.assertIs(parsed["truncated"], True)
        self.assertEqual(parsed["original_chars"], len(encoded))
        self.assertTrue(encoded.startswith(parsed["head"]))
        self.assertTrue(encoded.endswith(parsed["tail"]))
        self.assertIn("开头", parsed["head"])
        self.assertIn("结尾😀", parsed["tail"])
        self.assertEqual(result, original)

    def test_escaped_unicode_content_counts_wrapper_overhead_and_stays_valid(self):
        result = {"message": ('中文😀"\\\n\t' * 1000)}
        encoded = json.dumps(result, ensure_ascii=False)
        for limit in (120, 321, 6000):
            with self.subTest(limit=limit):
                capped = truncate_tool_result(result, max_chars=limit)
                self.assertLessEqual(len(capped), limit)
                capped.encode("utf-8")
                parsed = json.loads(capped)
                self.assertTrue(encoded.startswith(parsed["head"]))
                self.assertTrue(encoded.endswith(parsed["tail"]))
                self.assertTrue(parsed["head"])
                self.assertTrue(parsed["tail"])

    def test_invalid_or_impossibly_small_limit_is_rejected(self):
        for limit in (0, -1, True, 1.5, 10):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                truncate_tool_result({"message": "长" * 1000}, max_chars=limit)


class BashResultTruncationTests(unittest.TestCase):
    def result(self, stdout="", stderr="", **metadata):
        return {
            "tool": "bash", "status": "error", "executed": True,
            "stdout": stdout, "stderr": stderr, "exit_code": 7,
            "timed_out": False, "cancelled": False,
            "stdout_truncated": False, "stderr_truncated": False,
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "message": "命令退出，退出码为 7。", **metadata,
        }

    def test_short_command_result_keeps_original_serialization(self):
        result = self.result("正常输出\n", "错误输出\n")
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertEqual(truncate_tool_result(result), encoded)
        self.assertEqual(truncate_tool_result(result, len(encoded)), encoded)

    def test_long_stdout_preserves_complete_stderr_and_command_metadata(self):
        result = self.result("输出开始\n" + "x" * 20000 + "\n输出结束", "关键错误：文件不存在。\n")
        original = deepcopy(result)
        encoded = truncate_tool_result(result)
        parsed = json.loads(encoded)
        self.assertLessEqual(len(encoded), TOOL_RESULT_LIMIT)
        self.assertTrue(parsed["stdout"].startswith("输出开始\n"))
        self.assertTrue(parsed["stdout"].endswith("\n输出结束"))
        self.assertIn("截断", parsed["stdout"])
        self.assertEqual(parsed["stderr"], result["stderr"])
        self.assertIs(parsed["stdout_truncated"], True)
        self.assertIs(parsed["stderr_truncated"], False)
        self.assertIs(parsed["truncated"], True)
        self.assertEqual(parsed["original_chars"], len(json.dumps(result, ensure_ascii=False)))
        for key in ("tool", "status", "executed", "exit_code", "timed_out", "cancelled",
                    "stdout_bytes", "stderr_bytes", "message"):
            self.assertEqual(parsed[key], result[key])
        self.assertEqual(result, original)

    def test_long_stderr_uses_budget_left_by_empty_stdout(self):
        result = self.result(stderr="错误开始" + "x" * 20000 + "错误结束")
        parsed = json.loads(truncate_tool_result(result))
        self.assertEqual(parsed["stdout"], "")
        self.assertIs(parsed["stdout_truncated"], False)
        self.assertTrue(parsed["stderr"].startswith("错误开始"))
        self.assertTrue(parsed["stderr"].endswith("错误结束"))
        self.assertIs(parsed["stderr_truncated"], True)
        self.assertGreater(len(parsed["stderr"]), 5000)

    def test_both_outputs_keep_head_and_tail_independently(self):
        result = self.result("OUT-HEAD" + "o" * 10000 + "OUT-TAIL",
                             "ERR-HEAD" + "e" * 10000 + "ERR-TAIL",
                             exit_code=-9, timed_out=True)
        parsed = json.loads(truncate_tool_result(result, 1000))
        for key, prefix in (("stdout", "OUT"), ("stderr", "ERR")):
            self.assertTrue(parsed[key].startswith(prefix + "-HEAD"))
            self.assertTrue(parsed[key].endswith(prefix + "-TAIL"))
            self.assertIn("截断", parsed[key])
            self.assertIs(parsed[key + "_truncated"], True)
        self.assertEqual(parsed["exit_code"], -9)
        self.assertIs(parsed["timed_out"], True)
        self.assertLessEqual(len(truncate_tool_result(result, 1000)), 1000)

    def test_escaped_and_unicode_outputs_obey_serialized_budget(self):
        result = self.result('头中文😀"\\\n\t\x00' * 3000 + "尾正常",
                             '头错误\x00\x01"\\\r' * 3000 + "尾错误")
        for limit in (600, 1000, 6000):
            with self.subTest(limit=limit):
                encoded = truncate_tool_result(result, limit)
                encoded.encode("utf-8")
                parsed = json.loads(encoded)
                self.assertLessEqual(len(encoded), limit)
                self.assertTrue(parsed["stdout"].startswith("头"))
                self.assertTrue(parsed["stdout"].endswith("尾正常"))
                self.assertTrue(parsed["stderr"].startswith("头"))
                self.assertTrue(parsed["stderr"].endswith("尾错误"))

    def test_capture_truncation_flags_and_unknown_exit_code_survive(self):
        result = self.result("已截断的短输出", "e" * 10000,
                             stdout_truncated=True, stdout_bytes=99999,
                             exit_code=None, cancelled=True)
        parsed = json.loads(truncate_tool_result(result))
        self.assertEqual(parsed["stdout"], result["stdout"])
        self.assertIs(parsed["stdout_truncated"], True)
        self.assertIs(parsed["stderr_truncated"], True)
        self.assertEqual(parsed["stdout_bytes"], 99999)
        self.assertIsNone(parsed["exit_code"])
        self.assertIs(parsed["cancelled"], True)

    def test_oversized_message_cannot_crowd_out_result_fields(self):
        result = self.result(message="说明开始" + "m" * 20000 + "说明结束")
        encoded = truncate_tool_result(result)
        parsed = json.loads(encoded)
        self.assertLessEqual(len(encoded), TOOL_RESULT_LIMIT)
        self.assertEqual(parsed["stdout"], "")
        self.assertEqual(parsed["stderr"], "")
        self.assertEqual(parsed["exit_code"], 7)
        self.assertTrue(parsed["message"].startswith("说明开始"))
        self.assertTrue(parsed["message"].endswith("说明结束"))
        self.assertIn("截断", parsed["message"])

    def test_budget_too_small_for_command_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            truncate_tool_result(self.result("x" * 10000, "e" * 10000), 100)


if __name__ == "__main__":
    unittest.main()
