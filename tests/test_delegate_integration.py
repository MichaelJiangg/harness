import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from harness.orchestration import get_current_query
from harness.audit import PermissionAuditLog
from harness.client import APIError
from harness.context import context_size
from harness.engine import QueryAborted, QueryState, compact_history, query_loop
from harness.permissions import PermissionPolicy, SessionPermissionCache
from harness.tools import create_tool_executor, get_tool_definitions
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


def call(call_id, name, **arguments):
    return tool_call(call_id, name, json.dumps(arguments, ensure_ascii=False))


def delegate(call_id="delegate-1", task="读取 main.py 并汇总", description="分析"):
    return call(call_id, "delegate", description=description, task=task)


def results(request):
    return {message["tool_call_id"]: json.loads(message["content"])
            for message in request["messages"] if message["role"] == "tool"}


class DelegateIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.workspace.joinpath("main.py").write_text("needle = 1\n", encoding="utf-8")

    def state(self, responses, *, permissions=None, confirm=None, session_cache=None,
              audit=None, client=None, **options):
        abort = options.setdefault("abort", Event())
        executor = create_tool_executor(
            self.workspace, confirm=confirm, abort=abort, session_cache=session_cache,
            audit=audit, permissions=permissions or PermissionPolicy(allow=["delegate"]),
        )
        state = QueryState(client or FakeClient(responses), UsageLedger(),
                           tool_executor=executor, **options)
        state.messages.extend([
            {"role": "user", "content": "父会话专属历史"},
            {"role": "assistant", "content": "父会话之前的回答"},
            {"role": "user", "content": "请委托子任务，再给我总结。"},
        ])
        return state

    def test_child_reads_and_searches_with_isolated_history_and_shared_accounting(self):
        task = "读取 main.py 并搜索 needle"
        child_calls = [call("read-1", "read_file", path="main.py"),
                       call("grep-1", "grep", keyword="needle", glob="*.py")]
        state = self.state([
            reply(None, [delegate(task=task)]),
            reply(None, child_calls),
            reply("子任务报告：main.py 第一行定义 needle。"),
            reply("主任务总结完成。"),
        ], model="test-model", turn=7)
        self.assertEqual(query_loop(state), "主任务总结完成。")
        requests = state.client.requests
        child_messages = requests[1]["messages"]
        self.assertEqual([message["role"] for message in child_messages], ["system", "user"])
        self.assertEqual(child_messages[1]["content"], task)
        self.assertNotIn("父会话专属历史", json.dumps(child_messages, ensure_ascii=False))
        child_results = results(requests[2])
        self.assertEqual(child_results["read-1"]["content"], "needle = 1\n")
        self.assertEqual(child_results["grep-1"]["matches"][0]["line_number"], 1)
        parent_results = results(requests[3])
        self.assertEqual(set(parent_results), {"delegate-1"})
        self.assertEqual(parent_results["delegate-1"]["content"], "子任务报告：main.py 第一行定义 needle。")
        self.assertEqual(parent_results["delegate-1"]["status"], "success")
        self.assertTrue(parent_results["delegate-1"]["executed"])
        self.assertEqual({request["model"] for request in requests}, {"test-model"})
        self.assertEqual(state.request_count, 4)
        self.assertEqual(state.ledger.summary()["requests"], 4)
        self.assertEqual(state.ledger.summary()["total_tokens"], 480)
        self.assertEqual([record["turn"] for record in state.ledger.records], [7] * 4)
        self.assertIsNone(get_current_query())

    def test_child_cannot_delegate_again_or_call_tools_hidden_by_parent(self):
        tools = [tool for tool in get_tool_definitions()
                 if tool["function"]["name"] in {"delegate", "read_file"}]
        state = self.state([
            reply(None, [delegate()]),
            reply(None, [delegate("nested", "不要递归"),
                         call("hidden", "grep", keyword="needle")]),
            reply("所请求工具不可用。"),
            reply("主任务结束。"),
        ], tools=tools)
        self.assertEqual(query_loop(state), "主任务结束。")
        self.assertEqual([tool["function"]["name"] for tool in state.client.requests[1]["tools"]],
                         ["read_file"])
        denied = results(state.client.requests[2])
        for call_id in ("nested", "hidden"):
            self.assertEqual(denied[call_id]["code"], "unknown_tool")
            self.assertFalse(denied[call_id]["executed"])
        self.assertEqual(len(state.client.requests), 4)

    def test_child_inherits_parent_denials_for_read_write_and_bash(self):
        confirm = Mock(return_value=True)
        policy = PermissionPolicy(allow=["delegate"], deny=["read_file", "write_file", "bash"])
        state = self.state([
            reply(None, [delegate()]),
            reply(None, [call("denied-read", "read_file", path="main.py"),
                         call("denied-write", "write_file", path="created/note.txt", content="不得写入"),
                         call("denied-bash", "bash", command="printf harmless")]),
            reply("子任务受权限限制，未执行操作。"),
            reply("已说明权限限制。"),
        ], permissions=policy, confirm=confirm)
        with patch("subprocess.Popen") as popen:
            self.assertEqual(query_loop(state), "已说明权限限制。")
            popen.assert_not_called()
        for result in results(state.client.requests[2]).values():
            self.assertEqual(result["code"], "permission_denied")
            self.assertFalse(result["executed"])
        confirm.assert_not_called()
        self.assertFalse(self.workspace.joinpath("created").exists())
        self.assertEqual(self.workspace.joinpath("main.py").read_text(encoding="utf-8"), "needle = 1\n")

    def test_child_write_uses_parent_confirmation_cache_and_audit_session(self):
        confirm = Mock(return_value=True)
        cache = SessionPermissionCache()
        audit = PermissionAuditLog(self.workspace)
        state = self.state([
            reply(None, [delegate(task="在 notes/child.txt 写入报告")]),
            reply(None, [call("child-write", "write_file", path="notes/child.txt", content="子报告")]),
            reply("已写入子报告。"),
            reply(None, [call("parent-write", "write_file", path="notes/parent.txt", content="主报告")]),
            reply("两个报告已完成。"),
        ], confirm=confirm, session_cache=cache, audit=audit)
        self.assertEqual(query_loop(state), "两个报告已完成。")
        confirm.assert_called_once_with("write_file", {"path": "notes/child.txt", "content": "子报告"},
                                        self.workspace)
        self.assertEqual(self.workspace.joinpath("notes/child.txt").read_text(encoding="utf-8"), "子报告")
        self.assertEqual(self.workspace.joinpath("notes/parent.txt").read_text(encoding="utf-8"), "主报告")
        records = [json.loads(line) for line in self.workspace.joinpath(".harness/permission.log")
                   .read_text(encoding="utf-8").splitlines()]
        self.assertEqual({record["session_id"] for record in records}, {audit.session_id})
        self.assertEqual({record["tool"] for record in records}, {"delegate", "write_file"})
        self.assertTrue(any(record["confirmation"] == "remembered" for record in records))

    def test_child_write_rejection_is_returned_to_child_without_creating_directories(self):
        confirm = Mock(return_value=False)
        state = self.state([
            reply(None, [delegate()]),
            reply(None, [call("write-rejected", "write_file", path="new/note.txt", content="报告")]),
            reply("用户拒绝了写入，文件未创建。"),
            reply("没有修改文件。"),
        ], confirm=confirm)
        self.assertEqual(query_loop(state), "没有修改文件。")
        rejected = results(state.client.requests[2])["write-rejected"]
        self.assertEqual(rejected["code"], "confirmation_denied")
        self.assertFalse(rejected["executed"])
        self.assertFalse(self.workspace.joinpath("new").exists())
        confirm.assert_called_once()

    def test_sibling_children_do_not_share_messages_but_share_total_usage(self):
        state = self.state([
            reply(None, [delegate("first", "第一子任务的独立说明"),
                         delegate("second", "第二子任务的独立说明")]),
            reply("第一份报告"),
            reply("第二份报告"),
            reply("两份报告汇总。"),
        ])
        self.assertEqual(query_loop(state), "两份报告汇总。")
        first, second = state.client.requests[1:3]
        self.assertEqual(first["messages"][-1]["content"], "第一子任务的独立说明")
        self.assertEqual(second["messages"][-1]["content"], "第二子任务的独立说明")
        self.assertNotIn("第一", json.dumps(second["messages"], ensure_ascii=False))
        self.assertEqual(len(second["messages"]), 2)
        parent_results = results(state.client.requests[3])
        self.assertEqual(parent_results["first"]["content"], "第一份报告")
        self.assertEqual(parent_results["second"]["content"], "第二份报告")
        self.assertEqual(state.ledger.summary()["requests"], 4)
        self.assertEqual(state.request_count, 4)

    def test_children_reserve_one_request_for_parent_when_shared_budget_runs_out(self):
        for budget in (2, 3):
            with self.subTest(budget=budget):
                responses = [reply(None, [delegate()])]
                if budget == 3:
                    responses.append(reply(None, [call("read-before-limit", "read_file", path="main.py")]))
                responses.append(reply("子任务额度不足，主任务报告限制。"))
                state = self.state(responses, max_requests=budget)
                self.assertEqual(query_loop(state), "子任务额度不足，主任务报告限制。")
                result = results(state.client.requests[-1])["delegate-1"]
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["code"], "delegate_request_limit")
                self.assertEqual(len(state.client.requests), budget)
                self.assertEqual(state.request_count, budget)
                self.assertEqual(state.ledger.summary()["requests"], budget)

    def test_child_retries_consume_shared_budget_and_are_billed_once(self):
        state = self.state([
            reply(None, [delegate()]),
            APIError("暂时失败", retryable=True),
            reply("重试后得到子报告。"),
            reply("主任务完成。"),
        ], max_requests=4, retry_initial_delay=0)
        self.assertEqual(query_loop(state), "主任务完成。")
        self.assertEqual(state.client.requests[1]["messages"], state.client.requests[2]["messages"])
        self.assertEqual(results(state.client.requests[3])["delegate-1"]["content"], "重试后得到子报告。")
        self.assertEqual(state.request_count, 4)
        self.assertEqual(state.ledger.summary()["requests"], 4)
        self.assertEqual(state.ledger.summary()["missing_usage_requests"], 1)
        self.assertEqual(state.ledger.summary()["total_tokens"], 360)

    def test_child_api_failure_returns_sanitized_error_and_parent_can_continue(self):
        state = self.state([
            reply(None, [delegate()]),
            APIError("private-child-client-detail"),
            reply("子任务失败，主任务说明无法取得报告。"),
        ])
        self.assertEqual(query_loop(state), "子任务失败，主任务说明无法取得报告。")
        result = results(state.client.requests[2])["delegate-1"]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "delegate_failed")
        self.assertNotIn("private-child-client-detail", json.dumps(result))
        self.assertEqual(state.ledger.summary()["requests"], 3)
        self.assertEqual(state.ledger.summary()["missing_usage_requests"], 1)
        self.assertIsNone(get_current_query())

    def test_cancellation_in_child_stops_parent_and_does_not_execute_child_tools(self):
        abort = Event()

        class CancellingClient(FakeClient):
            def complete(self, **request):
                response = super().complete(**request)
                if len(self.requests) == 2:
                    abort.set()
                return response

        client = CancellingClient([
            reply(None, [delegate()]),
            reply(None, [call("cancelled-write", "write_file", path="cancelled/note.txt", content="不得写入")]),
        ])
        confirm = Mock(return_value=True)
        state = self.state([], client=client, abort=abort, confirm=confirm)
        with self.assertRaises(QueryAborted):
            query_loop(state)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(state.request_count, 2)
        self.assertEqual(state.ledger.summary()["requests"], 2)
        self.assertFalse(self.workspace.joinpath("cancelled").exists())
        confirm.assert_not_called()
        self.assertIsNone(get_current_query())

    def test_parent_deny_of_delegate_starts_no_child_query(self):
        confirm = Mock(return_value=True)
        state = self.state([
            reply(None, [delegate()]), reply("委托功能被禁止，未启动子任务。"),
        ], permissions=PermissionPolicy(deny=["delegate"]), confirm=confirm)
        self.assertEqual(query_loop(state), "委托功能被禁止，未启动子任务。")
        denied = results(state.client.requests[1])["delegate-1"]
        self.assertEqual(denied["code"], "permission_denied")
        self.assertFalse(denied["executed"])
        self.assertEqual(len(state.client.requests), 2)
        confirm.assert_not_called()

    def large_read_batches(self):
        for name, content in (("old.py", "old_batch_marker\n" + "甲" * 4000),
                              ("recent_a.py", "recent_a_marker\n" + "乙" * 1500),
                              ("recent_b.py", "recent_b_marker\n" + "丙" * 1500)):
            self.workspace.joinpath(name).write_text(content, encoding="utf-8")
        return [call("old-read", "read_file", path="old.py")], [
            call("recent-a", "read_file", path="recent_a.py"),
            call("recent-b", "read_file", path="recent_b.py"),
        ]

    def test_child_compacts_older_tool_batches_and_keeps_latest_complete_batch(self):
        older, recent = self.large_read_batches()
        task = "阅读 old.py、recent_a.py 和 recent_b.py，比较三份内容。"
        events = []
        tools = [tool for tool in get_tool_definitions()
                 if tool["function"]["name"] in {"delegate", "read_file"}]
        state = self.state([
            reply(None, [delegate(task=task)]),
            reply(None, older),
            reply(None, recent),
            reply("old.py 已读取，主要内容为甲；还需合并最近两份读取结果。"),
            reply("三份内容已经比较，分别以甲、乙、丙为主。"),
            reply("主任务整理完成。"),
        ], tools=tools, context_limit=7000, summary_limit=200, on_event=events.append)
        self.assertEqual(query_loop(state), "主任务整理完成。")
        requests = state.client.requests
        self.assertEqual(len(requests), 6)
        summary = requests[3]
        self.assertEqual(summary["tools"], [])
        self.assertIsNone(summary["on_text"])
        source = json.dumps(summary["messages"], ensure_ascii=False)
        self.assertIn("old_batch_marker", source)
        self.assertNotIn("recent_a_marker", source)
        self.assertNotIn("recent_b_marker", source)
        self.assertNotIn("父会话专属历史", source)
        resumed = requests[4]["messages"]
        self.assertEqual(resumed[0], requests[1]["messages"][0])
        self.assertEqual([message["content"] for message in resumed if message["role"] == "user"], [task])
        self.assertIn("[历史对话摘要]", json.dumps(resumed, ensure_ascii=False))
        self.assertEqual(resumed[-3]["tool_calls"], recent)
        self.assertEqual([message["tool_call_id"] for message in resumed[-2:]], ["recent-a", "recent-b"])
        self.assertEqual(set(results(requests[4])), {"recent-a", "recent-b"})
        self.assertEqual(results(requests[5])["delegate-1"]["content"], "三份内容已经比较，分别以甲、乙、丙为主。")
        self.assertEqual(state.request_count, 6)
        self.assertEqual(state.ledger.summary()["requests"], 6)
        self.assertEqual(state.ledger.summary()["total_tokens"], 720)
        self.assertEqual(len([event for event in events if event["type"] == "compact_done"]), 1)

    def test_failed_child_compaction_stops_further_delegation_until_next_user_query(self):
        older, recent = self.large_read_batches()
        invalid_summary = "无效摘要"
        events = []
        tools = [tool for tool in get_tool_definitions()
                 if tool["function"]["name"] in {"delegate", "read_file"}]
        state = self.state([
            reply(None, [delegate("first"), delegate("same-batch", "同一批再试一次")]),
            reply(None, older),
            reply(None, recent),
            reply(invalid_summary, reason="length"),
            reply(invalid_summary, reason="length"),
            reply(None, [delegate("later", "下一轮再试一次")]),
            reply("子任务过长，建议开新会话。"),
        ], tools=tools, context_limit=7000, summary_limit=200, max_compactions=2, on_event=events.append)
        with patch("harness.engine.compact_history", wraps=compact_history) as compact:
            self.assertEqual(query_loop(state), "子任务过长，建议开新会话。")
        self.assertEqual(len(state.client.requests), 7)
        summaries = [request for request in state.client.requests if not request["tools"]]
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["messages"], summaries[1]["messages"])
        child = compact.call_args.args[0]
        self.assertEqual(child.compaction_count, 2)
        self.assertEqual({message["tool_call_id"] for message in child.messages if message["role"] == "tool"},
                         {"old-read", "recent-a", "recent-b"})
        self.assertNotIn(invalid_summary, json.dumps(child.messages, ensure_ascii=False))
        self.assertNotIn("[历史对话摘要]", json.dumps(child.messages, ensure_ascii=False))
        for request_index, call_id in ((5, "first"), (5, "same-batch"), (6, "later")):
            result = results(state.client.requests[request_index])[call_id]
            self.assertEqual(result["code"], "delegate_context_too_long")
            self.assertEqual(result["status"], "error")
        self.assertEqual(len([event for event in events if event["type"] == "delegate_start"]), 1)
        self.assertEqual(state.delegation_stop_code, "delegate_context_too_long")
        self.assertEqual(state.request_count, 7)
        self.assertEqual(state.ledger.summary()["total_tokens"], 840)

        state.client.responses.extend([
            reply(None, [delegate("new-turn", "新任务只需简短分析")]),
            reply("新任务的简短报告。"),
            reply("新一轮任务完成。"),
        ])
        state.messages.append({"role": "user", "content": "缩小范围，重新委托一个简短任务。"})
        state.turn += 1
        self.assertEqual(query_loop(state), "新一轮任务完成。")
        self.assertEqual(state.delegation_stop_code, "")
        self.assertEqual(state.request_count, 3)
        self.assertEqual(state.ledger.summary()["requests"], 10)
        self.assertEqual(len(state.client.requests[8]["messages"]), 2)
        self.assertEqual(results(state.client.requests[9])["new-turn"]["content"], "新任务的简短报告。")
        self.assertEqual(len([event for event in events if event["type"] == "delegate_start"]), 2)

    def test_child_recompacts_valid_summary_when_first_reduction_still_exceeds_budget(self):
        older, recent = self.large_read_batches()
        first_summary = "第一轮已提取旧文件信息。" + "摘要" * 750
        second_summary = "旧文件已读，主要内容为甲。"
        tools = [tool for tool in get_tool_definitions()
                 if tool["function"]["name"] in {"delegate", "read_file"}]
        state = self.state([
            reply(None, [delegate()]),
            reply(None, older),
            reply(None, recent),
            reply(first_summary),
            reply(second_summary),
            reply("第二次压缩后完成比较。"),
            reply("主任务整理完成。"),
        ], tools=tools, context_limit=6000, summary_limit=1600, max_compactions=2)
        with patch("harness.engine.compact_history", wraps=compact_history) as compact:
            self.assertEqual(query_loop(state), "主任务整理完成。")
        requests = state.client.requests
        self.assertEqual(len(requests), 7)
        self.assertEqual(requests[3]["tools"], [])
        self.assertEqual(requests[4]["tools"], [])
        second_source = json.dumps(requests[4]["messages"], ensure_ascii=False)
        self.assertIn(first_summary, second_source)
        self.assertNotIn("甲" * 4000, second_source)
        self.assertNotIn("recent_a_marker", second_source)
        self.assertNotIn("recent_b_marker", second_source)
        resumed = requests[5]["messages"]
        self.assertEqual(resumed[-3]["tool_calls"], recent)
        self.assertEqual([message["tool_call_id"] for message in resumed[-2:]], ["recent-a", "recent-b"])
        self.assertLessEqual(context_size(resumed, requests[5]["tools"]), state.context_limit)
        first_candidate = [
            {**message, "content": "[历史对话摘要]\n" + first_summary}
            if message.get("content") == "[历史对话摘要]\n" + second_summary else message
            for message in resumed
        ]
        self.assertGreater(context_size(first_candidate, requests[5]["tools"]), state.context_limit)
        self.assertEqual(compact.call_args.args[0].compaction_count, 2)
        self.assertEqual(results(requests[6])["delegate-1"]["content"], "第二次压缩后完成比较。")
        self.assertEqual(state.request_count, 7)
        self.assertEqual(state.ledger.summary()["requests"], 7)
        self.assertEqual(state.ledger.summary()["total_tokens"], 840)

    def test_child_stage_compacts_more_than_two_growing_tool_batches(self):
        events = []
        tools = [tool for tool in get_tool_definitions()
                 if tool["function"]["name"] in {"delegate", "read_file"}]
        for index in range(5):
            self.workspace.joinpath(f"part-{index}.py").write_text(
                f"part_{index} = 1\n" + f"文件内容 {index}\n" * 1200,
                encoding="utf-8",
            )
        reads = [
            reply(None, [call(f"read-{index}", "read_file", path=f"part-{index}.py")])
            for index in range(5)
        ]
        summaries = [reply("已读取并记录此前文件。") for _ in range(4)]
        responses = [reply(None, [delegate(task="依次读取五份文件并汇总")])]
        for index in range(5):
            responses.append(reads[index])
            if index:
                responses.append(summaries[index - 1])
        responses.extend([reply("五份文件已经汇总。"), reply("主任务整理完成。")])
        state = self.state(responses, tools=tools, context_limit=6000,
                           summary_limit=300, max_compactions=6, on_event=events.append)
        self.assertEqual(query_loop(state), "主任务整理完成。")
        compacted = [event for event in events
                     if event["type"] == "compact_done" and event.get("agent") == "delegate"]
        self.assertEqual(len(compacted), 4)
        self.assertEqual(len(state.client.requests), 12)
        self.assertEqual(state.ledger.summary()["requests"], 12)

    def test_delegate_request_limit_stop_is_cleared_for_next_user_query(self):
        state = self.state([
            reply(None, [delegate("budget-first"), delegate("budget-second")]),
            reply(None, [call("budget-read", "read_file", path="main.py")]),
            reply("本轮请求额度耗尽，停止继续委托。"),
        ], max_requests=3)
        self.assertEqual(query_loop(state), "本轮请求额度耗尽，停止继续委托。")
        self.assertEqual(state.delegation_stop_code, "delegate_request_limit")
        self.assertEqual(len(state.client.requests), 3)
        for call_id in ("budget-first", "budget-second"):
            self.assertEqual(results(state.client.requests[2])[call_id]["code"], "delegate_request_limit")
        state.client.responses.extend([
            reply(None, [delegate("fresh-budget", "只提供一段简短报告")]),
            reply("新预算下已完成子任务。"),
            reply("主任务完成。"),
        ])
        state.messages.append({"role": "user", "content": "请开始一个更短的新任务。"})
        state.turn += 1
        self.assertEqual(query_loop(state), "主任务完成。")
        self.assertEqual(state.delegation_stop_code, "")
        self.assertEqual(state.request_count, 3)
        self.assertEqual(state.ledger.summary()["requests"], 6)
        self.assertEqual(results(state.client.requests[5])["fresh-budget"]["content"], "新预算下已完成子任务。")


if __name__ == "__main__":
    unittest.main()
