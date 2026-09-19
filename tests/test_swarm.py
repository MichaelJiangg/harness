import json
from threading import Event
import unittest

from harness.engine import QueryState
from harness.orchestration import Handoff, SwarmRole, _parse_handoff, run_swarm
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


def role(name, system, tools=(), handoff_to=()):
    return {"name": name, "system": system, "tools": list(tools),
            "handoff_to": list(handoff_to)}


def handoff(next_role, summary, artifacts=None, feedback=None):
    return json.dumps({
        "next_role": next_role,
        "summary": summary,
        "artifacts": artifacts or [],
        "feedback": feedback,
    }, ensure_ascii=False)


class SwarmOrchestrationTests(unittest.TestCase):
    def parent(self, responses, max_requests=20):
        return QueryState(FakeClient(responses), UsageLedger(), tools=[],
                          turn=3, max_requests=max_requests)

    def run_case(self, responses, roles, *, max_rounds=10, max_requests=1):
        events = []
        state = self.parent(responses, max_requests=max_requests)
        state.on_event = events.append
        result = run_swarm(
            parent=state, description="团队任务", task="实现缓存模块",
            roles=roles, max_rounds=max_rounds,
            make_executor=lambda tools, abort: lambda *args: {"status": "success"},
        )
        return state, result, events

    def test_roles_follow_handoff_protocol_and_keep_isolated_contexts(self):
        roles = [
            role("Coder", "编写代码", handoff_to=["Reviewer", "Tester"]),
            role("Reviewer", "审查代码", handoff_to=["Coder", "Tester"]),
            role("Tester", "编写测试", handoff_to=[]),
        ]
        responses = [
            reply(handoff("Reviewer", "已创建 cache.py", ["cache.py"])),
            reply(handoff("Coder", "发现两个建议", ["cache.py"], "补充边界检查")),
            reply(handoff("Tester", "已按建议修改", ["cache.py"])),
            reply(handoff(None, "团队完成，5 个测试通过", ["test_cache.py"])),
        ]
        state, result, events = self.run_case(responses, roles)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "团队完成，5 个测试通过")
        self.assertEqual(result["rounds"], 4)
        self.assertEqual(result["artifacts"], ["test_cache.py"])
        event_types = [event["type"] for event in events
                       if event.get("agent") == "swarm" and event["type"] != "usage"]
        self.assertEqual(event_types, [
            "swarm_start", "role_start", "role_complete", "handoff",
            "role_start", "role_complete", "handoff",
            "role_start", "role_complete", "handoff",
            "role_start", "role_complete", "swarm_complete",
        ])
        requests = state.client.requests
        self.assertEqual(len(requests), 4)
        for request in requests:
            self.assertEqual([message["role"] for message in request["messages"]],
                             ["system", "user"])
            self.assertNotIn("实现缓存模块", request["messages"][0]["content"])
        self.assertIn("打回反馈：补充边界检查", requests[2]["messages"][1]["content"])
        self.assertIn("产物文件：cache.py", requests[2]["messages"][1]["content"])
        self.assertEqual(state.request_count, 0)
        self.assertEqual(state.ledger.summary()["requests"], 4)

    def test_swarm_has_independent_budget_and_role_cap(self):
        state = self.parent([
            reply(None, [tool_call("first", "read_file", '{"path":"a.py"}')]),
            reply(None, [tool_call("second", "read_file", '{"path":"b.py"}')]),
            reply(handoff(None, "不应到达")),
        ], max_requests=1)
        state.swarm_max_requests = 10
        state.swarm_max_role_requests = 2
        result = run_swarm(
            parent=state, description="独立预算", task="读取并完成",
            roles=[role("Coder", "coder", tools=["read_file"])],
            max_rounds=2,
            make_executor=lambda tools, abort: lambda *args: {"status": "success"},
        )
        self.assertEqual(result["code"], "swarm_role_request_limit")
        self.assertEqual(state.request_count, 0)
        self.assertEqual(state.ledger.summary()["requests"], 2)

    def test_max_rounds_stops_reviewer_coder_loop(self):
        roles = [
            role("Coder", "coder", handoff_to=["Reviewer"]),
            role("Reviewer", "reviewer", handoff_to=["Coder"]),
        ]
        responses = [
            reply(handoff("Reviewer", "第一版")),
            reply(handoff("Coder", "仍需修改", feedback="继续改")),
        ]
        state, result, events = self.run_case(responses, roles, max_rounds=2)
        self.assertEqual(result["code"], "swarm_max_rounds")
        self.assertEqual(state.client.requests[-1]["messages"][1]["content"],
                         "团队任务：实现缓存模块\n你当前的角色：Reviewer\n交接来自：Coder\n交接摘要：第一版")

    def test_invalid_handoff_is_reported_without_leaking_model_text(self):
        state, result, _ = self.run_case(
            [reply("不是 JSON，包含 private detail")],
            [role("Coder", "coder", handoff_to=[])],
        )
        self.assertEqual(result["code"], "swarm_invalid_handoff")
        self.assertNotIn("private detail", json.dumps(result, ensure_ascii=False))

    def test_parse_accepts_fenced_json_and_validates_fields(self):
        parsed, error = _parse_handoff('```json\n{"next_role": "Next", '
                                       '"summary": "完成", "artifacts": ["a.py"], '
                                       '"feedback": null}\n```')
        self.assertIsNone(error)
        self.assertEqual(parsed["next_role"], "Next")
        for text in ('{"summary": "完成"}', '{"next_role": 3, "summary": "完成"}',
                     '{"next_role": "Next", "summary": ""}'):
            self.assertIsNotNone(_parse_handoff(text)[1])

    def test_dataclasses_carry_role_and_handoff_data(self):
        role = SwarmRole("Coder", "system", ("read_file",), ("Reviewer",))
        handoff = Handoff("Coder", "Reviewer", "摘要", ("cache.py",), "意见")
        self.assertEqual(role.tools, ("read_file",))
        self.assertEqual(handoff.artifacts, ("cache.py",))
        self.assertEqual(handoff.feedback, "意见")


if __name__ == "__main__":
    unittest.main()
