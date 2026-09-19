import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from harness.engine import QueryState, query_loop
from harness.permissions import PermissionPolicy
from harness.tools import create_tool_executor
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


def handoff(next_role, summary, artifacts=None, feedback=None):
    return json.dumps({
        "next_role": next_role,
        "summary": summary,
        "artifacts": artifacts or [],
        "feedback": feedback,
    }, ensure_ascii=False)


class SwarmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def test_main_query_runs_swarm_roles_and_returns_final_report(self):
        confirm = Mock(return_value=True)
        events = []
        responses = [
            reply(None, [tool_call("swarm-1", "swarm", json.dumps({
                "description": "实现缓存模块",
                "task": "团队实现缓存模块：编写、审查并测试。",
            }, ensure_ascii=False))]),
            reply(handoff("Reviewer", "已创建 cache.py", ["cache.py"])),
            reply(handoff("Coder", "需要补充边界检查", ["cache.py"], "补充边界检查")),
            reply(handoff("Tester", "已按建议修改", ["cache.py"])),
            reply(handoff(None, "团队完成，5 个测试通过", ["test_cache.py"])),
            reply("主 AI 已整理团队协作结果。"),
        ]
        state = QueryState(
            FakeClient(responses), UsageLedger(), turn=2,
            tool_executor=create_tool_executor(
                self.root, confirm=confirm,
                permissions=PermissionPolicy(allow=["swarm"]),
            ),
            on_event=events.append,
        )
        state.messages.append({"role": "user", "content": "用团队模式实现缓存模块"})
        self.assertEqual(query_loop(state), "主 AI 已整理团队协作结果。")

        requests = state.client.requests
        self.assertEqual(len(requests), 6)
        role_requests = requests[1:5]
        self.assertEqual([
            [item["function"]["name"] for item in request["tools"]]
            for request in role_requests
        ], [
            ["grep", "read_file", "write_file"],
            ["grep", "read_file"],
            ["grep", "read_file", "write_file"],
            ["bash", "grep", "read_file", "run_verify", "write_file"],
        ])
        for request in role_requests:
            source = json.dumps(request["messages"], ensure_ascii=False)
            self.assertNotIn("用团队模式实现缓存模块", source)
            self.assertNotIn("主 AI 已整理", source)
        self.assertIn("打回反馈：补充边界检查",
                      role_requests[2]["messages"][1]["content"])
        for request in role_requests:
            for item in request["tools"]:
                self.assertNotIn(item["function"]["name"],
                                 {"delegate", "background_submit", "background_check", "swarm"})
        self.assertEqual(state.ledger.summary()["requests"], 6)
        self.assertEqual(state.request_count, 2)
        self.assertTrue(any(event["type"] == "swarm_complete" for event in events))
        confirm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
