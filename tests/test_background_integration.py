import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import Mock

from harness.background import BackgroundManager, COMPLETED
from harness.engine import QueryState, query_loop
from harness.permissions import PermissionPolicy
from harness.tools import create_tool_executor
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


def background_call(call_id, *, command=None, task=None):
    arguments = {"description": "运行测试" if command else "后台分析"}
    if command is not None:
        arguments["command"] = command
    if task is not None:
        arguments["task"] = task
    return tool_call(call_id, "background_submit", json.dumps(arguments, ensure_ascii=False))


class BackgroundIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.manager = BackgroundManager(max_concurrent=2, default_timeout=2)
        self.addCleanup(self.manager.shutdown)

    def executor(self, on_event=None):
        return create_tool_executor(
            self.root, confirm=Mock(return_value=True),
            permissions=PermissionPolicy(allow=["background_submit"]),
            background_manager=self.manager,
        )

    def wait_completed(self, task_id):
        deadline = monotonic() + 1
        while monotonic() < deadline:
            task = self.manager.check(task_id)
            if task["status"] == COMPLETED:
                return task
            sleep(0.005)
        return self.manager.check(task_id)

    def test_main_query_continues_then_next_turn_checks_background_result(self):
        events = []
        responses = [
            reply(None, [background_call("submit-1", command="printf '12 passed, 0 failed'")]),
            reply("已提交，先继续说明项目结构。"),
            reply(None, [tool_call("check-1", "background_check", '{"task_id":1}')]),
            reply("测试已完成，12 passed, 0 failed。"),
        ]
        state = QueryState(
            FakeClient(responses), UsageLedger(), turn=1,
            tool_executor=self.executor(), on_event=events.append,
        )
        state.messages.append({"role": "user", "content": "后台跑测试，同时说明项目结构"})
        self.assertEqual(query_loop(state), "已提交，先继续说明项目结构。")

        submitted = next(event for event in events if event["type"] == "background_submitted")
        task_id = submitted["task"]["task_id"]
        task = self.wait_completed(task_id)
        self.assertEqual(task["result"]["status"], "success")
        self.assertIn("12 passed", task["result"]["stdout"])

        state.turn += 1
        state.messages.append({"role": "user", "content": "测试跑完了吗？"})
        self.assertEqual(query_loop(state), "测试已完成，12 passed, 0 failed。")
        checked_message = next(
            message for message in reversed(state.messages)
            if message["role"] == "tool" and message["tool_call_id"] == "check-1"
        )
        checked = json.loads(checked_message["content"])
        self.assertEqual(checked["status"], "success")
        self.assertEqual(checked["task"]["task_id"], task_id)
        self.assertIn("12 passed", checked["task"]["result"]["stdout"])
        self.assertEqual(state.ledger.summary()["requests"], 4)


if __name__ == "__main__":
    unittest.main()
