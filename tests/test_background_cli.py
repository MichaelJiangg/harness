import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.tools import create_tool_executor
from test_cli import CLISession, reply


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class BackgroundCLITests(unittest.TestCase):
    def test_submit_output_and_next_turn_check(self):
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=[
                call("submit-1", "background_submit", {
                    "description": "运行测试",
                    "command": "printf '12 passed, 0 failed'",
                }),
            ]),
            reply("已提交，先继续说明项目结构。"),
            reply(None, tool_calls=[call("check-1", "background_check", {"task_id": 1})]),
            reply("测试已完成，12 passed, 0 failed。"),
        ]))
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)

        def factory(*args, **kwargs):
            return create_tool_executor(workspace.name, *args, **kwargs)

        factory_patch = patch("harness.cli.create_tool_executor", side_effect=factory)
        factory_patch.start()
        self.addCleanup(factory_patch.stop)

        session = CLISession(
            client, lines=("后台跑测试，同时说明项目结构\n",),
            terminal=True, character_delay=0,
        )
        try:
            self.assertTrue(session.output.wait_for("DeepSeek > 已提交，先继续说明项目结构。"))
            self.assertTrue(session.output.wait_for("[background] 任务 #1 已提交：运行测试"))
            self.assertTrue(session.output.wait_for("[background] 任务 #1：状态 RUNNING"))
            self.assertTrue(session.output.wait_for("[background] 任务 #1：已完成"))
            session.input.send("测试跑完了吗？\n")
            self.assertTrue(session.output.wait_for("DeepSeek > 测试已完成，12 passed, 0 failed。"))
            session.close()
            output = session.output.getvalue()
            self.assertIn("[background] 任务 #1：已完成（耗时", output)
            self.assertEqual(session.errors.getvalue(), "")
            self.assertEqual(client.complete.call_count, 4)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
