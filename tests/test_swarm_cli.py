import json
from pathlib import Path
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


def handoff(next_role, summary):
    return json.dumps({
        "next_role": next_role, "summary": summary,
        "artifacts": [], "feedback": None,
    }, ensure_ascii=False)


class SwarmCLITests(unittest.TestCase):
    def test_swarm_output_and_handoff_labels(self):
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=[call("swarm-1", "swarm", {
                "description": "实现缓存模块",
                "task": "编写、审查并测试缓存模块。",
            })]),
            reply(None, tool_calls=[
                call("read-1", "read_file", {"path": "login-page/spec.md"}),
            ]),
            reply(handoff("Reviewer", "已创建 cache.py")),
            reply(handoff("Tester", "审查通过")),
            reply(handoff(None, "团队完成，测试通过")),
            reply("团队协作结果已整理。"),
        ]))
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        Path(workspace.name, "login-page").mkdir()
        Path(workspace.name, "login-page", "spec.md").write_text("缓存模块规格", encoding="utf-8")

        def factory(*args, **kwargs):
            return create_tool_executor(workspace.name, *args, **kwargs)

        factory_patch = patch("harness.cli.create_tool_executor", side_effect=factory)
        factory_patch.start()
        self.addCleanup(factory_patch.stop)

        session = CLISession(
            client, lines=("用团队模式实现缓存模块\n",),
            terminal=True, character_delay=0,
        )
        try:
            self.assertTrue(session.output.wait_for("[确认] 输入 y 批准本次团队协作"))
            session.input.send("y\n")
            self.assertTrue(session.output.wait_for(
                "[swarm] 角色分配：Coder、Reviewer、Tester"
            ))
            self.assertTrue(session.output.wait_for("[swarm] Coder 开始工作"))
            self.assertTrue(session.output.wait_for("[swarm] Reviewer 完成"))
            self.assertTrue(session.output.wait_for("[swarm] 团队协作完成（3 轮）"))
            self.assertTrue(session.output.wait_for("团队协作结果已整理。"))
            session.close()
            self.assertEqual(session.errors.getvalue(), "")
            self.assertEqual(client.complete.call_count, 6)
            output = session.output.getvalue()
            self.assertNotIn("[swarm] [工具] read_file", output)
            self.assertNotIn("[swarm] [请求 #", output)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
