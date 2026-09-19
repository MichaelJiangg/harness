import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock

from harness.client import APIError
from harness.engine import QueryState, query_loop
from harness.tools import create_tool_executor, execute_tool
from harness.usage import UsageLedger
from test_engine import FakeClient, reply, tool_call


def bash_call(identifier, command, **options):
    return tool_call(identifier, "bash", json.dumps({"command": command, **options}, ensure_ascii=False))


class BashIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name).resolve()
        self.confirm = Mock(return_value=True)

    def state(self, responses, **kwargs):
        abort = kwargs.pop("abort", Event())
        return QueryState(FakeClient(responses), UsageLedger(), abort=abort,
                          messages=[{"role": "user", "content": "检查命令结果"}],
                          tool_executor=create_tool_executor(self.workspace, confirm=self.confirm, abort=abort),
                          **kwargs)

    def test_both_streams_and_failed_exit_code_reach_model_with_matching_call_id(self):
        command = "printf '正常输出'; printf '错误输出' >&2; exit 7"
        state = self.state([reply(None, [bash_call("failed-command", command)]), reply("命令退出码为 7。")])
        self.assertEqual(query_loop(state), "命令退出码为 7。")
        result_message = state.client.requests[1]["messages"][-1]
        self.assertEqual(result_message["tool_call_id"], "failed-command")
        result = json.loads(result_message["content"])
        self.assertEqual(result["stdout"], "正常输出")
        self.assertEqual(result["stderr"], "错误输出")
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["executed"])
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["cancelled"])
        self.confirm.assert_called_once_with("bash", {"command": command}, self.workspace)
        self.assertEqual(state.ledger.summary()["requests"], 2)
        self.assertEqual(state.ledger.summary()["total_tokens"], 240)

    def test_timeout_preserves_partial_stdout_stderr_and_returns_failure(self):
        state = self.state([
            reply(None, [bash_call("timeout-command", "printf before; printf error-before >&2; sleep 10", timeout=1)]),
            reply("命令超时，已停止。"),
        ])
        self.assertEqual(query_loop(state), "命令超时，已停止。")
        result = json.loads(state.client.requests[1]["messages"][-1]["content"])
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["executed"])
        self.assertEqual(result["status"], "error")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "before")
        self.assertEqual(result["stderr"], "error-before")

    def test_large_dual_output_keeps_each_stream_and_exit_metadata_in_model_context(self):
        program = "import sys; print('OUT_HEAD'+'o'*100000+'OUT_TAIL'); print('ERR_HEAD'+'e'*100000+'ERR_TAIL',file=sys.stderr); sys.exit(3)"
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
        state = self.state([reply(None, [bash_call("long-command", command)]), reply("输出已截断，退出码 3。")])
        self.assertEqual(query_loop(state), "输出已截断，退出码 3。")
        message = state.client.requests[1]["messages"][-1]
        self.assertLessEqual(len(message["content"]), state.tool_result_limit)
        result = json.loads(message["content"])
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["executed"])
        self.assertFalse(result["timed_out"])
        for stream, prefix in (("stdout", "OUT"), ("stderr", "ERR")):
            self.assertIn(prefix + "_HEAD", result[stream])
            self.assertIn(prefix + "_TAIL", result[stream])
            self.assertTrue(result[stream + "_truncated"])

    def test_invalid_timeout_is_returned_before_approval_then_model_can_correct_it(self):
        command = "printf done; touch marker.txt"
        state = self.state([
            reply(None, [bash_call("invalid-timeout", command, timeout=121)]),
            reply(None, [bash_call("valid-timeout", command, timeout=1)]),
            reply("已完成。"),
        ])
        self.assertEqual(query_loop(state), "已完成。")
        invalid = json.loads(state.client.requests[1]["messages"][-1]["content"])
        self.assertFalse(invalid["executed"])
        self.assertEqual(invalid["code"], "invalid_arguments")
        valid = json.loads(state.client.requests[2]["messages"][-1]["content"])
        self.assertEqual(valid["exit_code"], 0)
        self.assertEqual(valid["stdout"], "done")
        self.assertTrue((self.workspace / "marker.txt").exists())
        self.confirm.assert_called_once_with("bash", {"command": command, "timeout": 1}, self.workspace)

    def test_unapproved_command_never_starts(self):
        command = "printf changed > marker.txt"
        for options, code in (({}, "confirmation_required"), ({"confirm": lambda *args: False}, "confirmation_denied")):
            result = execute_tool("bash", {"command": command}, workspace=self.workspace, **options)
            self.assertEqual(result["code"], code)
            self.assertFalse(result["executed"])
            self.assertFalse((self.workspace / "marker.txt").exists())

    def test_model_retry_does_not_execute_or_confirm_the_command_twice(self):
        abort = Mock(wraps=Event())
        abort.wait.return_value = False
        error = APIError("暂时无法连接", retryable=True)
        state = self.state([
            reply(None, [bash_call("once", "printf x >> marker.txt; printf done")]),
            error,
            reply("已完成。"),
        ], abort=abort)
        self.assertEqual(query_loop(state), "已完成。")
        self.assertEqual((self.workspace / "marker.txt").read_text(), "x")
        self.assertEqual(self.confirm.call_count, 1)
        self.assertEqual(len(state.client.requests), 3)


if __name__ == "__main__":
    unittest.main()
