from copy import deepcopy
from io import StringIO
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Condition, Event, Thread
import unittest
from unittest.mock import Mock, patch

from harness.cli import HELP, run_cli
from harness.client import DEFAULT_MODEL
from harness.engine import query_loop
from harness.tools import create_tool_executor
from harness.usage import UsageLedger


def reply(content="回答完成。", *, prompt=10, completion=5, hit=2, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": DEFAULT_MODEL,
        "choices": [{
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": message,
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": prompt - hit,
        },
    }


TOOL_CALL = {
    "id": "call_read_1",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path":"example.txt"}'},
}


class QueuedInput:
    def __init__(self, *lines):
        self.lines = Queue()
        self.eof_read = Event()
        for line in lines:
            self.send(line)

    def send(self, line):
        self.lines.put(line)

    def readline(self):
        line = self.lines.get(timeout=3)
        if not line:
            self.eof_read.set()
        return line

    def isatty(self):
        return False


class ObservableOutput(StringIO):
    def __init__(self):
        super().__init__()
        self.changed = Condition()

    def write(self, value):
        with self.changed:
            result = super().write(value)
            self.changed.notify_all()
            return result

    def wait_for(self, text):
        with self.changed:
            return self.changed.wait_for(lambda: text in self.getvalue(), timeout=3)


class CLISession:
    def __init__(self, client, *, ledger=None, lines=(), terminal=False, character_delay=0.02):
        self.input = QueuedInput(*lines)
        self.output = ObservableOutput()
        self.input.isatty = lambda: terminal
        self.output.isatty = lambda: terminal
        self.errors = StringIO()
        self.failures = []
        self.finished = Event()

        def run():
            try:
                run_cli(
                    client, ledger=ledger, input_stream=self.input,
                    output=self.output, error_output=self.errors,
                    character_delay=character_delay,
                )
            except BaseException as error:
                self.failures.append(error)
            finally:
                self.finished.set()

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise AssertionError("CLI test session did not finish")
        if self.failures:
            raise self.failures[0]

    def close(self):
        self.input.send("")
        self.join()


class CLITests(unittest.TestCase):
    def track_workers(self):
        workers = []

        def create_worker(*args, **kwargs):
            worker = Thread(*args, **kwargs)
            workers.append(worker)
            return worker

        patcher = patch("harness.cli.Thread", side_effect=create_worker)
        patcher.start()
        self.addCleanup(patcher.stop)
        return workers

    def finish_worker(self, workers):
        workers[-1].join(timeout=3)
        self.assertFalse(workers[-1].is_alive())

    def test_local_commands_do_not_request_the_model(self):
        client = Mock()
        output = StringIO()
        errors = StringIO()
        run_cli(
            client,
            input_stream=StringIO("/cost\n/help\n/unknown\n/exit\n"),
            output=output,
            error_output=errors,
        )
        client.complete.assert_not_called()
        self.assertIn("模型请求：0 次。", output.getvalue())
        self.assertEqual(output.getvalue().count(HELP), 2)
        self.assertIn("未知命令，输入 /help 查看帮助。", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_mode_command_switches_auto_and_rejects_invalid_value(self):
        output = StringIO()
        errors = StringIO()
        run_cli(
            Mock(),
            input_stream=StringIO("/mode auto\n/mode invalid\n/mode ask\n/exit\n"),
            output=output,
            error_output=errors,
        )
        text = output.getvalue()
        self.assertIn("权限模式已切换为 auto：当前工作目录。", text)
        self.assertIn("用法：/mode ask|auto [目录]", text)
        self.assertIn("权限模式已切换为 ask：当前工作目录。", text)
        self.assertEqual(errors.getvalue(), "")

    def test_eof_waits_for_the_answer_to_a_single_question(self):
        started = Event()
        release = Event()

        def complete(**_):
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release the model request")
            return reply()

        client = Mock(complete=Mock(side_effect=complete))
        session = CLISession(client, lines=("你好\n", ""))
        try:
            self.assertTrue(started.wait(3))
            self.assertTrue(session.input.eof_read.wait(3))
            self.assertFalse(session.finished.wait(0.05))
            release.set()
            session.join()
            self.assertIn("DeepSeek > 回答完成。", session.output.getvalue())
            self.assertIn("合计 15 token", session.output.getvalue())
            client.complete.assert_called_once()
            self.assertEqual(session.errors.getvalue(), "")
        finally:
            release.set()
            session.close()

    def test_tool_loop_records_each_request_and_cost_accumulates(self):
        ledger = UsageLedger(pricing={
            "input_hit_per_million": 1,
            "input_miss_per_million": 2,
            "output_per_million": 3,
            "currency": "USD",
        })
        client = Mock(complete=Mock(side_effect=[
            reply(None, tool_calls=[deepcopy(TOOL_CALL)]),
            reply("文件已经读取。", prompt=20, completion=3, hit=8),
        ]))
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        content = "来自测试文件的内容，不应直接打印到终端。"
        Path(workspace.name, "example.txt").write_text(content, encoding="utf-8")
        factory = patch("harness.cli.create_tool_executor", return_value=create_tool_executor(workspace.name))
        factory.start()
        self.addCleanup(factory.stop)
        session = CLISession(client, ledger=ledger, lines=("读取文件\n",))
        try:
            self.assertTrue(session.output.wait_for("DeepSeek > 文件已经读取。"))
            session.input.send("/cost\n")
            self.assertTrue(session.output.wait_for("模型请求：2 次。"))
            session.close()
            self.assertEqual(client.complete.call_count, 2)
            self.assertEqual(ledger.summary()["total_tokens"], 38)
            self.assertEqual([record["turn"] for record in ledger.records], [1, 1])
            output = session.output.getvalue()
            self.assertIn(f"[工具] read_file：已读取文件（{len(content)} 字符）。", output)
            self.assertNotIn(content, output)
            import json
            message = next(item for item in client.complete.call_args_list[1].kwargs["messages"]
                           if item["role"] == "tool")
            result = json.loads(message["content"])
            self.assertEqual(result["content"], content)
            self.assertTrue(result["executed"])
            self.assertIn("Token 合计：输入 30，输出 8，合计 38。", output)
            self.assertIn("预估费用合计：USD 0.00007400", output)
            self.assertIn("[请求 #2／对话 1]", output)
            self.assertEqual(session.errors.getvalue(), "")
        finally:
            session.close()

    def test_cost_remains_available_while_model_request_is_running(self):
        started = Event()
        release = Event()

        def complete(**_):
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release the model request")
            return reply()

        client = Mock(complete=Mock(side_effect=complete))
        session = CLISession(client, lines=("你好\n",))
        try:
            self.assertTrue(started.wait(3))
            session.input.send("/cost\n")
            self.assertTrue(session.output.wait_for("尚未返回的请求用量将在返回后记录。"))
            self.assertIn("模型请求：0 次。", session.output.getvalue())
            self.assertNotIn("DeepSeek >", session.output.getvalue())
            client.complete.assert_called_once()
            release.set()
            session.close()
            self.assertIn("DeepSeek > 回答完成。", session.output.getvalue())
            self.assertEqual(session.errors.getvalue(), "")
        finally:
            release.set()
            session.close()

    def test_read_file_workspace_is_captured_before_first_question(self):
        import json

        with TemporaryDirectory() as initial, TemporaryDirectory() as later:
            Path(initial, "example.txt").write_text("启动目录的文件。", encoding="utf-8")
            Path(later, "example.txt").write_text("其他目录的文件。", encoding="utf-8")
            client = Mock(complete=Mock(side_effect=[
                reply(None, tool_calls=[deepcopy(TOOL_CALL)]), reply("读取结束。"),
            ]))
            with patch("harness.tools.executor.Path.cwd", return_value=Path(initial)) as cwd:
                session = CLISession(client)
                try:
                    self.assertTrue(session.output.wait_for(HELP))
                    cwd.return_value = Path(later)
                    session.input.send("读取文件\n")
                    session.close()
                    message = next(item for item in client.complete.call_args_list[1].kwargs["messages"]
                                   if item["role"] == "tool")
                    self.assertEqual(json.loads(message["content"])["content"], "启动目录的文件。")
                    self.assertIn("[工具] read_file：已读取文件", session.output.getvalue())
                    self.assertEqual(session.errors.getvalue(), "")
                finally:
                    session.close()

    def test_new_question_is_rejected_while_a_query_is_running(self):
        started = Event()
        release = Event()
        requested_messages = []

        def complete(**request):
            requested_messages.extend(deepcopy(request["messages"]))
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release the model request")
            return reply()

        client = Mock(complete=Mock(side_effect=complete))
        session = CLISession(client, lines=("第一个问题\n",))
        try:
            self.assertTrue(started.wait(3))
            session.input.send("第二个问题\n")
            self.assertTrue(session.output.wait_for("上一轮查询进行中，请等待回答"))
            session.input.send("/compact\n")
            self.assertTrue(session.output.wait_for("上一轮查询进行中，暂时无法压缩"))
            release.set()
            session.close()
            client.complete.assert_called_once()
            self.assertEqual(
                [message["content"] for message in requested_messages if message["role"] == "user"],
                ["第一个问题"],
            )
            self.assertEqual(session.errors.getvalue(), "")
        finally:
            release.set()
            session.close()

    def test_manual_compaction_without_old_history_does_not_call_model(self):
        client = Mock()
        output = StringIO()
        run_cli(client, input_stream=StringIO("/compact\n"), output=output)
        client.complete.assert_not_called()
        self.assertIn("历史较短，无需压缩", output.getvalue())
        self.assertIn("/compact", HELP)

    def test_manual_compaction_commits_only_compacted_history_without_command(self):
        workers = self.track_workers()
        requests = []
        compact_inputs = []
        compacted = [{"role": "system", "content": "摘要与系统指令"}]

        def query(state):
            requests.append(deepcopy(state.messages))
            answer = f"第 {len(requests)} 次回答"
            state.messages.append({"role": "assistant", "content": answer})
            return answer

        def compact(state, *, force):
            self.assertTrue(force)
            compact_inputs.append(deepcopy(state.messages))
            state.messages = deepcopy(compacted)
            state.on_event({"type": "compact_done", "before": 200, "after": 20, "attempt": 1})
            return True

        with patch("harness.cli.query_loop", side_effect=query), patch("harness.cli.compact_history", side_effect=compact):
            session = CLISession(Mock(), lines=("旧问题\n",))
            try:
                self.assertTrue(session.output.wait_for("DeepSeek > 第 1 次回答"))
                self.finish_worker(workers)
                session.input.send("/compact\n")
                self.assertTrue(session.output.wait_for("从 200 字符缩短至 20 字符"))
                self.finish_worker(workers)
                session.input.send("新问题\n")
                self.assertTrue(session.output.wait_for("DeepSeek > 第 2 次回答"))
                session.close()
                self.assertEqual(requests[1], compacted + [{"role": "user", "content": "新问题"}])
                self.assertEqual(compact_inputs[0][1:], [
                    {"role": "user", "content": "旧问题"},
                    {"role": "assistant", "content": "第 1 次回答"},
                ])
                self.assertEqual(session.errors.getvalue(), "")
            finally:
                session.close()

    def test_failed_manual_compaction_preserves_original_history(self):
        workers = self.track_workers()
        requests = []
        compact_called = Event()

        def query(state):
            requests.append(deepcopy(state.messages))
            answer = f"第 {len(requests)} 次回答"
            state.messages.append({"role": "assistant", "content": answer})
            return answer

        def compact(state, *, force):
            state.messages.clear()
            compact_called.set()
            raise RuntimeError("压缩失败")

        with patch("harness.cli.query_loop", side_effect=query), patch("harness.cli.compact_history", side_effect=compact):
            session = CLISession(Mock(), lines=("旧问题\n",))
            try:
                self.assertTrue(session.output.wait_for("DeepSeek > 第 1 次回答"))
                self.finish_worker(workers)
                session.input.send("/compact\n")
                self.assertTrue(compact_called.wait(3))
                self.finish_worker(workers)
                session.input.send("新问题\n")
                self.assertTrue(session.output.wait_for("DeepSeek > 第 2 次回答"))
                session.close()
                self.assertEqual(requests[1][1:], [
                    {"role": "user", "content": "旧问题"},
                    {"role": "assistant", "content": "第 1 次回答"},
                    {"role": "user", "content": "新问题"},
                ])
                self.assertIn("错误：压缩失败", session.errors.getvalue())
            finally:
                session.close()

    def test_cost_and_exit_remain_available_during_manual_compaction(self):
        workers = self.track_workers()
        started = Event()
        release = Event()

        def compact(state, *, force):
            state.on_event({"type": "compact_start", "attempt": 1})
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release compaction")
            state.on_event({"type": "compact_done", "before": 200, "after": 20, "attempt": 1})
            return True

        with patch("harness.cli.compact_history", side_effect=compact):
            session = CLISession(Mock(), lines=("/compact\n",))
            try:
                self.assertTrue(started.wait(3))
                session.input.send("/compact\n")
                session.input.send("/cost\n")
                self.assertTrue(session.output.wait_for("尚未返回的请求用量将在返回后记录"))
                self.assertIn("暂时无法压缩", session.output.getvalue())
                self.assertIn("模型请求：0 次。", session.output.getvalue())
                session.input.send("/exit\n")
                session.join()
                self.assertTrue(workers[-1].is_alive())
                release.set()
                self.finish_worker(workers)
                self.assertNotIn("缩短至", session.output.getvalue())
                self.assertEqual(session.errors.getvalue(), "")
            finally:
                release.set()
                session.close()

    def test_exit_prevents_tools_and_further_requests_after_inflight_reply(self):
        started = Event()
        release = Event()
        query_finished = Event()
        executor = Mock()
        ledger = UsageLedger()

        def complete(**_):
            started.set()
            if not release.wait(3):
                raise AssertionError("Test did not release the model request")
            return reply(None, tool_calls=[deepcopy(TOOL_CALL)])

        def tracked_query(state):
            state.tool_executor = executor
            try:
                return query_loop(state)
            finally:
                query_finished.set()

        client = Mock(complete=Mock(side_effect=complete))
        with patch("harness.cli.query_loop", side_effect=tracked_query):
            session = CLISession(client, ledger=ledger, lines=("读取文件\n",))
            try:
                self.assertTrue(started.wait(3))
                session.input.send("/exit\n")
                session.join()
                self.assertFalse(query_finished.is_set())
                release.set()
                self.assertTrue(query_finished.wait(3))
                executor.assert_not_called()
                client.complete.assert_called_once()
                self.assertEqual(ledger.summary()["requests"], 1)
                self.assertNotIn("[工具]", session.output.getvalue())
                self.assertNotIn("DeepSeek >", session.output.getvalue())
                self.assertEqual(session.errors.getvalue(), "")
            finally:
                release.set()
                session.close()
                self.assertTrue(query_finished.wait(3))

    def test_module_help_runs_without_api_key(self):
        environment = os.environ.copy()
        environment.pop("DEEPSEEK_API_KEY", None)
        result = subprocess.run(
            [sys.executable, "-m", "harness", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("/cost", result.stdout)
        self.assertIn("DEEPSEEK_API_KEY", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_module_reports_blank_api_key(self):
        environment = os.environ.copy()
        # 显式空值可避免测试读取用户本机 .env 中的真实密钥。
        environment["DEEPSEEK_API_KEY"] = ""
        result = subprocess.run(
            [sys.executable, "-m", "harness"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            input="",
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("请在项目 .env 或环境变量中设置 DEEPSEEK_API_KEY。", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
