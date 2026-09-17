import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Timer
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from harness.tools.bash import DEFINITION, DEFAULT_TIMEOUT, MAX_OUTPUT_BYTES, execute
from harness.tools.executor import ToolError


@unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "需要 macOS／Linux")
class BashTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name).resolve()

    def run_command(self, command, timeout=None, *, abort=None):
        arguments = {"command": command}
        if timeout is not None:
            arguments["timeout"] = timeout
        return execute(arguments, self.workspace, abort=abort)

    def python_command(self, source):
        return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"

    def assert_error(self, arguments, code):
        with self.assertRaises(ToolError) as caught:
            execute(arguments, self.workspace)
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn("private details", str(caught.exception))

    def test_definition_requires_confirmation_and_cancellation_with_bounded_timeout(self):
        self.assertTrue(DEFINITION.requires_confirmation)
        self.assertTrue(DEFINITION.supports_cancellation)
        schema = DEFINITION.to_deepseek()["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"command", "timeout"})
        self.assertEqual(schema["required"], ["command"])
        self.assertFalse(schema["additionalProperties"])
        timeout = schema["properties"]["timeout"]
        self.assertEqual((timeout["minimum"], timeout["maximum"], timeout["default"]), (1, 120, DEFAULT_TIMEOUT))

    def test_success_returns_separate_utf8_output_and_exit_code(self):
        result = self.run_command("printf '标准输出\\n'; printf '错误输出\\n' >&2")
        self.assertEqual(result["stdout"], "标准输出\n")
        self.assertEqual(result["stderr"], "错误输出\n")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["cancelled"])
        self.assertFalse(result["stdout_truncated"])
        self.assertFalse(result["stderr_truncated"])
        self.assertEqual(result["stdout_bytes"], len("标准输出\n".encode()))
        self.assertEqual(result["stderr_bytes"], len("错误输出\n".encode()))
        self.assertNotIn("command", result)

    def test_nonzero_exit_keeps_both_streams(self):
        result = self.run_command("printf 'partial'; printf 'failed' >&2; exit 7")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual((result["stdout"], result["stderr"]), ("partial", "failed"))

    def test_command_not_found_returns_127_and_stderr(self):
        result = self.run_command("harness_command_that_does_not_exist_12345")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 127)
        self.assertTrue(result["stderr"])

    def test_uses_workspace_and_supports_pipes_and_redirection(self):
        result = self.run_command("printf 'hello' | tr 'a-z' 'A-Z' > result.txt; pwd; cat result.txt")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], f"{self.workspace}\nHELLO")
        self.assertEqual((self.workspace / "result.txt").read_text(), "HELLO")

    def test_stdin_is_closed(self):
        result = self.run_command("read value; printf '%s' $?")
        self.assertEqual(result["stdout"], "1")
        self.assertEqual(result["exit_code"], 0)

    def test_spawn_uses_clean_environment_without_startup_scripts(self):
        startup = self.workspace / "startup.sh"
        startup.write_text("printf 'startup should not run'\n")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-test-key", "BASH_ENV": str(startup), "ENV": str(startup)}):
            with patch("harness.tools.bash.subprocess.Popen", wraps=subprocess.Popen) as launch:
                result = self.run_command("printf '%s|%s|%s' \"${DEEPSEEK_API_KEY-unset}\" \"${BASH_ENV-unset}\" \"${ENV-unset}\"")
        self.assertEqual(result["stdout"], "unset|unset|unset")
        self.assertEqual(launch.call_args.args[0][:4], ["/bin/bash", "--noprofile", "--norc", "-c"])
        self.assertIs(launch.call_args.kwargs["start_new_session"], True)
        self.assertEqual(launch.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_timeout_retains_partial_stdout_stderr_and_actual_signal_exit(self):
        command = self.python_command("import sys,time; print('before timeout',flush=True); print('diagnostic',file=sys.stderr,flush=True); time.sleep(10)")
        started = monotonic()
        result = self.run_command(command, timeout=1)
        self.assertLess(monotonic() - started, 2.5)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        self.assertEqual(result["stdout"], "before timeout\n")
        self.assertEqual(result["stderr"], "diagnostic\n")

    def test_parent_exit_does_not_escape_timeout_for_child_holding_pipes(self):
        child = self.python_command("import os,time; print(os.getpid(),flush=True); time.sleep(10)")
        started = monotonic()
        result = self.run_command(f"{child} & exit 0", timeout=1)
        self.assertLess(monotonic() - started, 2.5)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 0)
        child_pid = int(result["stdout"].strip())
        self.assert_process_stopped(child_pid)

    def assert_process_stopped(self, pid):
        deadline = monotonic() + 1
        while monotonic() < deadline:
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=1)
            if not status.stdout.strip() or status.stdout.lstrip().startswith("Z"):
                return
            sleep(0.02)
        self.fail(f"子进程 {pid} 仍在运行。")

    def test_large_outputs_are_drained_separately_and_keep_heads_and_tails(self):
        size = MAX_OUTPUT_BYTES * 3
        command = self.python_command(
            f"import sys; sys.stdout.buffer.write(b'OUT_HEAD'+b'x'*{size}+b'OUT_TAIL'); "
            f"sys.stdout.flush(); sys.stderr.buffer.write(b'ERR_HEAD'+b'y'*{size}+b'ERR_TAIL'); sys.stderr.flush()"
        )
        result = self.run_command(command, timeout=5)
        self.assertEqual(result["exit_code"], 0)
        for stream, prefix in (("stdout", "OUT"), ("stderr", "ERR")):
            self.assertTrue(result[stream].startswith(prefix + "_HEAD"))
            self.assertTrue(result[stream].endswith(prefix + "_TAIL"))
            self.assertIn("[output truncated]", result[stream])
            self.assertLessEqual(len(result[stream].encode()), MAX_OUTPUT_BYTES)
            self.assertTrue(result[stream + "_truncated"])
            self.assertEqual(result[stream + "_bytes"], size + 16)

    def test_exact_output_limit_is_not_marked_truncated(self):
        result = self.run_command(self.python_command(f"import sys; sys.stdout.buffer.write(b'x'*{MAX_OUTPUT_BYTES})"))
        self.assertEqual(len(result["stdout"]), MAX_OUTPUT_BYTES)
        self.assertFalse(result["stdout_truncated"])
        self.assertEqual(result["stdout_bytes"], MAX_OUTPUT_BYTES)

    def test_continuous_dual_stream_output_remains_bounded_and_times_out(self):
        command = self.python_command("import os\nwhile True:\n os.write(1,b'x'*4096)\n os.write(2,b'y'*4096)")
        started = monotonic()
        result = self.run_command(command, timeout=1)
        self.assertLess(monotonic() - started, 2.5)
        self.assertTrue(result["timed_out"])
        for stream in ("stdout", "stderr"):
            self.assertTrue(result[stream + "_truncated"])
            self.assertLessEqual(len(result[stream]), MAX_OUTPUT_BYTES)
            self.assertGreater(result[stream + "_bytes"], MAX_OUTPUT_BYTES)

    def test_invalid_output_encoding_is_replaced(self):
        result = self.run_command(self.python_command("import os; os.write(1,b'\\xfftext'); os.write(2,b'\\xfediagnostic')"))
        self.assertEqual(result["stdout"], "\ufffdtext")
        self.assertEqual(result["stderr"], "\ufffddiagnostic")
        self.assertEqual(result["exit_code"], 0)

    def test_abort_terminates_running_process_and_retains_output(self):
        abort = Event()
        timer = Timer(0.2, abort.set)
        self.addCleanup(timer.cancel)
        timer.start()
        started = monotonic()
        result = self.run_command("printf 'started'; sleep 10", timeout=5, abort=abort)
        self.assertLess(monotonic() - started, 1.5)
        self.assertTrue(result["cancelled"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stdout"], "started")
        self.assertEqual(result["exit_code"], -signal.SIGKILL)

    def test_already_cancelled_does_not_launch(self):
        abort = Event()
        abort.set()
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            with self.assertRaises(ToolError) as caught:
                self.run_command("printf 'never'", abort=abort)
        self.assertEqual(caught.exception.code, "execution_cancelled")
        launch.assert_not_called()

    def test_invalid_command_and_timeout_do_not_launch(self):
        with patch("harness.tools.bash.subprocess.Popen") as launch:
            for command in ("", " \n\t", "bad\x00command"):
                with self.subTest(command=repr(command)):
                    self.assert_error({"command": command}, "invalid_arguments")
            for timeout in (0, -1, 121, True, 1.0, "1"):
                with self.subTest(timeout=timeout):
                    self.assert_error({"command": "true", "timeout": timeout}, "invalid_arguments")
        launch.assert_not_called()

    def test_maximum_timeout_is_accepted(self):
        result = self.run_command("true", timeout=120)
        self.assertEqual(result["exit_code"], 0)

    def test_unsupported_platform_does_not_launch(self):
        with patch("harness.tools.bash.sys.platform", "win32"), patch("harness.tools.bash.subprocess.Popen") as launch:
            self.assert_error({"command": "true"}, "unsupported_platform")
        launch.assert_not_called()

    def test_launch_failure_returns_normalized_error(self):
        with patch("harness.tools.bash.subprocess.Popen", side_effect=OSError("private details")):
            self.assert_error({"command": "true"}, "launch_error")

    def test_missing_workspace_returns_launch_error(self):
        with self.assertRaises(ToolError) as caught:
            execute({"command": "true"}, self.workspace / "does-not-exist")
        self.assertEqual(caught.exception.code, "launch_error")

    def test_read_failure_cleans_up_launched_process(self):
        processes = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        selector = selectors.DefaultSelector()
        select = selector.select
        selected = False

        def select_then_fail(timeout):
            nonlocal selected
            if selected:
                raise OSError("private details")
            selected = True
            return select(timeout)

        with patch("harness.tools.bash.subprocess.Popen", side_effect=launch):
            with patch("harness.tools.bash.selectors.DefaultSelector", return_value=selector):
                with patch.object(selector, "select", side_effect=select_then_fail):
                    result = self.run_command("printf 'partial'; sleep 10")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "command_error")
        self.assertEqual(result["stdout"], "partial")
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        self.assertNotIn("private details", result["message"])
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(processes[0].stderr.closed)


if __name__ == "__main__":
    unittest.main()
