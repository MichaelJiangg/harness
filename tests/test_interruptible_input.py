from io import StringIO
from contextlib import contextmanager
from copy import deepcopy
from queue import Queue
from threading import Event, Lock
import time
import unittest
from unittest.mock import Mock, patch

from harness.cli import run_cli
from harness.engine import QueryAborted
from harness.interruptible_input import (
    EOF,
    IDLE,
    INTERRUPT,
    read_interruptible_line,
    terminal_cbreak,
)


class InterruptibleInputTests(unittest.TestCase):
    def read(self, chunks, *, select_result=False):
        output = StringIO()
        queue = Queue()
        for chunk in chunks:
            queue.put(chunk)
        with patch("harness.interruptible_input.os.read", side_effect=lambda *_args, **_kwargs: queue.get()), \
                patch("harness.interruptible_input.select.select", return_value=([1] if select_result else [], [], [])):
            return read_interruptible_line(1, output, Lock()), output.getvalue()

    def test_text_and_enter_are_decoded_as_one_line(self):
        value, output = self.read([b"\xe4", b"\xbd", b"\xa0", b"\xe5", b"\xa5", b"\xbd", b"\r"])
        self.assertEqual(value, "你好")
        self.assertEqual(output, "你好\r\n")

    def test_escape_alone_returns_interrupt(self):
        value, _ = self.read([b"\x1b"])
        self.assertIs(value, INTERRUPT)

    def test_arrow_escape_sequence_is_ignored(self):
        value, _ = self.read(
            [b"x", b"\x1b", b"[", b"C", b"y", b"\r", b"\n"],
            select_result=True,
        )
        self.assertEqual(value, "xy")

    def test_backspace_removes_character_and_repaints(self):
        value, output = self.read([b"\xe7", b"\x94", b"\xb2", b"\x7f", b"\r"])
        self.assertEqual(value, "")
        self.assertEqual(output, "甲\b \b\r\n")

    def test_eof_returns_eof(self):
        value, _ = self.read([b""])
        self.assertIs(value, EOF)

    def test_crlf_is_consumed_as_one_enter(self):
        value, _ = self.read([b"ok", b"\r", b"\n"], select_result=True)
        self.assertEqual(value, "ok")

    def test_idle_worker_state_exits_cbreak_without_input(self):
        should_continue = Mock(side_effect=[True, False])
        with patch("harness.interruptible_input.os.read", side_effect=AssertionError("不应阻塞读取")), \
                patch("harness.interruptible_input.select.select", return_value=([], [], [])):
            value = read_interruptible_line(
                1, StringIO(), Lock(), should_continue=should_continue,
            )
        self.assertIs(value, IDLE)

    def test_cbreak_restores_original_terminal_settings(self):
        original = [0, 1, 2, 255, 4, 5]
        applied = []
        with patch("harness.interruptible_input.termios.ICANON", 0x0002), \
                patch("harness.interruptible_input.termios.IEXTEN", 0x8000), \
                patch("harness.interruptible_input.termios.ECHO", 0x0008), \
                patch("harness.interruptible_input.termios.TCSADRAIN", 1), \
                patch("harness.interruptible_input.termios.tcgetattr", return_value=original[:]) as get, \
                patch("harness.interruptible_input.termios.tcsetattr", side_effect=lambda fd, when, mode: applied.append(mode[:])) as set_mode:
            with terminal_cbreak(3):
                pass
        get.assert_called_once_with(3)
        self.assertEqual(applied[0][3], original[3] & ~(0x0002 | 0x8000 | 0x0008))
        self.assertEqual(applied[1], original)
        self.assertEqual(set_mode.call_count, 2)

    def test_cli_escape_sets_query_abort_and_stops_current_run(self):
        class FakeInput(StringIO):
            def __init__(self, source):
                super().__init__(source)
                self.queue = Queue()
                for line in ("问题\n", "\n", "\n"):
                    self.queue.put(line)

            def isatty(self):
                return True

            def fileno(self):
                return 1

            def readline(self):
                return self.queue.get(timeout=3)

        class FakeOutput(StringIO):
            def isatty(self):
                return True

        observed = {}

        def tracked_query(state):
            while not state.abort.is_set():
                time.sleep(0.005)
            observed["aborted"] = state.abort.is_set()
            raise QueryAborted()

        fake_input = FakeInput("")
        output = FakeOutput()
        responses = iter([INTERRUPT, EOF])

        @contextmanager
        def noop_cbreak(_fd):
            yield

        with patch("harness.cli.sys.stdin", fake_input), \
                patch("harness.cli.query_loop", side_effect=tracked_query), \
                patch("harness.cli.terminal_cbreak", side_effect=noop_cbreak), \
                patch("harness.cli.read_interruptible_line", side_effect=lambda *_args, **_kwargs: next(responses)):
            run_cli(Mock(), output=output)

        self.assertTrue(observed.get("aborted"))
        self.assertIn("查询已停止。", output.getvalue())

    def test_cli_escape_keeps_streamed_prefix_and_ignores_later_text(self):
        class FakeInput(StringIO):
            def __init__(self):
                super().__init__("")
                self.queue = Queue()
                self.queue.put("问题\n")
                for line in ("\n", "\n", ""):
                    self.queue.put(line)

            def isatty(self):
                return True

            def fileno(self):
                return 1

            def readline(self):
                return self.queue.get(timeout=3)

        class FakeOutput(StringIO):
            def isatty(self):
                return True

        first_fragment_sent = Event()
        first_query_finished = Event()

        def tracked_query(state):
            state.on_event({"type": "response_start"})
            state.on_event({"type": "text", "text": "甲乙"})
            first_fragment_sent.set()
            while not state.abort.is_set():
                time.sleep(0.005)
            state.on_event({"type": "text", "text": "丙"})
            first_query_finished.set()
            raise QueryAborted()

        calls = 0

        def interrupt_reader(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_fragment_sent.wait(3)
                return INTERRUPT
            first_query_finished.wait(3)
            return EOF

        @contextmanager
        def noop_cbreak(_fd):
            yield

        output = FakeOutput()
        with patch("harness.cli.sys.stdin", FakeInput()), \
                patch("harness.cli.query_loop", side_effect=tracked_query), \
                patch("harness.cli.terminal_cbreak", side_effect=noop_cbreak), \
                patch("harness.cli.read_interruptible_line", side_effect=interrupt_reader):
            run_cli(Mock(), output=output)

        self.assertIn("甲乙", output.getvalue())
        self.assertNotIn("丙", output.getvalue())
        self.assertIn("查询已停止。", output.getvalue())

    def test_cli_escape_keeps_partial_reply_in_next_context(self):
        class FakeInput(StringIO):
            def __init__(self):
                super().__init__("")
                self.queue = Queue()
                self.queue.put("问题\n")
                for line in ("\n", "\n", ""):
                    self.queue.put(line)

            def isatty(self):
                return True

            def fileno(self):
                return 1

            def readline(self):
                return self.queue.get(timeout=3)

        class FakeOutput(StringIO):
            def isatty(self):
                return True

        first_fragment_sent = Event()
        first_query_finished = Event()
        second_state = {}
        query_calls = 0

        def first_query(state):
            state.on_event({"type": "response_start"})
            state.on_event({"type": "text", "text": "甲乙"})
            first_fragment_sent.set()
            while not state.abort.is_set():
                time.sleep(0.005)
            state.on_event({"type": "text", "text": "丙"})
            first_query_finished.set()
            raise QueryAborted()

        def second_query(state):
            second_state["messages"] = deepcopy(state.messages)
            return "继续回答"

        def query_side_effect(state):
            nonlocal query_calls
            query_calls += 1
            return first_query(state) if query_calls == 1 else second_query(state)

        reader_calls = 0

        def interrupt_reader(*_args, **_kwargs):
            nonlocal reader_calls
            reader_calls += 1
            if reader_calls == 1:
                first_fragment_sent.wait(3)
                return INTERRUPT
            if reader_calls == 2:
                first_query_finished.wait(3)
                return "继续\n"
            return EOF

        @contextmanager
        def noop_cbreak(_fd):
            yield

        output = FakeOutput()
        with patch("harness.cli.sys.stdin", FakeInput()), \
                patch("harness.cli.query_loop", side_effect=query_side_effect), \
                patch("harness.cli.terminal_cbreak", side_effect=noop_cbreak), \
                patch("harness.cli.read_interruptible_line", side_effect=interrupt_reader):
            run_cli(Mock(), output=output)

        roles = [message.get("role") for message in second_state["messages"]]
        self.assertEqual(roles[-2:], ["assistant", "user"])
        self.assertIn("继续", second_state["messages"][-1]["content"])
        self.assertIn("甲乙", second_state["messages"][-2]["content"])
        self.assertNotIn("丙", second_state["messages"][-2]["content"])

    def test_cli_can_accept_question_after_escape(self):
        class FakeInput(StringIO):
            def __init__(self):
                super().__init__("")
                self.queue = Queue()
                for line in ("问题\n", "\n", "\n", ""):
                    self.queue.put(line)

            def isatty(self):
                return True

            def fileno(self):
                return 1

            def readline(self):
                return self.queue.get(timeout=3)

        class FakeOutput(StringIO):
            def isatty(self):
                return True

        first_aborted = {}
        second_called = {}
        allow_first_finish = Event()
        calls = 0
        query_calls = 0

        def first_query(state):
            while not state.abort.is_set():
                time.sleep(0.005)
            first_aborted["value"] = True
            if not allow_first_finish.wait(3):
                raise AssertionError("Test did not release the interrupted query")
            raise QueryAborted()

        def second_query(_state):
            second_called["value"] = True
            return "继续回答"

        def query_side_effect(state):
            nonlocal query_calls
            query_calls += 1
            return first_query(state) if query_calls == 1 else second_query(state)

        def interrupt_reader(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return INTERRUPT
            if calls == 2:
                allow_first_finish.set()
                time.sleep(0.05)
                return "继续问题"
            return EOF

        @contextmanager
        def noop_cbreak(_fd):
            yield

        output = FakeOutput()
        with patch("harness.cli.sys.stdin", FakeInput()), \
                patch("harness.cli.query_loop", side_effect=query_side_effect), \
                patch("harness.cli.terminal_cbreak", side_effect=noop_cbreak), \
                patch("harness.cli.read_interruptible_line", side_effect=interrupt_reader):
            run_cli(Mock(), output=output)

        self.assertTrue(first_aborted.get("value"))
        self.assertTrue(second_called.get("value"))
        self.assertIn("查询已停止。", output.getvalue())
        self.assertIn("继续回答", output.getvalue())


if __name__ == "__main__":
    unittest.main()
