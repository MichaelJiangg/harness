"""在真实交互终端中读取可被 Esc 中断的单行输入。"""

import codecs
from contextlib import contextmanager
import os
import select
import termios


INTERRUPT = object()
EOF = object()
IDLE = object()


@contextmanager
def terminal_cbreak(fd):
    original = termios.tcgetattr(fd)
    modified = original[:]
    modified[3] &= ~(termios.ICANON | termios.IEXTEN | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSADRAIN, modified)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def read_interruptible_line(fd, output, output_lock, *, should_continue=None):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    characters = []
    pending = bytearray()

    def read_byte():
        if should_continue is None:
            if pending:
                return bytes([pending.pop(0)])
            return os.read(fd, 1)
        while True:
            if not should_continue():
                return IDLE
            readable, _, _ = select.select([fd], [], [], 0.1)
            if readable:
                break
        if not should_continue():
            return IDLE
        if pending:
            return bytes([pending.pop(0)])
        return os.read(fd, 1)

    def unread(byte):
        if byte:
            pending.insert(0, byte[0])

    def byte_waiting():
        return bool(select.select([fd], [], [], 0.05)[0])

    def backspace():
        if not characters:
            return
        characters.pop()
        with output_lock:
            output.write("\b \b")
            output.flush()

    def echo(text):
        if text:
            with output_lock:
                output.write(text)
                output.flush()

    while True:
        chunk = read_byte()
        if chunk is IDLE:
            return IDLE
        if chunk == b"":
            return EOF
        if chunk in {b"\r", b"\n"}:
            if chunk == b"\r" and byte_waiting():
                lookahead = read_byte()
                if lookahead is IDLE:
                    return IDLE
                if lookahead != b"\n":
                    unread(lookahead)
            echo("\r\n")
            return "".join(characters)
        if chunk in {b"\x08", b"\x7f"}:
            backspace()
            continue
        if chunk == b"\x1b":
            if not byte_waiting():
                return INTERRUPT
            prefix = read_byte()
            if prefix is IDLE:
                return IDLE
            if prefix in {b"[", b"O"}:
                while True:
                    part = read_byte()
                    if part is IDLE:
                        return IDLE
                    if part in {b"", b"\r", b"\n"} or (0x40 <= part[0] <= 0x7e):
                        break
            continue
        decoded = decoder.decode(chunk)
        if decoded:
            characters.append(decoded)
            if decoded.isprintable():
                echo(decoded)


__all__ = ["EOF", "IDLE", "INTERRUPT", "read_interruptible_line", "terminal_cbreak"]
