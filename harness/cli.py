"""终端交互和本地命令；模型请求在后台运行，等待时仍可查看 /cost。"""

import sys
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread

from .client import DEFAULT_MODEL
from .engine import QueryAborted, QueryState, SYSTEM_PROMPT, compact_history, query_loop
from .tools import create_tool_executor
from .tools.bash import DEFAULT_TIMEOUT
from .usage import UsageLedger, format_cost, format_usage

HELP = """输入问题开始查询，可读取启动目录内的 UTF-8 文本文件；写文件和执行终端命令须逐次预览并输入 y 确认。
确认期间，n 或直接回车拒绝；管道模式不允许写入或执行命令。
命令默认超时 30 秒，最多 120 秒；以当前用户权限运行，工作目录不是文件访问沙箱。
/cost  查看本次会话 token、USD 预估费用及逐请求明细
/compact  压缩旧对话，保留最近几轮
/help  查看帮助
/exit  退出并停止后续模型和工具调用"""


def run_cli(client, *, ledger=None, input_stream=None, output=None, error_output=None,
            character_delay=0.02):
    ledger = ledger if ledger is not None else UsageLedger()
    input_stream = input_stream if input_stream is not None else sys.stdin
    output = output if output is not None else sys.stdout
    error_output = error_output if error_output is not None else sys.stderr
    terminal = input_stream.isatty() and output.isatty()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    abort = Event()
    output_lock = Lock()
    confirmation_lock = Lock()
    pending_confirmation = None
    worker = None
    turn = 0
    stream_line_open = False
    response_streamed = False
    prompt_shown = False
    input_closed = False

    def show_prompt():
        nonlocal prompt_shown
        with output_lock:
            if terminal and not prompt_shown and not input_closed and not abort.is_set():
                print("你 > ", end="", file=output, flush=True)
                prompt_shown = True

    def end_stream_line():
        nonlocal stream_line_open
        if stream_line_open:
            print(file=output, flush=True)
            stream_line_open = False

    def write(text, *, error=False):
        with output_lock:
            end_stream_line()
            print(text, file=error_output if error else output, flush=True)

    def confirm_tool(name, arguments, workspace):
        nonlocal pending_confirmation
        if name not in {"write_file", "bash"}:
            return False
        label = "写入" if name == "write_file" else "命令执行"
        if not terminal:
            write(f"[确认] 非交互模式无法确认{label}，已拒绝本次操作。")
            return False
        request = {"done": Event(), "approved": False, "label": label}
        try:
            with confirmation_lock:
                if input_closed or abort.is_set():
                    return False
                if name == "write_file":
                    candidate = Path(workspace) / arguments["path"]
                    path = candidate.resolve()
                    existed = path.exists()
                    action = "覆盖已有文件的全部内容" if existed else "新建文件（确认后创建缺失的父目录）"
                    content = "\n".join("│ " + line for line in _preview_text(arguments["content"]).split("\n"))
                    preview = (
                        f"\n[写入确认] {name}\n目标路径：{_preview_text(str(path), multiline=False)}\n"
                        f"操作：{action}\n完整内容（控制字符转义显示，换行和制表符保留）：\n"
                        f"┌── 文件内容开始 ──\n{content}\n└── 文件内容结束 ──\n"
                    )
                else:
                    command = "\n".join("│ " + line for line in _preview_text(arguments["command"]).split("\n"))
                    preview = (
                        f"\n[命令确认] {name}\n工作目录：{_preview_text(str(workspace), multiline=False)}\n"
                        f"超时：{arguments.get('timeout', DEFAULT_TIMEOUT)} 秒\n"
                        "完整命令（控制字符转义显示，换行和制表符保留）：\n"
                        f"┌── 命令开始 ──\n{command}\n└── 命令结束 ──\n"
                    )
                pending_confirmation = request
                write(
                    preview + f"[确认] 输入 y 批准本次{label}，n 或直接回车拒绝；/cost、/help、/exit 仍可使用。"
                )
            request["done"].wait()
            with confirmation_lock:
                if not request["approved"] or input_closed or abort.is_set():
                    return False
                if name == "write_file" and (candidate.resolve() != path or path.exists() != existed):
                    write("[确认] 目标路径或文件存在状态已变化，本次确认失效，未写入；请重新发起写入请求。")
                    return False
                return True
        finally:
            with confirmation_lock:
                if pending_confirmation is request:
                    pending_confirmation = None

    def answer_confirmation(text):
        nonlocal pending_confirmation
        with confirmation_lock:
            request = pending_confirmation
            if request is None:
                return False
            if text not in {"y", "n", ""}:
                write(f"[确认] 正在等待本次{request['label']}确认，请输入 y 批准，n 或直接回车拒绝。")
                return True
            pending_confirmation = None
            request["approved"] = text == "y"
            request["done"].set()
            return True

    def cancel_confirmation(*, close_input=False):
        nonlocal pending_confirmation, input_closed
        with confirmation_lock:
            if close_input:
                input_closed = True
            request = pending_confirmation
            pending_confirmation = None
            if request is not None:
                request["done"].set()

    tool_executor = create_tool_executor(confirm=confirm_tool, abort=abort)

    def on_event(event):
        nonlocal stream_line_open, response_streamed
        if abort.is_set():
            return
        if event["type"] == "response_start":
            response_streamed = False
        elif event["type"] == "text":
            fragments = event["text"] if terminal else (event["text"],)
            for fragment in fragments:
                if abort.is_set():
                    return
                with output_lock:
                    if not stream_line_open:
                        print("\nDeepSeek > " if not response_streamed else "DeepSeek > ", end="", file=output)
                    print(fragment, end="", file=output, flush=True)
                    stream_line_open = True
                    response_streamed = True
                if terminal and character_delay > 0 and abort.wait(character_delay):
                    return
        elif event["type"] == "usage":
            write(format_usage(event["record"]))
        elif event["type"] == "tool":
            write(f'[工具] {event["name"]}：{event["result"]["message"]}')
        elif event["type"] == "compact_start":
            write(f'[压缩] 正在总结旧对话（第 {event["attempt"]} 次）。')
        elif event["type"] == "compact_done":
            write(f'[压缩] 已将上下文从 {event["before"]} 字符缩短至 {event["after"]} 字符。')
        elif event["type"] == "compact_skipped":
            write("[压缩] 历史较短，无需压缩。")
        elif event["type"] == "retry":
            if event["partial"]:
                write("[重试] 上次输出未完成，重新生成。")
            write(f'[重试] {event["message"]}；{event["delay"]} 秒后进行第 {event["attempt"]} 次重试。')

    def run_query(state, *, compact=False):
        nonlocal messages
        try:
            if compact:
                if compact_history(state, force=True) and not abort.is_set():
                    messages = state.messages
                return
            answer = query_loop(state)
            if not abort.is_set():
                messages = state.messages
                if not response_streamed:
                    write(f"\nDeepSeek > {answer}")
        except QueryAborted:
            pass
        except Exception as error:
            if not abort.is_set():
                write(f"错误：{error}", error=True)
        finally:
            show_prompt()

    write(f"DeepSeek 查询引擎 · {DEFAULT_MODEL}\n{HELP}")
    try:
        while True:
            if not worker or not worker.is_alive():
                show_prompt()
            line = input_stream.readline()
            with output_lock:
                prompt_shown = False
            if not line:
                cancel_confirmation(close_input=True)
                # 输入结束仍等待已批准操作的结果；命令自身有有限超时。
                if worker:
                    worker.join()
                return
            text = line.strip()
            if text == "/exit":
                abort.set()
                cancel_confirmation()
                if worker:
                    worker.join(timeout=1)
                with output_lock:
                    end_stream_line()
                return
            if text == "/help":
                write(HELP)
            elif text == "/cost":
                write(format_cost(ledger))
                if worker and worker.is_alive():
                    write("查询进行中：尚未返回的请求用量将在返回后记录。")
            elif text.startswith("/") and text != "/compact":
                write("未知命令，输入 /help 查看帮助。")
            elif text == "/compact" and worker and worker.is_alive():
                write("上一轮查询进行中，暂时无法压缩；期间可以输入 /cost。")
            elif answer_confirmation(text):
                continue
            elif not text:
                continue
            elif worker and worker.is_alive():
                write("上一轮查询进行中，请等待回答；期间可以输入 /cost。")
            else:
                turn += 1
                compact = text == "/compact"
                state = QueryState(
                    client=client, ledger=ledger, turn=turn, abort=abort, on_event=on_event,
                    tool_executor=tool_executor,
                    messages=deepcopy(messages) + ([] if compact else [{"role": "user", "content": text}]),
                )
                worker = Thread(target=run_query, args=(state,), kwargs={"compact": compact}, daemon=True)
                worker.start()
    except KeyboardInterrupt:
        abort.set()
        cancel_confirmation()
        if worker:
            worker.join(timeout=1)
        write("\n查询已停止。")


def _preview_text(text, *, multiline=True):
    """保持完整文本可见，防止终端控制序列覆盖确认提示。"""
    return "".join(
        character if character.isprintable() or (multiline and character in "\n\t")
        else f"\\u{ord(character):04x}" if ord(character) <= 0xffff
        else f"\\U{ord(character):08x}"
        for character in text
    )
