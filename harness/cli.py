"""终端交互和本地命令；模型请求在后台运行，等待时仍可查看 /cost。"""

import sys
from copy import deepcopy
from threading import Event, Lock, Thread

from .client import DEFAULT_MODEL
from .engine import QueryAborted, QueryState, SYSTEM_PROMPT, compact_history, query_loop
from .usage import UsageLedger, format_cost, format_usage

HELP = """输入问题开始查询，当前工具只返回占位结果。
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
            write(f'[工具占位] {event["name"]}：{event["result"]["message"]}')
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
                input_closed = True
                # 管道输入结束仍等待结果；/exit 则立即结束本地会话。
                if worker:
                    worker.join()
                return
            text = line.strip()
            if not text:
                continue
            if text == "/exit":
                abort.set()
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
            elif worker and worker.is_alive():
                write("上一轮查询进行中，暂时无法压缩；期间可以输入 /cost。" if text == "/compact"
                      else "上一轮查询进行中，请等待回答；期间可以输入 /cost。")
            else:
                turn += 1
                compact = text == "/compact"
                state = QueryState(
                    client=client, ledger=ledger, turn=turn, abort=abort, on_event=on_event,
                    messages=deepcopy(messages) + ([] if compact else [{"role": "user", "content": text}]),
                )
                worker = Thread(target=run_query, args=(state,), kwargs={"compact": compact}, daemon=True)
                worker.start()
    except KeyboardInterrupt:
        abort.set()
        write("\n查询已停止。")
