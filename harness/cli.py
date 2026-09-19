"""终端交互和本地命令；模型请求在后台运行，等待时仍可查看 /cost。"""

import json
import sys
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread

from .background import BackgroundManager, COMPLETED
from .client import DEFAULT_MODEL
from .config import get_settings
from .engine import QueryAborted, QueryState, SYSTEM_PROMPT, compact_history, query_loop
from .permissions import SessionPermissionCache, get_risk_level
from .tools import create_tool_executor
from .tools.bash import DEFAULT_TIMEOUT, MAX_TIMEOUT
from .tools.swarm import DEFAULT_ROLES as DEFAULT_SWARM_ROLES
from .usage import UsageLedger, format_cost, format_usage

HELP = f"""输入问题开始查询，默认可直接读取和搜索文件；工具权限由 pyproject.toml 配置。
默认低风险读取和已识别的只读命令直接放行，中风险操作确认，破坏性命令醒目警告后确认。
权限规则按禁止优先、数值优先级匹配，首条命中生效；目录授权可免除写入确认。
写入确认会展示本会话授权目录，批准且写入成功后，同目录及子目录不再询问；重启失效。
其余需要确认的操作输入 y 批准，Bash 不记忆授权；明确禁止的调用不会执行。
确认期间，n 或直接回车拒绝；管道模式不允许执行需要确认的工具。
命令默认超时 {DEFAULT_TIMEOUT} 秒，最多 {MAX_TIMEOUT} 秒；以当前用户权限运行，工作目录不是文件访问沙箱。
权限决策和确认结果记录到 .harness/permission.log，不记录正文、搜索词或原始命令。
目录级、多文件及代码质量任务优先通过 delegate 委托独立子任务，主子请求统一计入 /cost，子任务不能递归委托。
耗时测试、批量命令和大型独立分析可通过 background_submit 提交后台，background_check 查询状态和结果。
多角色接力任务可通过 swarm 启动团队协作，角色拥有独立上下文并按交接协议传递成果。
读文件按页返回，全文分析可按下一页游标继续；旧工具批次过长时生成阶段摘要。
/cost  查看本次会话 token、USD 预估费用及逐请求明细
/compact  压缩旧对话，保留最近几轮
/help  查看帮助
/exit  退出并停止后续模型和工具调用"""


def run_cli(client, *, ledger=None, input_stream=None, output=None, error_output=None,
            character_delay=None):
    if character_delay is None:
        character_delay = get_settings()["display"]["character_delay"]
    ledger = ledger if ledger is not None else UsageLedger()
    input_stream = input_stream if input_stream is not None else sys.stdin
    output = output if output is not None else sys.stdout
    error_output = error_output if error_output is not None else sys.stderr
    terminal = input_stream.isatty() and output.isatty()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    abort = Event()
    background_manager = BackgroundManager(**get_settings()["background"])
    background_default_timeout = get_settings()["background"]["default_timeout"]
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
        label = {
            "write_file": "写入", "bash": "命令执行", "background_submit": "后台任务",
            "swarm": "团队协作", "run_verify": "验证",
        }.get(name, "工具调用")
        if not terminal:
            write(f"[确认] 非交互模式无法确认{label}，已拒绝本次操作。")
            return False
        request = {"done": Event(), "approved": False, "label": label}
        try:
            with confirmation_lock:
                if input_closed or abort.is_set():
                    return False
                risk = _confirmation_risk(name, arguments)
                if name == "write_file":
                    candidate = Path(workspace) / arguments["path"]
                    path = candidate.resolve()
                    existed = path.exists()
                    action = "覆盖已有文件的全部内容" if existed else "新建文件（确认后创建缺失的父目录）"
                    content = "\n".join("│ " + line for line in _preview_text(arguments["content"]).split("\n"))
                    preview = (
                        f"\n[写入确认] {name}\n{risk}\n目标路径：{_preview_text(str(path), multiline=False)}\n"
                        f"操作：{action}\n完整内容（控制字符转义显示，换行和制表符保留）：\n"
                        f"┌── 文件内容开始 ──\n{content}\n└── 文件内容结束 ──\n"
                        f"本会话授权目录：{_preview_text(str(path.parent), multiline=False)}（含子目录）\n"
                        "输入 y 将批准本次写入；写入成功后记住以上目录，当前会话内复用，重启失效。\n"
                    )
                elif name == "bash":
                    command = "\n".join("│ " + line for line in _preview_text(arguments["command"]).split("\n"))
                    preview = (
                        f"\n{risk}\n[命令确认] {name}\n工作目录：{_preview_text(str(workspace), multiline=False)}\n"
                        f"超时：{arguments.get('timeout', DEFAULT_TIMEOUT)} 秒\n"
                        "完整命令（控制字符转义显示，换行和制表符保留）：\n"
                        f"┌── 命令开始 ──\n{command}\n└── 命令结束 ──\n"
                    )
                elif name == "background_submit":
                    background_timeout = arguments.get("timeout", background_default_timeout)
                    if arguments.get("command") is not None:
                        background_timeout = min(background_timeout, MAX_TIMEOUT)
                        detail = "\n".join(
                            "│ " + line for line in
                            _preview_text(arguments["command"]).split("\n")
                        )
                        task_type = "后台 Bash 命令"
                        block = f"完整命令（控制字符转义显示，换行和制表符保留）：\n┌── 命令开始 ──\n{detail}\n└── 命令结束 ──\n"
                    else:
                        detail = "\n".join(
                            "│ " + line for line in
                            _preview_text(arguments["task"]).split("\n")
                        )
                        task_type = "后台独立分析"
                        block = f"完整任务说明：\n{detail}\n"
                    preview = (
                        f"\n[后台任务确认] {name}\n{risk}\n"
                        f"描述：{_preview_text(arguments['description'], multiline=False)}\n"
                        f"类型：{task_type}\n"
                        f"工作目录：{_preview_text(str(workspace), multiline=False)}\n"
                        f"超时：{background_timeout} 秒\n{block}"
                    )
                elif name == "swarm":
                    role_definitions = arguments.get("roles") or DEFAULT_SWARM_ROLES
                    role_lines = "\n".join(
                        f"│ {role['name']}：工具 {', '.join(role.get('tools', [])) or '未指定'}，"
                        f"可交接 {', '.join(role.get('handoff_to', [])) or '结束'}"
                        for role in role_definitions
                    )
                    preview = (
                        f"\n[团队协作确认] {name}\n{risk}\n"
                        f"描述：{_preview_text(arguments['description'], multiline=False)}\n"
                        f"完整任务：\n{_preview_text(arguments['task'])}\n"
                        f"角色：\n{role_lines}\n"
                        f"最大轮次：{arguments.get('max_rounds', 10)}\n"
                    )
                elif name == "run_verify":
                    preview = (
                        f"\n[验证确认] {name}\n{risk}\n"
                        f"目标：{_preview_text(arguments['target'], multiline=False)}\n"
                        f"运行方式：{arguments.get('runner', 'node')}\n"
                        f"工作目录：{_preview_text(str(workspace), multiline=False)}\n"
                        f"超时：{arguments.get('timeout', DEFAULT_TIMEOUT)} 秒\n"
                    )
                else:
                    parameters = _preview_text(json.dumps(arguments, ensure_ascii=False, indent=2))
                    preview = (
                        f"\n[工具确认] {_preview_text(name, multiline=False)}\n{risk}\n"
                        f"工作目录：{_preview_text(str(workspace), multiline=False)}\n"
                        f"完整参数：\n{parameters}\n"
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

    tool_executor = create_tool_executor(
        confirm=confirm_tool, abort=abort, session_cache=SessionPermissionCache(),
        background_manager=background_manager,
    )

    def on_event(event):
        nonlocal stream_line_open, response_streamed
        if abort.is_set():
            return
        agent = event.get("agent")
        delegated = agent == "delegate"
        background = agent == "background"
        swarm = agent == "swarm"
        prefix = (
            "[delegate] " if delegated
            else "[background] " if background
            else "[swarm] " if swarm
            else ""
        )
        if (delegated or background or swarm) and event["type"] in {"text", "response_start"}:
            return
        if swarm and event["type"] in {
            "tool", "usage", "compact_start", "compact_done", "compact_skipped", "retry",
        }:
            return
        if event["type"] == "background_submitted":
            task = event["task"]
            write(f'[background] 任务 #{task["task_id"]} 已提交：{_preview_text(task["description"], multiline=False)}')
        elif event["type"] == "background_started":
            task = event["task"]
            write(f'[background] 任务 #{task["task_id"]}：状态 {task["status"]}')
        elif event["type"] == "background_finished":
            task = event["task"]
            status = "已完成" if task["status"] == COMPLETED else "已失败"
            write(f'[background] 任务 #{task["task_id"]}：{status}（耗时 {task["elapsed"]:.1f}s）')
        elif event["type"] == "swarm_start":
            roles = "、".join(event.get("roles", []))
            write(f"[swarm] 角色分配：{roles}")
        elif event["type"] == "role_start":
            write(f"[swarm] {_preview_text(event.get('description', ''), multiline=False)}")
        elif event["type"] == "role_complete":
            write(f"[swarm] {_preview_text(event.get('description', ''), multiline=False)}")
        elif event["type"] == "handoff":
            write(f"[swarm] {event.get('from')} 交给 {event.get('to')}")
        elif event["type"] == "swarm_complete":
            write(f"[swarm] 团队协作完成（{event.get('rounds')} 轮）")
        elif event["type"] == "swarm_failed":
            write(f"[swarm] 团队协作失败：{_preview_text(event.get('message', ''), multiline=False)}")
        elif event["type"] == "delegate_start":
            write(f'[delegate] 启动子任务：{_preview_text(event["description"], multiline=False)}')
        elif event["type"] == "delegate_complete":
            write(f'[delegate] 子任务完成，返回结果：{_preview_text(event["description"], multiline=False)}')
        elif event["type"] == "delegate_failed":
            description = _preview_text(event["description"], multiline=False)
            message = _preview_text(event["message"], multiline=False)
            write(f"[delegate] 子任务失败：{description}；{message}")
        elif event["type"] == "response_start":
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
            write(prefix + format_usage(event["record"]))
        elif event["type"] == "tool":
            result = event["result"]
            message = result["message"]
            if event["name"] == "read_file" and type(result.get("eof")) is bool:
                if result["eof"]:
                    message += " 本页已到文件末尾。"
                else:
                    message += f' 下一页：offset={result["next_offset"]}，column={result["next_column"]}。'
            write(f'{prefix}[工具] {event["name"]}：{message}')
        elif event["type"] == "compact_start":
            write(f'{prefix}[压缩] 正在总结旧对话（第 {event["attempt"]} 次）。')
        elif event["type"] == "compact_done":
            write(f'{prefix}[压缩] 已将上下文从 {event["before"]} 字符缩短至 {event["after"]} 字符。')
        elif event["type"] == "compact_skipped":
            write(prefix + "[压缩] 历史较短，无需压缩。")
        elif event["type"] == "retry":
            if event["partial"]:
                write(prefix + "[重试] 上次输出未完成，重新生成。")
            write(f'{prefix}[重试] {event["message"]}；{event["delay"]} 秒后进行第 {event["attempt"]} 次重试。')

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
                background_manager.shutdown()
                return
            text = line.strip()
            if text == "/exit":
                abort.set()
                cancel_confirmation()
                if worker:
                    worker.join(timeout=1)
                background_manager.shutdown()
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
        background_manager.shutdown()
        write("\n查询已停止。")


def _confirmation_risk(name, arguments):
    """提示与权限引擎使用同一份风险分类。"""
    risk = get_risk_level(name, arguments)
    if risk == "high":
        return (
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "⚠ 高风险操作警告：可能具有破坏性的终端命令\n"
            "命令可能修改或删除文件、执行其他程序或发起网络请求。\n"
            "将以当前用户权限运行，请核对下方完整命令和影响范围。\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
    if risk == "low":
        return "风险等级：低风险（只读；当前权限规则要求确认）。"
    if name == "write_file":
        return "风险等级：中风险（写入文件）。"
    if name == "bash":
        return "风险等级：中风险（可能修改文件、环境或执行程序；请核对完整命令）。"
    return "风险等级：中风险（工具调用，执行前需要确认）。"


def _preview_text(text, *, multiline=True):
    """保持完整文本可见，防止终端控制序列覆盖确认提示。"""
    return "".join(
        character if character.isprintable() or (multiline and character in "\n\t")
        else f"\\u{ord(character):04x}" if ord(character) <= 0xffff
        else f"\\U{ord(character):08x}"
        for character in text
    )
