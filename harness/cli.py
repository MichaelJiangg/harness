"""终端交互和本地命令；模型请求在后台运行，等待时仍可查看 /cost。"""

import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .background import BackgroundManager, COMPLETED
from .client import DEFAULT_MODEL
from .config import get_settings
from .engine import QueryAborted, QueryState, SYSTEM_PROMPT, compact_history, query_loop
from .memory import (
    MemoryStore, format_record, has_memory_candidates, memory_system_prompt, summarize_session,
)
from .notes import NotesStore, notes_system_prompt
from .permissions import PermissionPolicy, SessionPermissionCache, get_risk_level
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
/mode ask|auto [目录]  切换权限模式；auto 模式信任当前或指定工作目录，危险操作仍确认
/memory [list|show <id>|delete <id> --yes|clear --yes]  查看和管理本地会话记忆
/notes [append <text>|replace --yes <text>|clear --yes]  查看和编辑项目长期笔记
/activity [latest|all|clear]  查看后台工具、请求和编排活动
/cost  查看本次会话 token、USD 预估费用及逐请求明细
/compact  压缩旧对话，保留最近几轮
/help  查看帮助
/exit  退出并停止后续模型和工具调用"""

PRODUCT_NAME = "Delin Harness"
PRODUCT_SUBTITLE = "Powered by Deepseek"
BRIEF_HELP = """直接输入任务开始：

1. 帮我调研 Personal Agent 的国内外竞品，包括 MUSE、Today 等。
2. 帮我按照五看三定，做一份产品规划。
3. 帮我做 Agent 的发布页产品设计。

核心能力
文件读写 · 命令执行 · 网页读取 · 长期记忆

常用命令

/mode auto [目录]   减少重复确认
/memory             查看会话记忆
/notes              查看项目笔记
/cost               查看用量
/help               完整帮助

输入 /activity 查看后台执行记录
输入 /exit 退出"""

ACTIVITY_DEFAULT_COUNT = 50
ACTIVITY_MAX_EVENTS = 500


def run_cli(client, *, ledger=None, input_stream=None, output=None, error_output=None,
            character_delay=None, memory_enabled=False, memory_store=None,
            notes_enabled=False, notes_store=None):
    if character_delay is None:
        character_delay = get_settings()["display"]["character_delay"]
    ledger = ledger if ledger is not None else UsageLedger()
    input_stream = input_stream if input_stream is not None else sys.stdin
    output = output if output is not None else sys.stdout
    error_output = error_output if error_output is not None else sys.stderr
    terminal = input_stream.isatty() and output.isatty()
    console = Console(
        file=output, force_terminal=terminal, highlight=True, color_system="standard",
        no_color=False,
    )
    error_console = Console(
        file=error_output, force_terminal=terminal, highlight=False, color_system="standard",
        no_color=False,
    )
    memory_store = memory_store if memory_store is not None else MemoryStore(Path.cwd())
    if not isinstance(memory_store, MemoryStore):
        raise ValueError("memory_store 必须是 MemoryStore。")
    notes_store = notes_store if notes_store is not None else NotesStore(Path.cwd())
    if not isinstance(notes_store, NotesStore):
        raise ValueError("notes_store 必须是 NotesStore。")
    memory_load_error = None
    memory_records = []
    if memory_enabled:
        try:
            memory_records = memory_store.records()
        except Exception as error:
            memory_load_error = str(error)
    notes_load_error = None
    notes_content = ""
    if notes_enabled:
        try:
            notes_content = notes_store.read()
        except Exception as error:
            notes_load_error = str(error)
    messages = [{
        "role": "system",
        "content": memory_system_prompt(
            notes_system_prompt(SYSTEM_PROMPT, notes_content),
            memory_records,
        ),
    }]
    abort = Event()
    background_manager = BackgroundManager(**get_settings()["background"])
    background_default_timeout = get_settings()["background"]["default_timeout"]
    security_settings = get_settings()["security"]
    permission_mode = security_settings["mode"]
    auto_directories = list(security_settings["auto_directories"])
    output_lock = Lock()
    confirmation_lock = Lock()
    activity_entries = []
    activity_next_id = 1
    pending_confirmation = None
    worker = None
    turn = 0
    stream_line_open = False
    response_streamed = False
    stream_buffer = ""
    stream_live = None
    prompt_shown = False
    input_closed = False

    def show_prompt():
        nonlocal prompt_shown
        with output_lock:
            if terminal and not prompt_shown and not input_closed and not abort.is_set():
                console.print(Text("你", style="bold cyan"), end=" ")
                console.print(Text(">", style="bold cyan"), end=" ")
                prompt_shown = True

    def stop_stream_live():
        nonlocal stream_live
        if stream_live is not None:
            try:
                stream_live.stop()
            except Exception:
                pass
            stream_live = None

    def end_stream_line():
        nonlocal stream_line_open
        if terminal:
            stop_stream_live()
        elif stream_line_open:
            print(file=output, flush=True)
            stream_line_open = False

    def write(text, *, error=False, style=None):
        with output_lock:
            end_stream_line()
            if terminal:
                target = error_console if error else console
                resolved_style = style or ("bold red" if error else None)
                target.print(
                    Text(_preview_text(str(text)), style=resolved_style, overflow="fold"),
                    soft_wrap=True,
                )
            else:
                print(text, file=error_output if error else output, flush=True)

    def write_markdown(title, content):
        with output_lock:
            if terminal:
                stop_stream_live()
                console.print(Panel(
                    Markdown(_preview_text(content), code_theme="monokai", justify="left"),
                    title=title,
                    border_style="cyan",
                    expand=False,
                    padding=(0, 1),
                ))
            else:
                end_stream_line()
                print(f"\nDeepSeek > {content}", file=output, flush=True)

    def record_activity(message, kind):
        nonlocal activity_entries, activity_next_id
        if not message:
            return
        with output_lock:
            activity_entries.append({
                "id": str(activity_next_id),
                "kind": kind,
                "message": _preview_text(message, multiline=True),
            })
            activity_next_id += 1
            if len(activity_entries) > ACTIVITY_MAX_EVENTS:
                activity_entries = activity_entries[-ACTIVITY_MAX_EVENTS:]

    def render_activity(entries):
        if not entries:
            write("暂无后台活动记录。", style="dim")
            return
        if terminal:
            with output_lock:
                stop_stream_live()
                for entry in entries:
                    console.print(Panel(
                        Text(entry["message"], style="default", overflow="fold"),
                        title=f"#{entry['id']} {entry['kind']}",
                        border_style="dim",
                        expand=False,
                        padding=(0, 1),
                    ))
        else:
            write("\n".join(
                f"[activity #{entry['id']}] {entry['kind']}：{entry['message']}"
                for entry in entries
            ))

    def handle_activity_command(text):
        parts = shlex.split(text)
        if len(parts) == 1:
            render_activity(activity_entries[-ACTIVITY_DEFAULT_COUNT:])
            return
        if parts[1:] == ["all"]:
            render_activity(activity_entries)
            return
        if parts[1:] == ["clear"]:
            with output_lock:
                activity_entries.clear()
            write("已清空后台活动记录。")
            return
        target = parts[1]
        matches = [entry for entry in activity_entries if entry["id"] == target]
        render_activity(matches)
        if not matches:
            write(f"未找到活动记录 #{target}。", style="yellow")

    def start_stream():
        nonlocal response_streamed, stream_buffer, stream_line_open, stream_live
        response_streamed = False
        if not terminal:
            stream_line_open = False
            return
        with output_lock:
            stop_stream_live()
            stream_buffer = ""
            stream_live = Live(
                Panel(
                    Markdown("", code_theme="monokai"),
                    title="Assistant",
                    border_style="cyan",
                    padding=(0, 1),
                ),
                console=console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            stream_live.start()

    def append_stream(fragment):
        nonlocal response_streamed, stream_buffer, stream_line_open
        if not terminal:
            with output_lock:
                if not stream_line_open:
                    print("\nDeepSeek > " if not response_streamed else "DeepSeek > ",
                          end="", file=output, flush=True)
                print(fragment, end="", file=output, flush=True)
                stream_line_open = True
                response_streamed = True
            return
        stream_buffer += fragment
        response_streamed = True
        with output_lock:
            if stream_live is not None:
                try:
                    stream_live.update(Panel(
                        Markdown(stream_buffer, code_theme="monokai", justify="left"),
                        title="Assistant",
                        border_style="cyan",
                        padding=(0, 1),
                    ))
                    stream_live.refresh()
                except Exception:
                    pass

    def save_memory_summary():
        nonlocal messages
        if not memory_enabled or not has_memory_candidates(messages):
            return
        try:
            result = summarize_session(client, messages, model=DEFAULT_MODEL)
            response = result["response"]
            if response is not None:
                payload = response if isinstance(response, dict) else {}
                ledger.record(
                    usage=payload.get("usage"), model=payload.get("model", DEFAULT_MODEL),
                    turn=turn, created=payload.get("created"),
                )
            record = memory_store.add(
                result["summary"], topics=result["topics"],
                message_count=sum(
                    1 for message in messages if message.get("role") in {"user", "assistant"}
                ),
            )
            write(f"[memory] 已保存本次会话摘要 #{record['id']}。")
        except Exception as error:
            write(f"[memory] 保存会话摘要失败：{error}", error=True)

    def handle_memory_command(text):
        try:
            parts = shlex.split(text)
        except ValueError:
            write("用法：/memory [list|show <id>|delete <id> --yes|clear --yes]")
            return
        if len(parts) == 1 or parts[1:] == ["list"]:
            try:
                records = memory_store.records()
            except Exception as error:
                write(f"[memory] 读取记忆失败：{error}", error=True)
                return
            if not records:
                write("暂无本地记忆。")
                return
            write("本地记忆：\n" + "\n".join(format_record(record) for record in records))
            return
        if parts[1] == "show" and len(parts) == 3:
            try:
                record = memory_store.get(parts[2])
            except Exception as error:
                write(f"[memory] 读取记忆失败：{error}", error=True)
                return
            if record is None:
                write(f"未找到记忆 #{parts[2]}。")
            else:
                write(format_record(record))
            return
        if parts[1] == "delete" and len(parts) == 4 and parts[3] == "--yes":
            try:
                deleted = memory_store.delete(parts[2])
            except Exception as error:
                write(f"[memory] 删除记忆失败：{error}", error=True)
                return
            write(f"已删除记忆 #{parts[2]}。" if deleted else f"未找到记忆 #{parts[2]}。")
            return
        if parts[1] == "delete":
            write("删除记忆需要显式确认：/memory delete <id> --yes")
            return
        if parts[1] == "clear" and len(parts) == 3 and parts[2] == "--yes":
            try:
                count = memory_store.clear()
            except Exception as error:
                write(f"[memory] 清空记忆失败：{error}", error=True)
                return
            write(f"已清空 {count} 条本地记忆。")
            return
        if parts[1] == "clear":
            write("清空全部记忆需要显式确认：/memory clear --yes")
            return
        write("用法：/memory [list|show <id>|delete <id> --yes|clear --yes]")

    def handle_notes_command(text):
        try:
            parts = shlex.split(text)
        except ValueError:
            write("用法：/notes [append <text>|replace --yes <text>|clear --yes]")
            return
        if len(parts) == 1:
            try:
                content = notes_store.read()
            except Exception as error:
                write(f"[notes] 读取项目笔记失败：{error}", error=True)
                return
            if not content.strip():
                write("HARNESS.md 暂无内容。")
            else:
                write("HARNESS.md：\n" + _preview_text(content))
            return
        if parts[1] == "append" and len(parts) >= 3:
            content = shlex.join(parts[2:])
            try:
                appended = notes_store.append(content)
            except Exception as error:
                write(f"[notes] 追加项目笔记失败：{error}", error=True)
                return
            write(f"已追加项目笔记（{appended} 字节）。")
            return
        if parts[1] == "append":
            write("用法：/notes append <text>")
            return
        if parts[1] == "replace" and len(parts) >= 4 and parts[2] == "--yes":
            content = shlex.join(parts[3:])
            try:
                written = notes_store.replace(content)
            except Exception as error:
                write(f"[notes] 替换项目笔记失败：{error}", error=True)
                return
            write(f"已替换项目笔记（{written} 字节）。")
            return
        if parts[1] == "replace":
            write("替换全部项目笔记需要显式确认：/notes replace --yes <text>")
            return
        if parts[1] == "clear" and len(parts) == 3 and parts[2] == "--yes":
            try:
                notes_store.clear()
            except Exception as error:
                write(f"[notes] 清空项目笔记失败：{error}", error=True)
                return
            write("已清空 HARNESS.md。")
            return
        if parts[1] == "clear":
            write("清空项目笔记需要显式确认：/notes clear --yes")
            return
        write("用法：/notes [append <text>|replace --yes <text>|clear --yes]")

    def confirm_tool(name, arguments, workspace):
        nonlocal pending_confirmation
        label = {
            "write_file": "写入", "bash": "命令执行", "background_submit": "后台任务",
            "swarm": "团队协作", "run_verify": "验证",
            "notes_append": "追加项目笔记", "notes_replace": "替换项目笔记",
            "web_fetch": "网页读取",
            "web_search": "网页搜索",
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
                elif name == "web_fetch":
                    preview = (
                        f"\n[网页读取确认] {name}\n{risk}\n"
                        f"目标：{_preview_text(arguments['url'], multiline=False)}\n"
                        "只读取公开 http/https 网页，禁止私网、重定向和超过 1 MiB 的响应。\n"
                    )
                elif name == "web_search":
                    preview = (
                        f"\n[网页搜索确认] {name}\n{risk}\n"
                        f"搜索词：{_preview_text(arguments['query'], multiline=False)}\n"
                        f"返回条数：{arguments.get('max_results', 5)}\n"
                        "使用 Tavily 搜索公开网页；需要本地配置 TAVILY_API_KEY。\n"
                    )
                elif name in {"notes_append", "notes_replace"}:
                    candidate = Path(workspace) / "HARNESS.md"
                    path = candidate.resolve()
                    existed = path.exists()
                    action = "替换 HARNESS.md 的全部内容" if name == "notes_replace" else "追加到 HARNESS.md"
                    content = "\n".join(
                        "│ " + line for line in
                        _preview_text(arguments["content"]).split("\n")
                    )
                    preview = (
                        f"\n[项目笔记确认] {name}\n{risk}\n"
                        f"目标：{_preview_text('HARNESS.md', multiline=False)}\n"
                        f"操作：{action}\n"
                        "完整内容（控制字符转义显示，换行和制表符保留）：\n"
                        f"┌── 笔记内容开始 ──\n{content}\n└── 笔记内容结束 ──\n"
                    )
                else:
                    parameters = _preview_text(json.dumps(arguments, ensure_ascii=False, indent=2))
                    preview = (
                        f"\n[工具确认] {_preview_text(name, multiline=False)}\n{risk}\n"
                        f"工作目录：{_preview_text(str(workspace), multiline=False)}\n"
                        f"完整参数：\n{parameters}\n"
                    )
                pending_confirmation = request
                confirmation_style = "red" if risk.startswith("!!!!!!!!!!!!!!!!") else "yellow"
                write(
                    preview + f"[确认] 输入 y 批准本次{label}，n 或直接回车拒绝；/cost、/help、/exit 仍可使用。",
                    style=confirmation_style,
                )
            request["done"].wait()
            with confirmation_lock:
                if not request["approved"] or input_closed or abort.is_set():
                    return False
                if name in {"write_file", "notes_append", "notes_replace"} and (
                    candidate.resolve() != path or path.exists() != existed
                ):
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

    session_cache = SessionPermissionCache()

    def build_tool_executor():
        settings = get_settings()
        kwargs = {
            "confirm": confirm_tool, "abort": abort, "session_cache": session_cache,
            "background_manager": background_manager,
        }
        if permission_mode != "ask" or auto_directories:
            kwargs["permissions"] = PermissionPolicy(
                mode=permission_mode, auto_directories=auto_directories,
                **settings["permissions"],
            )
        return create_tool_executor(**kwargs)

    tool_executor = build_tool_executor()

    def on_event(event):
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
        agent_style = (
            "blue" if delegated
            else "magenta" if background
            else "green" if swarm
            else None
        )
        if (delegated or background or swarm) and event["type"] in {"text", "response_start"}:
            return
        hidden_types = {
            "tool", "usage", "compact_start", "compact_done", "compact_skipped",
        }
        if event["type"] in hidden_types:
            message = _activity_message(event)
            if message:
                record_activity(message, event["type"])
            return
        message = _activity_message(event)
        if message:
            record_activity(message, event["type"])
        if event["type"] == "background_submitted":
            task = event["task"]
            write(
                f'[background] 任务 #{task["task_id"]} 已提交：'
                f'{_preview_text(task["description"], multiline=False)}',
                style=agent_style,
            )
        elif event["type"] == "background_started":
            task = event["task"]
            write(
                f'[background] 任务 #{task["task_id"]}：状态 {task["status"]}',
                style=agent_style,
            )
        elif event["type"] == "background_finished":
            task = event["task"]
            status = "已完成" if task["status"] == COMPLETED else "已失败"
            write(
                f'[background] 任务 #{task["task_id"]}：{status}（耗时 {task["elapsed"]:.1f}s）',
                style="green" if status == "已完成" else "red",
            )
        elif event["type"] == "swarm_start":
            roles = "、".join(event.get("roles", []))
            write(f"[swarm] 角色分配：{roles}", style=agent_style)
        elif event["type"] == "role_start":
            write(
                f"[swarm] {_preview_text(event.get('description', ''), multiline=False)}",
                style=agent_style,
            )
        elif event["type"] == "role_complete":
            write(
                f"[swarm] {_preview_text(event.get('description', ''), multiline=False)}",
                style=agent_style,
            )
        elif event["type"] == "handoff":
            write(f"[swarm] {event.get('from')} 交给 {event.get('to')}", style=agent_style)
        elif event["type"] == "swarm_complete":
            write(f"[swarm] 团队协作完成（{event.get('rounds')} 轮）", style="green")
        elif event["type"] == "swarm_failed":
            write(
                f"[swarm] 团队协作失败：{_preview_text(event.get('message', ''), multiline=False)}",
                style="red",
            )
        elif event["type"] == "delegate_start":
            write(
                f'[delegate] 启动子任务：{_preview_text(event["description"], multiline=False)}',
                style=agent_style,
            )
        elif event["type"] == "delegate_complete":
            write(
                f'[delegate] 子任务完成，返回结果：'
                f'{_preview_text(event["description"], multiline=False)}',
                style=agent_style,
            )
        elif event["type"] == "delegate_failed":
            description = _preview_text(event["description"], multiline=False)
            message_text = _preview_text(event["message"], multiline=False)
            write(f"[delegate] 子任务失败：{description}；{message_text}", style="red")
        elif event["type"] == "response_start":
            start_stream()
        elif event["type"] == "text":
            fragments = event["text"] if terminal else (event["text"],)
            for fragment in fragments:
                if abort.is_set():
                    return
                append_stream(fragment)
                if terminal and character_delay > 0 and abort.wait(character_delay):
                    return
        elif event["type"] == "retry":
            if event["partial"]:
                write(prefix + "[重试] 上次输出未完成，重新生成。", style="yellow")
            write(
                f'{prefix}[重试] {event["message"]}；{event["delay"]} 秒后进行第 {event["attempt"]} 次重试。',
                style="yellow",
            )

    def run_query(state, *, compact=False):
        nonlocal messages, response_streamed
        try:
            if compact:
                if compact_history(state, force=True) and not abort.is_set():
                    messages = state.messages
                return
            answer = query_loop(state)
            if not abort.is_set():
                messages = state.messages
                if not response_streamed:
                    write_markdown("Assistant", answer)
        except QueryAborted:
            pass
        except Exception as error:
            if not abort.is_set():
                write(f"错误：{error}", error=True)
        finally:
            with output_lock:
                stop_stream_live()
            show_prompt()

    if terminal:
        with output_lock:
            console.print(Panel(
                Group(
                    Text(PRODUCT_NAME, style="bold cyan", justify="center"),
                    Text(PRODUCT_SUBTITLE, style="dim", justify="center"),
                ),
                border_style="cyan",
                expand=False,
            ))
            console.print(Text(BRIEF_HELP, style="dim"), soft_wrap=True)
    else:
        write(f"{PRODUCT_NAME} · {PRODUCT_SUBTITLE}\n{BRIEF_HELP}")
    if memory_load_error:
        write(f"[memory] 启动时未加载本地记忆：{memory_load_error}", error=True)
    if notes_load_error:
        write(f"[notes] 启动时未加载项目笔记：{notes_load_error}", error=True)
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
                save_memory_summary()
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
                save_memory_summary()
                return
            if text == "/help":
                write(HELP)
            elif text.startswith("/mode") and worker and worker.is_alive():
                write("上一轮查询进行中，暂时无法切换权限模式；期间可以输入 /cost。")
            elif text.startswith("/mode"):
                parts = text.split()
                if len(parts) not in {2, 3} or parts[1] not in {"ask", "auto"}:
                    write("用法：/mode ask|auto [目录]")
                else:
                    try:
                        permission_mode = parts[1]
                        auto_directories = [parts[2]] if len(parts) == 3 else []
                        tool_executor = build_tool_executor()
                        suffix = f"：{auto_directories[0]}" if auto_directories else "：当前工作目录"
                        write(f"权限模式已切换为 {permission_mode}{suffix}。")
                    except ValueError as error:
                        write(f"错误：{error}", error=True)
            elif text.startswith("/memory") and worker and worker.is_alive():
                write("上一轮查询进行中，暂时无法管理记忆；期间可以输入 /cost。")
            elif text.startswith("/memory"):
                handle_memory_command(text)
            elif text.startswith("/notes") and worker and worker.is_alive():
                write("上一轮查询进行中，暂时无法管理项目笔记；期间可以输入 /cost。")
            elif text.startswith("/notes"):
                handle_notes_command(text)
            elif text.startswith("/activity"):
                handle_activity_command(text)
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


def _tool_result_summary(name, result):
    """终端只显示结构摘要，不重复输出文件正文或子任务报告。"""
    message = _preview_text(str(result.get("message", "")), multiline=False)
    if name == "read_file" and type(result.get("eof")) is bool:
        if result["eof"]:
            message += " 本页已到文件末尾。"
        else:
            message += f' 下一页：offset={result["next_offset"]}，column={result["next_column"]}。'
    if name == "grep":
        message += f' 返回 {result.get("returned_count", 0)} 条匹配。'
    if name == "bash":
        message += (
            f' 退出码：{result.get("exit_code")}'
            f'，超时：{result.get("timed_out", False)}'
            f'，取消：{result.get("cancelled", False)}'
        )
        stdout = _preview_text(str(result.get("stdout", "")), multiline=True)[:1200]
        stderr = _preview_text(str(result.get("stderr", "")), multiline=True)[:1200]
        if stdout.strip():
            message += f'\nstdout：\n{stdout}'
        if stderr.strip():
            message += f'\nstderr：\n{stderr}'
    return message


def _activity_message(event):
    """把执行过程转成可延迟查看的短消息，不回传正文或密钥。"""
    kind = event.get("type")
    if kind == "tool":
        name = event.get("name", "tool")
        summary = _tool_result_summary(name, event.get("result", {}))
        arguments = _preview_text(
            json.dumps(event.get("arguments", {}), ensure_ascii=False, indent=2)
        )[:4000]
        return f"tool:{name}：{summary}\narguments：\n{arguments}"
    if kind == "usage":
        record = event.get("record", {})
        return f"usage：{format_usage(record)}"
    if kind == "compact_start":
        return f"compact：正在总结旧对话（第 {event.get('attempt', 1)} 次）。"
    if kind == "compact_done":
        return (
            f"compact：已将上下文从 {event.get('before')} 字符"
            f"缩短至 {event.get('after')} 字符。"
        )
    if kind == "compact_skipped":
        return "compact：历史较短，无需压缩。"
    if kind == "retry":
        message = event.get("message", "")
        return (
            f"retry：{message}；{event.get('delay')} 秒后进行"
            f"第 {event.get('attempt', 1)} 次重试。"
        )
    if kind in {"delegate_start", "delegate_complete", "delegate_failed"}:
        return f"{kind}：{event.get('description', '')}"
    if kind in {"background_submitted", "background_started", "background_finished"}:
        task = event.get("task", {})
        return f"{kind}：任务 #{task.get('task_id')} {task.get('status', '')}"
    if kind in {"swarm_start", "role_start", "role_complete", "handoff",
                "swarm_complete", "swarm_failed"}:
        return f"{kind}：{event.get('description', '') or event.get('from', '') or event.get('to', '')}"
    return ""


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
