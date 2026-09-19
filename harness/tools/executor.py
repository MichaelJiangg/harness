"""Tool Executor：校验调用、检查权限、分发执行，并统一结果与错误。"""

from copy import deepcopy
from functools import partial
from pathlib import Path
from uuid import uuid4

from ..audit import AuditError, PermissionAuditLog
from ..background import BackgroundManager
from ..config import get_settings
from ..permissions import PermissionDecision, PermissionPolicy, SessionPermissionCache, build_denial_message
from .registry import REGISTRY, ToolRegistry


class ToolError(Exception):
    """可安全回传给模型的工具错误。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def create_tool_executor(workspace=None, *, confirm=None, abort=None, permissions=None,
                         session_cache=None, audit=None, registry=None, background_manager=None):
    """固定会话启动目录及权限快照，后续调用不随环境改变。"""
    root = (Path.cwd() if workspace is None else Path(workspace)).resolve()
    policy = _get_permissions(permissions)
    if session_cache is not None and not isinstance(session_cache, SessionPermissionCache):
        raise ValueError("session_cache 必须是 SessionPermissionCache。")
    if background_manager is not None and not isinstance(background_manager, BackgroundManager):
        raise ValueError("background_manager 必须是 BackgroundManager。")
    selected = REGISTRY if registry is None else registry
    if not isinstance(selected, ToolRegistry):
        raise ValueError("registry 必须是 ToolRegistry。")
    # 固定本执行器开放的工具及参数说明，后续注册不能扩展当前会话的能力。
    registry = ToolRegistry()
    for item in selected.definitions():
        definition, handler = selected.get(item["function"]["name"])
        registry.register(deepcopy(definition), handler)
    audit = audit if audit is not None else PermissionAuditLog(root)
    return partial(execute_tool, workspace=root, confirm=confirm, abort=abort,
                   permissions=policy, session_cache=session_cache, audit=audit,
                   registry=registry, background_manager=background_manager)


def execute_tool(name, arguments, *, workspace=None, confirm=None, abort=None, permissions=None,
                 session_cache=None, audit=None, registry=None, background_manager=None):
    def error(code, message):
        if code.startswith(("permission_", "confirmation_")) or code in {"audit_failed", "access_denied"}:
            message = build_denial_message(name, arguments, message)
        return {
            "status": "error", "executed": False,
            "tool": name if isinstance(name, str) else None,
            "code": code, "message": message,
        }

    if not isinstance(name, str):
        return error("unknown_tool", "未知工具。")
    registry = REGISTRY if registry is None else registry
    registered = registry.get(name)
    if registered is None:
        return error("unknown_tool", "未知工具。")
    definition, handler = registered
    try:
        # 将权限匹配、预览和执行绑定到同一份参数快照。
        arguments = deepcopy(arguments)
        _validate(arguments, definition.input_schema)
        root = (Path.cwd() if workspace is None else Path(workspace)).resolve()
        if abort is not None and abort.is_set():
            return error("cancelled", "查询已停止，未执行工具。")
        policy = _get_permissions(permissions)
        audit = audit if audit is not None else PermissionAuditLog(root)
        call_id = uuid4().hex
        confirmation = None
        approved_directory = None

        def record(result, *, event="decision"):
            audit.record(name, arguments, result.decision, result.matched_rule,
                         risk=result.risk, confirmation=confirmation, event=event, call_id=call_id)

        def check_permissions():
            try:
                result = policy.evaluate(name, arguments, workspace=root)
                if name == "background_submit" and isinstance(arguments.get("command"), str):
                    command_result = policy.evaluate(
                        "bash", {"command": arguments["command"]}, workspace=root,
                    )
                    if command_result.decision == "deny":
                        return command_result
                    if command_result.decision == "ask" and result.decision == "allow":
                        return PermissionDecision(
                            "ask", result.risk, "background:bash_guard",
                            "后台命令不属于明确只读操作，仍须用户确认。",
                        )
                # 先检查完整策略，再用会话授权把 ask 转为 allow；deny 永不被覆盖。
                if (result.decision == "ask" and session_cache is not None
                        and session_cache.is_approved(name, arguments, workspace=root)):
                    matched = "session:run_verify_directory" if name == "run_verify" \
                        else "session:write_directory"
                    action = "验证" if name == "run_verify" else "写入"
                    return PermissionDecision("allow", result.risk, matched,
                                              f"本会话已授权该目录{action}。")
                if (name == "run_verify" and result.decision == "allow"
                        and session_cache is not None
                        and session_cache.is_approved(name, arguments, workspace=root)):
                    return PermissionDecision(
                        "allow", result.risk, "session:run_verify_directory",
                        "本会话已授权该目录验证。",
                    )
                if (name == "run_verify" and result.decision == "allow"
                        and result.matched_rule != "session:auto"):
                    return PermissionDecision(
                        "ask", result.risk, "verification:confirm",
                        "验证脚本可能执行任意本地代码，仍须用户确认。",
                    )
                return result
            except (OSError, RuntimeError, ValueError):
                record(PermissionDecision("deny", "unknown", "error:path_check", "路径检查失败。"))
                raise ToolError("permission_check_failed", "无法完成权限规则检查，未执行工具，请检查路径状态。") from None

        def file_target():
            path = arguments.get("path")
            if name not in {"read_file", "write_file"} or not path or "\x00" in path:
                return None
            try:
                return (root / path).resolve()
            except (OSError, RuntimeError, ValueError):
                record(PermissionDecision("deny", "unknown", "error:path_check", "路径检查失败。"))
                raise ToolError("permission_check_failed", "无法解析文件目标，未执行工具。") from None

        def verification_target():
            path = arguments.get("target")
            if name != "run_verify" or not path or "\x00" in path:
                return None
            try:
                return (root / path).resolve()
            except (OSError, RuntimeError, ValueError):
                record(PermissionDecision("deny", "unknown", "error:path_check", "路径检查失败。"))
                raise ToolError("permission_check_failed", "无法解析验证目标，未执行工具。") from None

        def notes_target():
            if name not in {"notes_append", "notes_replace"}:
                return None
            try:
                return (root / "HARNESS.md").resolve()
            except (OSError, RuntimeError, ValueError):
                record(PermissionDecision("deny", "unknown", "error:path_check", "路径检查失败。"))
                raise ToolError("permission_check_failed", "无法解析项目笔记目标，未执行工具。") from None

        initial_target = file_target()
        initial_verification_target = verification_target()
        initial_notes_target = notes_target()
        decision = check_permissions()
        if decision.matched_rule in {"session:write_directory", "session:run_verify_directory"}:
            confirmation = "remembered"
        elif decision.decision == "ask":
            confirmation = "pending"
        # 初始决策先记入审计；若落盘失败，外层捕获 AuditError 并停止执行。
        record(decision)
        if decision.decision == "deny":
            return error("permission_denied", decision.reason)
        if decision.decision == "ask":
            # 决策只是「需要确认」，必须由本地入口取得批准，模型不能代替用户。
            if confirm is None:
                confirmation = "unavailable"
                record(decision, event="confirmation")
                return error("confirmation_required", "此工具需要用户本地确认，当前未提供确认入口。")
            try:
                approved = confirm(name, deepcopy(arguments), root)
            except Exception:
                confirmation = "failed"
                record(decision, event="confirmation")
                return error("confirmation_failed", "未能取得用户本地确认，未执行工具。")
            if abort is not None and abort.is_set():
                confirmation = "cancelled"
                record(decision, event="confirmation")
                return error("cancelled", "查询已停止，未执行工具。")
            confirmation = "approved" if approved is True else "rejected"
            record(decision, event="confirmation")
            # 严格要求布尔值 True，字符串 "y"、数字 1 等都不是有效批准。
            if approved is not True:
                return error("confirmation_denied", "用户未批准本次操作，未执行工具；请等待用户的新要求，不要自行重试。")
            if (name == "write_file" and session_cache is not None and initial_target is not None
                    and initial_target.is_relative_to(root)):
                approved_directory = str(initial_target.parent.relative_to(root))
            if (name == "run_verify" and session_cache is not None
                    and initial_verification_target is not None
                    and initial_verification_target.is_relative_to(root)):
                candidate = initial_verification_target
                approved_directory = str((candidate if candidate.is_dir() else candidate.parent)
                                         .relative_to(root))
        current = check_permissions()
        record(current, event="before_execute")
        # 日志落盘也可能耗时，必须在落盘之后再次核对规则与实际目标。
        current = check_permissions()
        if current.decision == "deny":
            record(current, event="guard_denial")
            return error("permission_denied", current.reason)
        if (current != decision or file_target() != initial_target
                or verification_target() != initial_verification_target
                or notes_target() != initial_notes_target):
            record(PermissionDecision("deny", current.risk, "guard:target_changed", "权限或目标发生变化。"),
                   event="guard_denial")
            return error("permission_changed", "路径对应的权限条件已变化，未执行工具；请重新发起调用。")
        if abort is not None and abort.is_set():
            return error("cancelled", "查询已停止，未执行工具。")
        options = {"abort": abort} if definition.supports_cancellation else {}
        if name == "delegate":
            # 延迟导入避免工具自动发现与查询引擎循环导入；上下文来自本地绑定。
            from ..orchestration import get_current_query, run_delegate

            parent = get_current_query()
            if parent is not None:
                child_registry = ToolRegistry()
                for item in parent.tools:
                    function = item.get("function", {}) if isinstance(item, dict) else {}
                    child_name = function.get("name") if isinstance(function, dict) else None
                    if not isinstance(child_name, str) or child_name in {
                        "delegate", "background_submit", "background_check", "swarm",
                    }:
                        continue
                    child_tool = registry.get(child_name)
                    if child_tool is not None and child_registry.get(child_name) is None:
                        child_registry.register(*child_tool)
                # 模型说明与实际执行使用同一个子注册表，不能只隐藏递归入口。
                options["runner"] = partial(
                    run_delegate,
                    executor_factory=partial(
                        create_tool_executor, root, confirm=confirm, permissions=policy,
                        session_cache=session_cache, audit=audit, registry=child_registry,
                    ),
                    tools=child_registry.definitions(),
                )
        elif name == "background_submit":
            from ..orchestration import get_current_query, run_background_analysis

            parent = get_current_query()

            def submit_runner(*, description, command, task, timeout):
                if background_manager is None:
                    raise ToolError("background_unavailable", "当前会话没有后台任务管理器。")

                def manager_event(task_id, event):
                    if parent is not None:
                        parent.on_event({**event, "agent": "background", "task_id": task_id})

                if command is not None:
                    from .bash import MAX_TIMEOUT, execute as execute_bash

                    return background_manager.submit(
                        lambda stop_event: execute_bash(
                            {"command": command, "timeout": min(timeout, MAX_TIMEOUT)},
                            root, abort=stop_event,
                        ),
                        description, timeout=timeout, on_event=manager_event,
                    )
                if parent is None:
                    raise ToolError("background_unavailable",
                                    "后台分析需要当前查询上下文，无法启动任务。")
                child_registry = ToolRegistry()
                for item in parent.tools:
                    function = item.get("function", {}) if isinstance(item, dict) else {}
                    child_name = function.get("name") if isinstance(function, dict) else None
                    if not isinstance(child_name, str) or child_name in {
                        "delegate", "background_submit", "background_check", "swarm",
                    }:
                        continue
                    child_tool = registry.get(child_name)
                    if child_tool is not None and child_registry.get(child_name) is None:
                        child_registry.register(*child_tool)
                child_executor = partial(
                    create_tool_executor, root, confirm=None, permissions=policy,
                    session_cache=session_cache, audit=audit, registry=child_registry,
                )

                def child_event(event):
                    if parent is not None:
                        parent.on_event({**event, "agent": "background"})

                return background_manager.submit(
                    lambda stop_event: run_background_analysis(
                        parent=parent, task=task, tools=child_registry.definitions(),
                        executor_factory=child_executor, stop_event=stop_event,
                        on_event=child_event, turn=parent.turn,
                    ),
                    description, timeout=timeout, on_event=manager_event,
                )

            options["runner"] = submit_runner
        elif name == "background_check":
            if background_manager is not None:
                options["runner"] = background_manager.check
        elif name == "swarm":
            from ..orchestration import get_current_query, run_swarm

            parent = get_current_query()

            def make_executor(tool_names, abort_event):
                child_registry = ToolRegistry()
                visible_names = {
                    item.get("function", {}).get("name")
                    for item in (parent.tools if parent is not None else [])
                    if isinstance(item, dict) and isinstance(item.get("function", {}), dict)
                }
                for tool_name in tool_names:
                    if tool_name in {"delegate", "background_submit", "background_check", "swarm"}:
                        raise ToolError("swarm_tool_denied",
                                        f"团队角色不能使用编排工具 {tool_name}。")
                    if tool_name not in visible_names:
                        raise ToolError("swarm_tool_hidden",
                                        f"角色请求了主查询未开放的工具：{tool_name}。")
                    child_tool = registry.get(tool_name)
                    if child_tool is None:
                        raise ToolError("swarm_tool_unknown", f"角色工具不存在：{tool_name}。")
                    if child_registry.get(tool_name) is None:
                        child_registry.register(*child_tool)
                return create_tool_executor(
                    root, confirm=confirm, abort=abort_event, permissions=policy,
                    session_cache=session_cache, audit=audit, registry=child_registry,
                )

            def swarm_runner(*, description, task, roles, max_rounds):
                if parent is None:
                    raise ToolError("swarm_unavailable",
                                    "团队协作需要当前查询上下文，无法启动。")
                return run_swarm(
                    parent=parent, description=description, task=task, roles=roles,
                    max_rounds=max_rounds, make_executor=make_executor,
                )

            options["runner"] = swarm_runner
        # 到这里才真正调用工具：前面的校验、权限、确认、审计和复核均已通过。
        result = handler(arguments, root, **options)
        # 只记忆「本次明确批准且写入成功」的目录，失败不会扩大后续权限。
        if approved_directory is not None and result.get("status", "success") == "success":
            session_cache.remember(name, approved_directory, workspace=root)
        return {**result, "status": result.get("status", "success"), "executed": True, "tool": name}
    except AuditError:
        return error("audit_failed", "无法写入权限审计日志，未执行工具；请检查工作目录的日志权限。")
    except ToolError as failure:
        return error(failure.code, str(failure))
    except Exception:
        return error("execution_error", "工具执行器发生错误，未取得有效结果。")


def _get_permissions(permissions):
    if permissions is None:
        settings = get_settings()
        security = settings.get("security", {"mode": "ask", "auto_directories": []})
        return PermissionPolicy(
            mode=security["mode"],
            auto_directories=security["auto_directories"],
            **settings["permissions"],
        )
    if not isinstance(permissions, PermissionPolicy):
        raise ValueError("permissions 必须是 PermissionPolicy。")
    return permissions


def _validate(value, schema, field="工具参数"):
    """校验当前工具使用的 JSON Schema 子集，未支持的规则拒绝执行。"""
    supported = {"type", "description", "properties", "required", "additionalProperties",
                 "minLength", "minimum", "maximum", "default",
                 "items", "minItems", "maxItems"}
    if set(schema) - supported:
        raise ValueError("工具定义使用了尚未支持的参数校验规则。")
    kind = schema.get("type")
    if kind == "object":
        if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
            raise ValueError("当前仅支持布尔类型的 additionalProperties。")
        if not isinstance(value, dict):
            raise ToolError("invalid_arguments", f"{field}必须是 JSON 对象。")
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                raise ToolError("invalid_arguments", f"缺少必填参数 {name}。")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ToolError("invalid_arguments", "工具参数包含未声明的字段。")
        for name, item in value.items():
            if name in properties:
                _validate(item, properties[name], name)
    elif kind == "string":
        if not isinstance(value, str):
            raise ToolError("invalid_arguments", f"{field}必须是字符串。")
        if len(value) < schema.get("minLength", 0):
            raise ToolError("invalid_arguments", f'{field}至少需要 {schema["minLength"]} 个字符。')
    elif kind == "integer":
        if type(value) is not int:
            raise ToolError("invalid_arguments", f"{field}必须是整数。")
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolError("invalid_arguments", f'{field}不得小于 {schema["minimum"]}。')
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolError("invalid_arguments", f'{field}不得大于 {schema["maximum"]}。')
    elif kind == "array":
        if not isinstance(value, list):
            raise ToolError("invalid_arguments", f"{field}必须是数组。")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ToolError("invalid_arguments", f'{field}至少需要 {schema["minItems"]} 项。')
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolError("invalid_arguments", f'{field}最多允许 {schema["maxItems"]} 项。')
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(item, schema["items"], f"{field}[{index}]")
    else:
        raise ValueError("工具定义使用了尚未支持的参数类型。")
