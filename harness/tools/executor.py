"""Tool Executor：校验调用、检查权限、分发执行，并统一结果与错误。"""

from copy import deepcopy
from functools import partial
from pathlib import Path
from uuid import uuid4

from ..audit import AuditError, PermissionAuditLog
from ..config import get_settings
from ..permissions import PermissionDecision, PermissionPolicy, SessionPermissionCache, build_denial_message
from .registry import REGISTRY


class ToolError(Exception):
    """可安全回传给模型的工具错误。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def create_tool_executor(workspace=None, *, confirm=None, abort=None, permissions=None,
                         session_cache=None, audit=None):
    """固定会话启动目录及权限快照，后续调用不随环境改变。"""
    root = (Path.cwd() if workspace is None else Path(workspace)).resolve()
    policy = _get_permissions(permissions)
    if session_cache is not None and not isinstance(session_cache, SessionPermissionCache):
        raise ValueError("session_cache 必须是 SessionPermissionCache。")
    audit = audit if audit is not None else PermissionAuditLog(root)
    return partial(execute_tool, workspace=root, confirm=confirm, abort=abort,
                   permissions=policy, session_cache=session_cache, audit=audit)


def execute_tool(name, arguments, *, workspace=None, confirm=None, abort=None, permissions=None,
                 session_cache=None, audit=None):
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
    registered = REGISTRY.get(name)
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
                if (result.decision == "ask" and session_cache is not None
                        and session_cache.is_approved(name, arguments, workspace=root)):
                    return PermissionDecision("allow", result.risk, "session:write_directory",
                                              "本会话已授权该目录写入。")
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

        initial_target = file_target()
        decision = check_permissions()
        if decision.matched_rule == "session:write_directory":
            confirmation = "remembered"
        elif decision.decision == "ask":
            confirmation = "pending"
        record(decision)
        if decision.decision == "deny":
            return error("permission_denied", decision.reason)
        if decision.decision == "ask":
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
            if approved is not True:
                return error("confirmation_denied", "用户未批准本次操作，未执行工具；请等待用户的新要求，不要自行重试。")
            if (name == "write_file" and session_cache is not None and initial_target is not None
                    and initial_target.is_relative_to(root)):
                approved_directory = str(initial_target.parent.relative_to(root))
        current = check_permissions()
        record(current, event="before_execute")
        # 日志落盘也可能耗时，必须在落盘之后再次核对规则与实际目标。
        current = check_permissions()
        if current.decision == "deny":
            record(current, event="guard_denial")
            return error("permission_denied", current.reason)
        if current != decision or file_target() != initial_target:
            record(PermissionDecision("deny", current.risk, "guard:target_changed", "权限或目标发生变化。"),
                   event="guard_denial")
            return error("permission_changed", "路径对应的权限条件已变化，未执行工具；请重新发起调用。")
        if abort is not None and abort.is_set():
            return error("cancelled", "查询已停止，未执行工具。")
        options = {"abort": abort} if definition.supports_cancellation else {}
        result = handler(arguments, root, **options)
        if approved_directory is not None and result.get("status", "success") == "success":
            session_cache.remember("write_file", approved_directory, workspace=root)
        return {**result, "status": result.get("status", "success"), "executed": True, "tool": name}
    except AuditError:
        return error("audit_failed", "无法写入权限审计日志，未执行工具；请检查工作目录的日志权限。")
    except ToolError as failure:
        return error(failure.code, str(failure))
    except Exception:
        return error("execution_error", "工具执行器发生错误，未取得有效结果。")


def _get_permissions(permissions):
    if permissions is None:
        return PermissionPolicy(**get_settings()["permissions"])
    if not isinstance(permissions, PermissionPolicy):
        raise ValueError("permissions 必须是 PermissionPolicy。")
    return permissions


def _validate(value, schema, field="工具参数"):
    """校验当前工具使用的 JSON Schema 子集，未支持的规则拒绝执行。"""
    supported = {"type", "description", "properties", "required", "additionalProperties",
                 "minLength", "minimum", "maximum", "default"}
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
    else:
        raise ValueError("工具定义使用了尚未支持的参数类型。")
