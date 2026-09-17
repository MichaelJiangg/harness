"""Tool Executor：校验调用、分发执行，并统一结果与错误。"""

from copy import deepcopy
from functools import partial
from pathlib import Path

from .registry import REGISTRY


class ToolError(Exception):
    """可安全回传给模型的工具错误。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def create_tool_executor(workspace=None, *, confirm=None, abort=None):
    """固定会话启动目录，后续调用不随工作目录改变。"""
    root = Path.cwd() if workspace is None else Path(workspace)
    return partial(execute_tool, workspace=root.resolve(), confirm=confirm, abort=abort)


def execute_tool(name, arguments, *, workspace=None, confirm=None, abort=None):
    def error(code, message):
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
        _validate(arguments, definition.input_schema)
        root = Path.cwd() if workspace is None else Path(workspace)
        if abort is not None and abort.is_set():
            return error("cancelled", "查询已停止，未执行工具。")
        if definition.requires_confirmation:
            if confirm is None:
                return error("confirmation_required", "此工具需要用户本地确认，当前未提供确认入口。")
            # 将预览和执行绑定到同一份参数；确认回调不能修改将执行的内容。
            arguments = deepcopy(arguments)
            try:
                approved = confirm(name, deepcopy(arguments), root)
            except Exception:
                return error("confirmation_failed", "未能取得用户本地确认，未执行工具。")
            if abort is not None and abort.is_set():
                return error("cancelled", "查询已停止，未执行工具。")
            if approved is not True:
                return error("confirmation_denied", "用户未批准本次操作，未执行工具；请等待用户的新要求，不要自行重试。")
        options = {"abort": abort} if definition.supports_cancellation else {}
        result = handler(arguments, root, **options)
        return {**result, "status": result.get("status", "success"), "executed": True, "tool": name}
    except ToolError as failure:
        return error(failure.code, str(failure))
    except Exception:
        return error("execution_error", "工具执行器发生错误，未取得有效结果。")


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
