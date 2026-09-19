"""多角色团队协作：角色定义、交接参数和执行器注入的编排入口。"""

from copy import deepcopy

from .definition import ToolDefinition
from .executor import ToolError


DEFAULT_ROLES = [
    {
        "name": "Coder",
        "system": (
            "你是 Coder，负责编写和修改代码。只使用可用的读取、搜索和写入工具；"
            "检查文件时优先使用 read_file 和 grep，不要用 Bash 做 ls/cat/sed/head 等只读检查。"
            "完成后输出交接 JSON，交给 Reviewer。"
        ),
        "tools": ["read_file", "write_file", "grep"],
        "handoff_to": ["Reviewer", "Tester"],
    },
    {
        "name": "Reviewer",
        "system": (
            "你是 Reviewer，只审查代码，不直接修改。发现问题时输出 next_role=Coder 和明确 feedback；"
            "优先使用 read_file 和 grep 检查文件，不做 Bash 只读命令。"
            "审查通过时输出 next_role=Tester。"
        ),
        "tools": ["read_file", "grep"],
        "handoff_to": ["Coder", "Tester"],
    },
    {
        "name": "Tester",
        "system": (
            "你是 Tester，负责为已有代码编写和运行测试。可以使用读取、搜索和 Bash 工具；"
            "检查文件和结果时优先使用 read_file/grep；运行工作区内已有验证脚本时使用 run_verify，"
            "不要用 node -e、cat > /tmp/... 等内联临时脚本；Bash 只用于其他真正需要执行的程序，"
            "并把多次核验合并成尽可能少的命令。完成后输出 next_role=null，并给出最终团队报告。"
        ),
        "tools": ["read_file", "grep", "bash", "write_file", "run_verify"],
        "handoff_to": [],
    },
]


DEFINITION = ToolDefinition(
    name="swarm",
    description=(
        "启动多角色团队协作，让多个独立 AI 按角色顺序完成同一任务，例如编码、审查、测试。"
        "description 是短标题，task 是团队总任务；可选 roles 定义每个角色的名称、专属提示、"
        "可用工具和允许的交接目标，省略时使用 Coder、Reviewer、Tester 默认团队。"
        "每个角色拥有独立上下文，只接收交接摘要、产物路径和打回反馈；"
        "角色不得使用 delegate、后台工具或 swarm，避免递归协作。"
        "max_rounds 默认 10，用于防止审查和修改之间无限循环。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {"type": "string", "minLength": 1, "description": "团队协作的简短标题。"},
            "task": {"type": "string", "minLength": 1, "description": "团队要完成的完整任务。"},
            "roles": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "description": "角色名称。"},
                        "system": {"type": "string", "minLength": 1, "description": "角色专属系统提示。"},
                        "tools": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {"type": "string", "minLength": 1},
                            "description": "角色可以使用的已注册工具名称。",
                        },
                        "handoff_to": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "minLength": 1},
                            "description": "该角色允许交给的其他角色名称。",
                        },
                    },
                    "required": ["name", "system"],
                    "additionalProperties": False,
                },
                "description": "团队角色定义；省略时使用 Coder、Reviewer、Tester。",
            },
            "max_rounds": {
                "type": "integer", "minimum": 1, "maximum": 20, "default": 10,
                "description": "最多执行的角色轮次，默认 10，防止无限打回。",
            },
        },
        "required": ["description", "task"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace, *, runner=None, abort=None):
    description = arguments["description"]
    task = arguments["task"]
    roles = arguments.get("roles")
    max_rounds = arguments.get("max_rounds", 10)
    if not isinstance(description, str) or not description.strip() or "\x00" in description:
        raise ToolError("invalid_arguments", "description 必须是非空文本，且不能含空字符。")
    if not isinstance(task, str) or not task.strip() or "\x00" in task:
        raise ToolError("invalid_arguments", "task 必须是非空文本，且不能含空字符。")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 20:
        raise ToolError("invalid_arguments", "max_rounds 必须是 1～20 之间的整数。")
    roles = deepcopy(DEFAULT_ROLES if roles is None else roles)
    names = []
    for role in roles:
        name = role.get("name")
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            raise ToolError("invalid_arguments", "每个角色必须有非空且不含空字符的 name。")
        if name in names:
            raise ToolError("invalid_arguments", f"角色名称重复：{name}。")
        if (not isinstance(role.get("system"), str) or not role["system"].strip()
                or "\x00" in role["system"]):
            raise ToolError("invalid_arguments", "每个角色必须有非空且不含空字符的 system。")
        if any(not isinstance(item, str) or not item.strip() or "\x00" in item
               for item in role.get("tools", [])):
            raise ToolError("invalid_arguments", "角色工具名称必须是非空且不含空字符的字符串。")
        if any(not isinstance(item, str) or not item.strip() or "\x00" in item
               for item in role.get("handoff_to", [])):
            raise ToolError("invalid_arguments", "交接目标必须是非空且不含空字符的字符串。")
        names.append(name)
        role["tools"] = role.get("tools", [])
        role["handoff_to"] = role.get("handoff_to", [])
        for target in role["handoff_to"]:
            if target == name:
                raise ToolError("invalid_arguments", "角色不能交接给自己。")
    role_names = set(names)
    for role in roles:
        unknown = set(role["handoff_to"]) - role_names
        if unknown:
            raise ToolError("invalid_arguments", f"交接目标不存在：{sorted(unknown)}。")
        if "bash" in role["tools"] and "run_verify" not in role["tools"]:
            role["tools"].append("run_verify")
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "查询已停止，未启动团队协作。")
    if runner is None:
        raise ToolError("swarm_unavailable", "当前没有可用的查询上下文，无法启动团队协作。")
    return runner(description=description, task=task, roles=roles, max_rounds=max_rounds)
