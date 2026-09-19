"""权限引擎：加载时校验并排序规则，调用时匹配规则并返回决策。

PermissionPolicy 处理工具级禁止和默认配置；check_permission 负责首条匹配。
这里只回答「能否执行」，实际询问用户、记录审计和执行工具由 executor 负责。
"""

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import re
import unicodedata

from .bash_risk import bash_auto_allowed, classify_bash_risk


# 风险等级与权限决策是两件事：高风险默认仍须确认，由 CLI 加醒目警告；
# 永久禁止要靠 deny 配置。ask 对应示例骨架中的 CONFIRM。
DEFAULT_POLICY = {"low": "allow", "medium": "ask", "high": "ask"}


def get_risk_level(tool_name, params=None):
    """评估操作本身的风险，不在这里判断规则或询问用户。"""
    if tool_name in {"read_file", "grep", "notes_read"}:
        return "low"
    if tool_name == "background_check":
        return "low"
    if tool_name == "background_submit":
        command = params.get("command") if isinstance(params, dict) else None
        return get_risk_level("bash", {"command": command}) if isinstance(command, str) else "medium"
    if tool_name == "bash":
        command = params.get("command") if isinstance(params, dict) else None
        if not isinstance(command, str):
            return "high"
        return {"read_only": "low", "write": "medium", "destructive": "high"}[classify_bash_risk(command)]
    return "medium"


@dataclass(frozen=True)
class PermissionDecision:
    """决策及其依据：既用于执行分支，也用于拒绝说明和审计定位。"""

    decision: str
    risk: str
    matched_rule: str
    reason: str


def build_denial_message(tool_name, params, reason):
    """仅说明操作与权限原因，不把正文或命令参数复制到拒绝消息。"""
    return (
        f"操作 {tool_name} 被权限系统拒绝。\n原因：{reason}\n"
        "建议：向用户说明限制并提出更安全的方案，或请用户手动执行；"
        "不要更换工具绕过限制，也不要自行重试。"
    )


@dataclass(frozen=True)
class PermissionRule:
    """一条规则 = 目标工具 + 可选匹配条件 + 命中后的 action。"""

    tool: str
    action: str
    priority: int = 0
    directory: str | None = None
    command_pattern: str | None = None
    name: str | None = None
    _pattern: re.Pattern | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.tool, str) or not self.tool
                or any(character.isspace() for character in self.tool)):
            raise ValueError("规则 tool 必须是精确的非空工具名。")
        if self.action not in ("allow", "ask", "deny"):
            raise ValueError("规则 action 必须是 allow、ask 或 deny。")
        if type(self.priority) is not int:
            raise ValueError("规则 priority 必须是整数。")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()
                                     or len(self.name) > 128 or not self.name.isprintable()):
            raise ValueError("规则 name 必须是最多 128 字符的非空可显示名称。")
        if self.directory is not None:
            if (self.tool not in {"read_file", "write_file", "run_verify"}
                    or not isinstance(self.directory, str) or not self.directory.strip()
                    or "\x00" in self.directory or Path(self.directory).is_absolute()
                    or ".." in Path(self.directory).parts):
                raise ValueError("规则 directory 仅用于 read_file／write_file，须为不含 .. 的非空相对目录。")
        if self.command_pattern is not None:
            if (self.tool != "bash" or self.directory is not None
                    or not isinstance(self.command_pattern, str) or not self.command_pattern.strip()):
                raise ValueError("规则 command_pattern 仅用于 Bash，须为非空正则表达式。")
            try:
                pattern = re.compile(self.command_pattern)
            except (re.error, OverflowError, RecursionError):
                raise ValueError("规则 command_pattern 不是有效的正则表达式。") from None
            object.__setattr__(self, "_pattern", pattern)
        if self.action == "allow" and (
            self.tool == "bash"
            or self.tool in {"write_file", "run_verify"} and self.directory is None
        ):
            raise ValueError("Bash 不支持强制 allow；写入和验证 allow 规则必须限定 directory。")

    def matches(self, tool_name, params, *, workspace=None):
        """只判断本条规则是否适用；是否放行由调用方读取 action 决定。"""
        if self.tool != tool_name:
            return False
        if self._pattern is not None:
            # Bash 条件搜索整条命令原文，不能当成完整的 Shell 行为分析。
            command = params.get("command")
            return isinstance(command, str) and self._pattern.search(command) is not None
        if self.directory is None:
            # 工具名已经匹配，且没有目录或命令条件，规则即适用。
            return True
        raw_path = params.get("path") if self.tool in {"read_file", "write_file"} else params.get("target")
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise ValueError("无法检查目录规则，工具路径无效。")
        root = Path.cwd() if workspace is None else Path(workspace)
        root = root.resolve()
        boundary = root / self.directory
        candidate = root / raw_path
        # lexical 消除 .. 等路径写法，resolved 进一步跟随软链接找到实际目标。
        lexical = Path(os.path.abspath(candidate))
        resolved = candidate.resolve()
        # 放行须同时满足两种路径；限制也覆盖指向受限目录的其他别名。
        if self.action == "allow":
            return lexical.is_relative_to(boundary) and resolved.is_relative_to(boundary)
        # 限制也覆盖配置目录本身的软链接目标及大小写／Unicode 等价路径。
        boundaries = {
            Path(unicodedata.normalize("NFC", str(path)).casefold())
            for path in (boundary, boundary.resolve())
        }
        return any(Path(unicodedata.normalize("NFC", str(path)).casefold()).is_relative_to(limit)
                   for path in (lexical, resolved) for limit in boundaries)


def parse_rules(values):
    """验证并冻结配置，禁止规则优先，同类按数值降序、同值按声明顺序。"""
    if not isinstance(values, (list, tuple)):
        raise ValueError("权限 rules 必须是规则列表。")
    rules = []
    for index, value in enumerate(values):
        if isinstance(value, dict):
            try:
                value = PermissionRule(**value)
            except TypeError:
                raise ValueError("权限规则缺少规定字段或包含未知字段。") from None
        if not isinstance(value, PermissionRule):
            raise ValueError("权限 rules 必须包含有效的规则对象。")
        # 在排序前取声明位置，日志中的 rules[index] 才能对应原配置。
        rules.append(value if value.name is not None else replace(value, name=f"rules[{index}]"))
    if len({rule.name for rule in rules}) != len(rules):
        raise ValueError("权限规则名称不能重复。")
    # 第一排序项：deny 为 False，排在其他规则的 True 前面。
    # 第二排序项：取负数让更大的 priority 排在前面；同值保留原声明顺序。
    # 因此 allow 的数字再大，也不能压过一条匹配的 deny。
    return tuple(sorted(rules, key=lambda rule: (rule.action != "deny", -rule.priority)))


def check_permission(tool_name, params, rules, *, workspace=None, default=None, details=False):
    """按顺序匹配规则，首条命中决定结果；没有命中才使用默认策略。

    调用方须传入 parse_rules 排好序的 rules，本函数不重新排序。
    default 是工具级兜底决策；未提供时才按 DEFAULT_POLICY[risk] 兜底。
    默认返回 allow／ask／deny；details=True 额外返回风险、规则及原因。
    """
    # 1. 先评估风险，供默认策略、界面提示和审计共同使用。
    risk = get_risk_level(tool_name, params)
    # 2. 先准备「没有规则命中」的结果；后面的匹配规则可以覆盖它。
    result = PermissionDecision(default if default is not None else DEFAULT_POLICY[risk],
                                risk, "default", "使用风险等级默认策略。")
    # 3. 已按优先级排列，遇到第一条匹配就停止，不合并后续规则。
    for rule in rules:
        if rule.matches(tool_name, params, workspace=workspace):
            result = PermissionDecision(rule.action, risk, rule.name or "rule",
                                        f"命中权限规则「{rule.name or 'rule'}」，决策为 {rule.action}。")
            break
    # 若循环没有命中，result 仍是上面准备的兜底结果。
    return result if details else result.decision


@dataclass(frozen=True)
class PermissionPolicy:
    """完整策略入口：工具级禁止 → 有序规则 → 工具级或风险默认策略。"""

    allow: frozenset[str] = frozenset()
    ask: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()
    rules: tuple[PermissionRule, ...] = ()
    mode: str = "ask"
    auto_directories: frozenset[str] = frozenset()

    def __post_init__(self):
        for name in ("allow", "ask", "deny"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple, set, frozenset)):
                raise ValueError(f"权限 {name} 必须是工具名称列表。")
            if any(not isinstance(value, str) or not value.strip() or value != value.strip()
                   for value in values):
                raise ValueError(f"权限 {name} 必须包含有效的非空工具名称。")
            object.__setattr__(self, name, frozenset(values))
        if self.mode not in {"ask", "auto"}:
            raise ValueError("权限模式必须是 ask 或 auto。")
        directories = self.auto_directories
        if not isinstance(directories, (list, tuple, set, frozenset)):
            raise ValueError("auto_directories 必须是目录列表。")
        if any(not isinstance(value, str) or not value.strip() or "\x00" in value
               or Path(value).is_absolute() or ".." in Path(value).parts
               for value in directories):
            raise ValueError("auto_directories 必须是工作区内的非空相对目录。")
        object.__setattr__(self, "auto_directories", frozenset(directories))
        object.__setattr__(self, "rules", parse_rules(self.rules))

    def check(self, name, params=None, *, workspace=None):
        return self.evaluate(name, params, workspace=workspace).decision

    def evaluate(self, name, params=None, *, workspace=None):
        params = params if params is not None else {}
        risk = get_risk_level(name, params)
        # 工具级 deny 直接返回，目录规则和会话授权都不能覆盖它。
        if name in self.deny:
            return PermissionDecision("deny", risk, "tools:deny", "当前工具位于配置的 deny 列表中。")
        if self.mode == "auto" and self._auto_allowed(name, params, workspace):
            return PermissionDecision("allow", risk, "session:auto",
                                      "auto 模式已授权当前工作区内操作。")
        # 以下只是准备兜底，并不提前返回；细粒度规则仍有机会先决定结果。
        # 写文件不能仅凭工具名 allow 放行，Bash 的 allow 也不能跳过风险判断。
        if name in self.ask or name == "write_file":
            default = "ask"
        elif name in self.allow and name != "bash":
            default = "allow"
        else:
            default = None
        # 核心循环：首条规则匹配生效，否则采用刚才准备的兜底。
        result = check_permission(name, params, self.rules, workspace=workspace, default=default, details=True)
        if result.matched_rule == "default" and default is not None:
            result = replace(result, matched_rule=f"tools:{default}", reason="使用工具级默认权限配置。")
        return result

    def _auto_allowed(self, name, params, workspace):
        root = Path.cwd() if workspace is None else Path(workspace)
        root = root.resolve()
        if name == "bash":
            command = params.get("command") if isinstance(params, dict) else None
            return isinstance(command, str) and bash_auto_allowed(
                command, root, self.auto_directories,
            )
        if name not in {"read_file", "grep", "write_file", "run_verify"}:
            return False
        key = "path" if name in {"read_file", "grep", "write_file"} else "target"
        raw = params.get(key) if isinstance(params, dict) else None
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
            return False
        if name == "grep" and raw == ".":
            raw = "."
        try:
            candidate = (root / raw).resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        roots = [root] if not self.auto_directories else [
            (root / directory).resolve() for directory in self.auto_directories
        ]
        return any(candidate.is_relative_to(allowed) for allowed in roots)


class SessionPermissionCache:
    """仅由本地确认入口授予当前执行器的写入目录权限，不保存到磁盘。"""

    def __init__(self):
        self._approved_patterns = set()

    def remember(self, tool_name, directory, *, workspace):
        # 执行器只在明确确认且成功执行后调用；Bash 不获得会话授权。
        if tool_name not in {"write_file", "run_verify"}:
            return
        root = Path(workspace).resolve()
        rule = PermissionRule(tool_name, "allow", directory=directory)
        self._approved_patterns.add((str(root), rule.directory))

    def is_approved(self, tool_name, params, *, workspace):
        if tool_name not in {"write_file", "run_verify"}:
            return False
        root = Path(workspace).resolve()
        return any(saved_root == str(root) and PermissionRule(
            tool_name, "allow", directory=directory).matches(tool_name, params, workspace=root)
            for saved_root, directory in self._approved_patterns)
