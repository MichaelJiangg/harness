"""权限规则按优先级匹配，第一条命中生效，未命中使用默认策略。"""

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import re
import unicodedata

from .bash_risk import classify_bash_risk


DEFAULT_POLICY = {"low": "allow", "medium": "ask", "high": "ask"}


def get_risk_level(tool_name, params=None):
    if tool_name in {"read_file", "grep"}:
        return "low"
    if tool_name == "bash":
        command = params.get("command") if isinstance(params, dict) else None
        if not isinstance(command, str):
            return "high"
        return {"read_only": "low", "write": "medium", "destructive": "high"}[classify_bash_risk(command)]
    return "medium"


@dataclass(frozen=True)
class PermissionDecision:
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
            if (self.tool not in {"read_file", "write_file"}
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
        if self.action == "allow" and (self.tool == "bash"
                                      or self.tool == "write_file" and self.directory is None):
            raise ValueError("Bash 不支持强制 allow；写入 allow 规则必须限定 directory。")

    def matches(self, tool_name, params, *, workspace=None):
        if self.tool != tool_name:
            return False
        if self._pattern is not None:
            command = params.get("command")
            return isinstance(command, str) and self._pattern.search(command) is not None
        if self.directory is None:
            return True
        raw_path = params.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise ValueError("无法检查目录规则，工具路径无效。")
        root = Path.cwd() if workspace is None else Path(workspace)
        root = root.resolve()
        boundary = root / self.directory
        candidate = root / raw_path
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
        rules.append(value if value.name is not None else replace(value, name=f"rules[{index}]"))
    if len({rule.name for rule in rules}) != len(rules):
        raise ValueError("权限规则名称不能重复。")
    return tuple(sorted(rules, key=lambda rule: (rule.action != "deny", -rule.priority)))


def check_permission(tool_name, params, rules, *, workspace=None, default=None, details=False):
    """rules 由 parse_rules 按优先级排列；第一条匹配规则决定结果。"""
    risk = get_risk_level(tool_name, params)
    result = PermissionDecision(default if default is not None else DEFAULT_POLICY[risk],
                                risk, "default", "使用风险等级默认策略。")
    for rule in rules:
        if rule.matches(tool_name, params, workspace=workspace):
            result = PermissionDecision(rule.action, risk, rule.name or "rule",
                                        f"命中权限规则「{rule.name or 'rule'}」，决策为 {rule.action}。")
            break
    return result if details else result.decision


@dataclass(frozen=True)
class PermissionPolicy:
    allow: frozenset[str] = frozenset()
    ask: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()
    rules: tuple[PermissionRule, ...] = ()

    def __post_init__(self):
        for name in ("allow", "ask", "deny"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple, set, frozenset)):
                raise ValueError(f"权限 {name} 必须是工具名称列表。")
            if any(not isinstance(value, str) or not value.strip() or value != value.strip()
                   for value in values):
                raise ValueError(f"权限 {name} 必须包含有效的非空工具名称。")
            object.__setattr__(self, name, frozenset(values))
        object.__setattr__(self, "rules", parse_rules(self.rules))

    def check(self, name, params=None, *, workspace=None):
        return self.evaluate(name, params, workspace=workspace).decision

    def evaluate(self, name, params=None, *, workspace=None):
        params = params if params is not None else {}
        risk = get_risk_level(name, params)
        if name in self.deny:
            return PermissionDecision("deny", risk, "tools:deny", "当前工具位于配置的 deny 列表中。")
        if name in self.ask or name == "write_file":
            default = "ask"
        elif name in self.allow and name != "bash":
            default = "allow"
        else:
            default = None
        result = check_permission(name, params, self.rules, workspace=workspace, default=default, details=True)
        if result.matched_rule == "default" and default is not None:
            result = replace(result, matched_rule=f"tools:{default}", reason="使用工具级默认权限配置。")
        return result


class SessionPermissionCache:
    """仅由本地确认入口授予当前执行器的写入目录权限，不保存到磁盘。"""

    def __init__(self):
        self._approved_patterns = set()

    def remember(self, tool_name, directory, *, workspace):
        if tool_name != "write_file":
            return
        root = Path(workspace).resolve()
        rule = PermissionRule("write_file", "allow", directory=directory)
        self._approved_patterns.add((str(root), rule.directory))

    def is_approved(self, tool_name, params, *, workspace):
        if tool_name != "write_file":
            return False
        root = Path(workspace).resolve()
        return any(saved_root == str(root) and PermissionRule(
            "write_file", "allow", directory=directory).matches(tool_name, params, workspace=root)
            for saved_root, directory in self._approved_patterns)
