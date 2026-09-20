"""从 .harness/skills/*.json 加载可复用技能包。"""

from dataclasses import dataclass
import json
from pathlib import Path


SKILLS_DIR = ".harness/skills"
MAX_SKILLS = 100
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 200
MAX_PROMPT_CHARS = 12000
MAX_TOOLS = 20


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    prompt: str
    tools: tuple[str, ...]
    path: Path


class SkillManager:
    def __init__(self, workspace=None):
        self.workspace = (Path.cwd() if workspace is None else Path(workspace)).resolve()
        self.directory = self.workspace / SKILLS_DIR
        self.skills = ()
        self.load_errors = []
        self._load()

    def _load(self):
        if not self.directory.is_dir():
            return
        skills = []
        seen = set()
        try:
            files = sorted(
                list(self.directory.glob("*.json"))
                + list(self.directory.glob("*.skill"))
            )
        except OSError:
            self.load_errors.append("无法读取 skills 目录。")
            return
        for path in files[:MAX_SKILLS]:
            try:
                skill = self._parse(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                self.load_errors.append(f"{path.name}: {error}")
                continue
            if skill.name in seen:
                self.load_errors.append(f"{path.name}: 技能名称重复：{skill.name}")
                continue
            seen.add(skill.name)
            skills.append(skill)
        self.skills = tuple(skills)

    def _parse(self, path):
        content = path.read_text(encoding="utf-8")
        document = json.loads(content)
        if not isinstance(document, dict):
            raise ValueError("必须是 JSON 对象。")
        name = document.get("name")
        description = document.get("description")
        prompt = document.get("prompt")
        tools = document.get("tools")
        if not isinstance(name, str) or not name.strip() or len(name) > MAX_NAME_CHARS:
            raise ValueError("name 无效。")
        if not isinstance(description, str) or not description.strip() \
                or len(description) > MAX_DESCRIPTION_CHARS:
            raise ValueError("description 无效。")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError("prompt 无效。")
        if not isinstance(tools, list) or not tools or len(tools) > MAX_TOOLS:
            raise ValueError("tools 必须是非空列表。")
        if any(not isinstance(tool, str) or not tool.strip() or "\x00" in tool
               for tool in tools):
            raise ValueError("tools 包含无效名称。")
        return Skill(
            name=name.strip(),
            description=description.strip(),
            prompt=prompt.strip(),
            tools=tuple(tools),
            path=path.resolve(),
        )

    def list(self):
        return self.skills

    def get(self, name):
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def activate(self, name):
        return self.get(name)


def format_skill_list(skills):
    if not skills:
        return "暂无可用技能。"
    lines = ["Available skills:"]
    for skill in skills:
        lines.append(f"  {skill.name} — {skill.description}")
    return "\n".join(lines)
