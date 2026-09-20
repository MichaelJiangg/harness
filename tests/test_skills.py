import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from harness.cli import run_cli
from harness.skills import SkillManager, format_skill_list


def reply(content="技能完成。"):
    return {
        "model": "deepseek-flash",
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10,
        },
    }


class SkillManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.skills = self.workspace / ".harness" / "skills"
        self.skills.mkdir(parents=True)

    def write_skill(self, name, tools=("read_file", "grep")):
        path = self.skills / f"{name}.json"
        path.write_text(json.dumps({
            "name": name,
            "description": f"{name} 描述",
            "prompt": f"执行 {name} 技能。",
            "tools": list(tools),
        }), encoding="utf-8")
        return path

    def test_loads_lists_and_activates_skills(self):
        self.write_skill("code-review")
        self.write_skill("project-overview", tools=("read_file", "grep", "notes_append"))
        manager = SkillManager(self.workspace)
        self.assertEqual([skill.name for skill in manager.list()],
                         ["code-review", "project-overview"])
        skill = manager.activate("code-review")
        self.assertEqual(skill.tools, ("read_file", "grep"))
        self.assertIn("code-review", format_skill_list(manager.list()))

    def test_duplicate_and_invalid_files_are_reported(self):
        self.write_skill("shared")
        (self.skills / "duplicate.json").write_text(json.dumps({
            "name": "shared", "description": "重复", "prompt": "重复", "tools": ["read_file"],
        }), encoding="utf-8")
        (self.skills / "broken.json").write_text("{broken", encoding="utf-8")
        manager = SkillManager(self.workspace)
        self.assertEqual([skill.name for skill in manager.list()], ["shared"])
        self.assertTrue(manager.load_errors)

    def test_skill_extension_is_supported(self):
        path = self.skills / "qa.skill"
        path.write_text(json.dumps({
            "name": "qa",
            "description": "测试技能",
            "prompt": "设计并运行测试。",
            "tools": ["run_verify"],
        }), encoding="utf-8")
        manager = SkillManager(self.workspace)
        self.assertEqual(manager.activate("qa").name, "qa")


class SkillCLITests(unittest.TestCase):
    def test_skill_list_activation_prompt_and_tool_restriction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            skills = root / ".harness" / "skills"
            skills.mkdir(parents=True)
            (skills / "code-review.json").write_text(json.dumps({
                "name": "code-review",
                "description": "代码审查：分析代码质量和潜在问题",
                "prompt": "以 Reviewer 身份审查代码。",
                "tools": ["read_file", "grep", "notes_append"],
            }), encoding="utf-8")
            client = Mock(complete=Mock(return_value=reply()))
            output = StringIO()
            with patch("harness.cli.Path.cwd", return_value=root):
                run_cli(
                    client,
                    input_stream=StringIO("/skill\n/skill code-review\n请审查代码\n"),
                    output=output,
                    memory_enabled=False,
                    notes_enabled=False,
                    hooks_enabled=False,
                )
            text = output.getvalue()
            self.assertIn("code-review", text)
            self.assertIn("[skill] Activated: code-review", text)
            request = client.complete.call_args_list[0].kwargs
            system = request["messages"][0]["content"]
            tools = [item["function"]["name"] for item in request["tools"]]
            self.assertIn("以 Reviewer 身份审查代码", system)
            self.assertEqual(set(tools), {"read_file", "grep", "notes_append"})


if __name__ == "__main__":
    unittest.main()
