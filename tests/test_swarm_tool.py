from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from harness.tools import REGISTRY, get_tool_definitions
from harness.tools.executor import ToolError
from harness.tools.swarm import DEFAULT_ROLES, DEFINITION, execute


class SwarmToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def test_definition_and_defaults_are_registered(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("swarm", names)
        self.assertIs(REGISTRY.get("swarm")[0], DEFINITION)
        schema = DEFINITION.to_deepseek()["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"description", "task", "roles", "max_rounds"})
        self.assertEqual(schema["properties"]["roles"]["type"], "array")
        self.assertEqual([role["name"] for role in DEFAULT_ROLES],
                         ["Coder", "Reviewer", "Tester"])
        self.assertIn("write_file", DEFAULT_ROLES[-1]["tools"])

    def test_handler_validates_roles_handoffs_and_runtime_runner(self):
        arguments = {
            "description": "团队任务",
            "task": "实现缓存模块",
            "roles": [
                {"name": "Coder", "system": "编写", "tools": ["read_file"],
                 "handoff_to": ["Reviewer"]},
                {"name": "Reviewer", "system": "审查", "tools": ["read_file"],
                 "handoff_to": ["Coder", "Tester"]},
                {"name": "Tester", "system": "测试", "tools": ["bash"]},
            ],
            "max_rounds": 3,
        }
        result = execute(arguments, self.root, runner=lambda **kwargs: kwargs)
        self.assertEqual(result["max_rounds"], 3)
        self.assertEqual([role["name"] for role in result["roles"]],
                         ["Coder", "Reviewer", "Tester"])
        for invalid in (
            {**arguments, "roles": arguments["roles"] + [arguments["roles"][0]]},
            {**arguments, "roles": [dict(arguments["roles"][0], handoff_to=["Missing"])]},
            {**arguments, "roles": [dict(arguments["roles"][0], handoff_to=["Coder"])]},
            {**arguments, "max_rounds": 0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ToolError):
                execute(invalid, self.root, runner=lambda **kwargs: kwargs)
        with self.assertRaises(ToolError):
            execute(arguments, self.root)


if __name__ == "__main__":
    unittest.main()
