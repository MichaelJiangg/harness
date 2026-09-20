"""Tool Definition：模型可见的能力说明，不包含执行逻辑。"""

from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    supports_cancellation: bool = False
    validate_arguments: bool = True

    def to_deepseek(self):
        """将内部 input_schema 转换为 DeepSeek 的 function.parameters。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.input_schema),
            },
        }
