"""初始化工具注册表，向查询引擎导出描述和执行入口。"""

from .executor import create_tool_executor, execute_tool
from .registry import REGISTRY, ToolRegistry, get_tool_definitions


REGISTRY.discover()

__all__ = ["REGISTRY", "ToolRegistry", "get_tool_definitions", "create_tool_executor", "execute_tool"]
