"""Tool Registry：自动发现工具，并维护名称到定义与实现的映射。"""

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

from .definition import ToolDefinition


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, definition, handler):
        if (not isinstance(definition, ToolDefinition)
                or not isinstance(definition.name, str) or not definition.name.strip()):
            raise ValueError("工具必须提供有效的 ToolDefinition 和名称。")
        if not callable(handler):
            raise ValueError(f"工具 {definition.name} 必须提供可调用的 execute。")
        if definition.name in self._tools:
            raise ValueError(f"工具名称重复：{definition.name}。")
        self._tools[definition.name] = (definition, handler)

    def get(self, name):
        return self._tools.get(name)

    def definitions(self):
        return [definition.to_deepseek() for definition, _ in self._tools.values()]

    def discover(self):
        """仅发现本包直属工具模块；辅助模块使用下划线前缀。"""
        modules = iter_modules([str(Path(__file__).parent)])
        for module in sorted(modules, key=lambda item: item.name):
            if (module.ispkg or module.name.startswith("_")
                    or module.name in {"definition", "registry", "executor"}):
                continue
            implementation = import_module(f"{__package__}.{module.name}")
            try:
                self.register(getattr(implementation, "DEFINITION", None),
                              getattr(implementation, "execute", None))
            except ValueError as error:
                raise ValueError(f"工具模块 {module.name} 注册失败：{error}") from None


REGISTRY = ToolRegistry()


def get_tool_definitions():
    return REGISTRY.definitions()
