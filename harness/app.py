"""把配置、模型客户端和完整查询循环组装成可启动应用。"""

from .cli import run_cli
from .client import ChatCompletionClient
from .config import (
    get_settings,
    load_api_key,
    load_glm_api_key,
    select_model_provider,
)


def create_app(*, client=None, provider=None, model=None):
    """组装系统并返回唯一启动函数。

    默认从环境变量或项目 .env 选择 provider 并创建客户端；返回的 start()
    负责启动完整 CLI，工具权限、内置与 MCP 工具、记忆、笔记、Hooks、
    Agent 编排和终端渲染均由 run_cli 在对应时机连接。
    """
    settings = get_settings()
    if client is None:
        requested_provider = provider or settings.get("provider", "auto")
        if requested_provider == "auto":
            provider, api_key = select_model_provider()
        else:
            provider = requested_provider
            api_key = (
                load_api_key()
                if provider == "deepseek"
                else load_glm_api_key()
            )
            if not isinstance(api_key, str) or not api_key.strip():
                key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "GLM_API_KEY"
                raise ValueError(f"未配置 {key_name}，无法使用 provider={provider}。")
        selected_model = settings["model"]["name"]
        if provider == "glm" and settings["model"]["name"] == "deepseek-flash":
            selected_model = settings["glm"]["name"]
        client = ChatCompletionClient(
            api_key,
            provider=provider,
            model=model or selected_model,
        )

    def start(*, ledger=None, input_stream=None, output=None, error_output=None,
              character_delay=None, memory_store=None, notes_store=None,
              vector_store=None, hooks_manager=None):
        return run_cli(
            client,
            ledger=ledger,
            input_stream=input_stream,
            output=output,
            error_output=error_output,
            character_delay=character_delay,
            memory_enabled=True,
            memory_store=memory_store,
            notes_enabled=True,
            notes_store=notes_store,
            vector_store=vector_store,
            hooks_enabled=True,
            hooks_manager=hooks_manager,
            mcp_enabled=True,
        )

    start.client = client
    start.settings = settings
    start.provider = getattr(client, "provider", None)
    return start


__all__ = ["create_app"]
