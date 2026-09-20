import sys

from .config import get_settings


def main():
    try:
        # 先验证配置，再导入依赖配置的模块，确保错误只显示简短说明。
        get_settings()
        from .app import create_app
        from .cli import HELP

        if "--help" in sys.argv[1:]:
            print(f"在项目 .env 中填写 DEEPSEEK_API_KEY 或 GLM_API_KEY 后运行 python3 -m harness。\n\n{HELP}")
            return 0
        create_app()()
    except ValueError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
