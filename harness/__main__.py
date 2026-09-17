import sys

from .cli import HELP, run_cli
from .client import DeepSeekClient
from .config import load_api_key


def main():
    if "--help" in sys.argv[1:]:
        print(f"在项目 .env 中填写 DEEPSEEK_API_KEY 后运行 python3 -m harness。\n\n{HELP}")
        return 0
    try:
        run_cli(DeepSeekClient(load_api_key()))
    except ValueError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
