import sys

from .config import (
    cli_overrides_from_args,
    parse_cli_args,
    resolve_settings,
    set_process_settings,
)


def main():
    try:
        args = parse_cli_args(sys.argv[1:])
        set_process_settings(resolve_settings(
            config_file=args.config,
            cli_overrides=cli_overrides_from_args(args),
        ))
        from .app import create_app
        from .cli import HELP

        if args.help:
            print(f"在项目 .env 中填写 DEEPSEEK_API_KEY 或 GLM_API_KEY 后运行 python3 -m harness。\n\n{HELP}")
            return 0
        if args.show_config:
            from .config import get_settings
            print("已加载配置：")
            for key, value in get_settings().items():
                print(f"  {key}={value}")
            return 0
        create_app(provider=args.provider, model=args.model)()
    except ValueError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
