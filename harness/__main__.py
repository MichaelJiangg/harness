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
        app = create_app(provider=args.provider, model=args.model)
        from .startup import check_startup, format_startup_report

        report_items, warnings = check_startup(
            client=app.client,
            check_mcp=args.check,
        )
        print(format_startup_report(report_items, warnings, check_mode=args.check))
        if args.check:
            return 0
        app()
    except ValueError as error:
        message = str(error)
        print(f"错误：{message}", file=sys.stderr)
        if "API_KEY" in message:
            print("提示：请在项目 .env 中设置 DEEPSEEK_API_KEY 或 GLM_API_KEY。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
