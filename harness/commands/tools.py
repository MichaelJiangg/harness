from .base import register


@register("/tools", "查看当前可用的所有工具", available_while_busy=True)
def handle(context, arguments):
    if arguments:
        context.write("用法：/tools")
        return
    context.write(context.format_tools())
