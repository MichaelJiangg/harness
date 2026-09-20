from .base import register


@register("/history", "显示对话历史", available_while_busy=True)
def handle(context, arguments):
    if arguments:
        context.write("用法：/history")
        return
    context.write(context.format_history())
