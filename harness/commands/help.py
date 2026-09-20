from .base import register


@register("/help", "显示此帮助信息", available_while_busy=True)
def handle(context, arguments):
    if arguments:
        context.write("用法：/help")
        return
    context.write(context.help_text)
