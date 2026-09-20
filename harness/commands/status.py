from .base import register


@register("/status", "查看当前运行状态", available_while_busy=True)
def handle(context, arguments):
    if arguments:
        context.write("用法：/status")
        return
    context.write(context.status_text())
