from .base import register


@register("/clear", "清空对话，重新开始")
def handle(context, arguments):
    if arguments:
        context.write("用法：/clear")
        return
    context.reset_conversation()
    context.write("对话已清空，重新开始。")
