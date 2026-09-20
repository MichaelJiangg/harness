from .base import register


@register("/cost", "显示当前会话的 token 用量和费用", available_while_busy=True)
def handle(context, arguments):
    if arguments:
        context.write("用法：/cost")
        return
    context.write(context.format_cost())
    if context.busy:
        context.write("查询进行中：尚未返回的请求用量将在返回后记录。")
