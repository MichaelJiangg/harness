from .base import register


@register(
    "/compact",
    "手动压缩对话上下文",
    busy_message="上一轮查询进行中，暂时无法压缩；期间可以输入 /cost。",
)
def handle(context, arguments):
    if arguments:
        context.write("用法：/compact")
        return
    context.compact()
