from .base import register


@register("/model", "查看或切换当前使用的 AI 模型")
def handle(context, arguments):
    if not arguments:
        context.write(f"Current model: {context.active_model} ({context.active_label})")
        context.write("Available models:")
        for name, details in context.available_models.items():
            context.write(f"  {name} — {details['model']} ({details['label']})")
        return
    if len(arguments) != 1 or arguments[0] not in context.available_models:
        context.write("用法：/model [deepseek|glm]")
        return
    if context.switch_model(arguments[0]):
        details = context.available_models[arguments[0]]
        context.write(
            f"已切换到 {details['model']}（{details['label']}）。"
        )
