# harness

一个类似 Claude Code 核心查询循环的最小命令行实现，模型使用 DeepSeek。Python 3.10+，仅使用标准库，零第三方依赖。

当前版本：**V0.2 query engine**（Git 标签 `v0.2`）。功能与已知限制见 [CHANGELOG.md](CHANGELOG.md) 。

## 启动

首次使用，在项目根目录创建 `.env` 并填写密钥（该文件不会随仓库发布）：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
```

之后每次在项目目录直接运行：

```sh
python3 -m harness
```

启动时自动读取项目根目录 `.env`，无需每次 `export` 或安装依赖。如果当前进程已设置 `DEEPSEEK_API_KEY`，环境变量优先于文件（包括已设置的空值）。支持单行值、单／双引号、注释和可选 `export` 前缀，不执行 shell 命令或变量插值；仅读取该密钥，不注入其他变量。`.gitignore` 已排除 `.env*`，不要将真实密钥提交到 Git。

```text
你 > 帮我读取 README.md
[请求用量记录]
[工具占位] read_file：工具尚未实现，未执行任何操作。
DeepSeek > 当前读取工具尚未实现，因此没有实际读取文件。
[请求用量记录]
你 > /cost
[本次会话累计 token、预估费用及每次请求明细]
```

上方是交互流程示意，回答文字会随着模型生成显示。交互终端按每个字符约 20 毫秒输出，管道输出不人为限速；每次请求结束后显示用量记录。

| 命令 | 功能 |
| --- | --- |
| `/cost` | 查询本次会话用量、USD 预估费用和逐请求明细，不调用模型；等待回答时也可使用 |
| `/compact` | 空闲时让模型总结旧对话，保留最近几轮；摘要请求计入 `/cost` |
| `/help` | 查看帮助 |
| `/exit` | 退出，停止后续模型和工具调用 |

`python3 -m harness --help` 无需密钥即可查看帮助。支持单条管道输入，输入流结束后会等待回答；交互中一次处理一个问题，繁忙时的新问题会被提示稍后重发。

## 查询循环

```text
用户输入 → DeepSeek
              ├─ 最终文字回答 → 显示回答，等待下个问题
              └─ 工具调用 → 执行占位 → 按 tool_call_id 回传结果 → DeepSeek
```

核心入口是 `harness/engine.py` 中的 `query_loop(state)`：

```python
from harness.client import DeepSeekClient
from harness.config import load_api_key
from harness.engine import QueryState, query_loop
from harness.usage import UsageLedger

state = QueryState(
    client=DeepSeekClient(load_api_key()),
    ledger=UsageLedger(),
)
state.messages.append({"role": "user", "content": "帮我读取 README.md"})
print(query_loop(state))
```

沿用用户给定的 Python 循环结构，客户端直接调用 DeepSeek Chat Completions。该协议返回 `tool_calls`，对应 Anthropic 示例中的 `tool_use`；回传使用 `role: "tool"` 和 `tool_call_id`。不依赖 Anthropic SDK，也不调用 Claude 模型。

- 使用 `deepseek-flash`，流式请求，显式设置 `thinking.type = "disabled"`。收到文字后逐字显示，接收结束后保留完整回答到对话历史。
- 在同一进程内保留成功查询的对话历史；后续问题可以引用之前的回答。
- 一次返回多个工具调用时，逐个执行并回传所有结果。即使回复附带文字，也会优先处理其中的工具调用。
- 工具名称和参数分片先拼接完整，再交给原有工具循环；token 用量从末尾流式数据读取，每次请求只记录一次。
- 每次模型请求都记录用量；一次用户提问可能产生多次模型请求。
- 目前 `read_file` 和 `run_command` 均返回 `not_implemented` 与 `executed: false`，没有真实文件读取或命令执行。
- 未知工具、参数错误与执行器错误转成工具结果，交给模型决定下一步。
- 一次用户提问或手动压缩最多实际请求模型 20 次，包含摘要和重试，每次请求超时为 120 秒；截断、空回答或循环达到上限均报错，不当作完成。
- 查询失败时保留已产生的用量，但不将未完成的对话加入后续历史。

`/exit` 和 Ctrl+C 会设置 `state.abort`，停止后续循环。Python 标准库的在途 HTTP 请求不会被这个标志立即中断，也不保证远端停止生成；直接退出时可能拿不到该请求的最终用量。命令行使用临时对话副本，只在查询成功后保存；直接调用 `query_loop(state)` 时由调用方管理失败状态。

## 上下文压缩与工具结果限制

| 规则 | 默认值与行为 |
| --- | --- |
| 自动触发 | 每次正常模型请求前检查 `messages` 和 `tools` 的序列化长度，超过 24000 字符时压缩 |
| 保留历史 | 原始系统提示、最近 4 个完整用户轮次和当前未完成轮；工具调用与对应结果整组保留 |
| 摘要 | 独立调用 DeepSeek，总结目标、约束、关键事实与决策、进展和待办，最多 2000 字符 |
| 压缩次数 | 每次用户提问或 `/compact` 最多尝试 2 次，计数贯穿整个工具循环 |
| 长度仍超限 | 提示「太长了，建议开个新会话」；保留部分本身超限时直接提示 |
| 工具结果 | 每个结果序列化后最多 6000 字符，超长时用包含 `truncated`、`original_chars`、`head`、`tail` 的 JSON 保留首尾 |

摘要正文不会作为聊天回答打印。摘要有效、未截断、确实缩短且候选上下文满足预算后才替换历史；摘要失败时保留原历史。`/compact` 在历史较短、没有可压缩的旧轮次时直接提示，无需模型请求。压缩过程中仍可使用 `/cost` 或 `/exit`。

字符预算是本地控制上下文体积的规则，不代表模型的精确 token 上限。实际 token 和费用仍以 API 返回的 `usage` 为依据。摘要可能遗漏旧对话细节，最近保留的完整轮次不变。

## 自动重试

网络断开、超时及 HTTP 429、500、502、503、504 最多重试 3 次，即首次请求加 3 次重试；等待分别为 1、2、4 秒。认证失败、余额不足、请求参数错误与无效响应格式直接报错。

若已显示部分回答后断线，终端会标明该次输出未完成，并另起一行重新生成；只有完整成功的回复才进入对话历史和工具执行。重试只重新发送模型请求，不重跑之前已执行的工具。

摘要请求同样使用重试机制。每次实际请求均单独计入 `/cost`，失败时拿不到用量会标为未知，不当作免费。等待重试和逐字显示期间，`/cost` 与退出命令仍能响应。

## Token 与费用

读取 API 的 `usage`，不在本地估算 token 数：

- 输入：`prompt_tokens`，已包含缓存命中与未命中。
- 输出：`completion_tokens`。
- 总数：`total_tokens`。
- 输入缓存：`prompt_cache_hit_tokens` 和 `prompt_cache_miss_tokens`。

费用按以下公式计算，费率单位为 USD／百万 tokens：

```text
费用 =（缓存命中 token × 命中单价
      + 缓存未命中 token × 未命中单价
      + 输出 token × 输出单价）÷ 1,000,000
```

`deepseek-flash` 费率于 2026-09-18 核验：

| 时段 | 输入缓存命中 | 输入缓存未命中 | 输出 |
| --- | ---: | ---: | ---: |
| 谷段 | $0.003 | $0.15 | $0.60 |
| 峰段 | $0.006 | $0.30 | $1.20 |

峰段为周一至周五 UTC 01:00–04:00、06:00–10:00（北京时间 09:00–12:00、14:00–18:00），其余时间为谷段。按响应 `created` 的 UTC 时间估算；该字段缺失时使用本机记录时刻。官方没有明确跨时段请求的定价时点，因此此金额为预估值，实际扣费以 DeepSeek 账单为准。人民币账单有独立价表，不应直接把这里的 USD 数字当作人民币。

价格保存在 `harness/usage.py` 的 `PRICING` 中，不自动联网刷新。官方费率变化时需更新该常量。统计只保存在当前进程内存中，退出后清空，不代表账户全部历史消费。

如果缓存明细缺失或不一致，按全部输入未命中缓存保守估算并标注；`usage` 缺失或非法、连接失败时，不会虚构零费用，会标记用量未知。汇总此时只代表已知小计，请以账单核对未知请求。

## 文件结构与扩展点

| 文件 | 职责 |
| --- | --- |
| `harness/__main__.py` | 读取配置并启动程序 |
| `harness/config.py` | 优先读取环境变量，回退读取项目根目录 `.env` |
| `harness/client.py` | DeepSeek HTTP 请求、超时与错误分类 |
| `harness/engine.py` | 查询循环、有限重试、摘要压缩与逐次记账 |
| `harness/context.py` | 字符预算、完整轮次切分、摘要资料和工具结果截断 |
| `harness/tools.py` | 工具声明和执行占位；后续真实工具在这里实现 |
| `harness/usage.py` | 逐请求 token 记录、模型费率与费用汇总 |
| `harness/cli.py` | 终端输入、回答和斜杠命令 |
| `tests/` | 模拟接口、工具、计费与 CLI 测试 |

## 验证

```sh
python3 -m unittest discover -s tests -v
```

测试覆盖纯文字回答、多工具与多轮调用、连续对话、压缩与回滚、工具结果限长、递增重试与取消、限速显示、请求上限、token 和峰谷费用、`/cost`、`/compact`、启动及 `.env` 加载行为。测试模拟 DeepSeek 响应和配置文件内容，不读取实际密钥、不产生真实 API 费用。真实 API 联调尚未验证。

官方依据：[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 、[Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/) 、[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) 、[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/) 、[Error Codes](https://api-docs.deepseek.com/quick_start/error_codes/) 。
