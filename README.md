# harness

一个类似 Claude Code 核心查询循环的最小命令行实现，模型使用 DeepSeek。Python 3.10+，仅使用标准库，零第三方依赖。

当前版本：**V0.3 — Add Tool System**（Git 标签 `v0.3`）。新增三层工具系统、文件读取与关键词搜索，以及确认后的写文件和终端命令执行。版本记录见 [CHANGELOG.md](CHANGELOG.md) 。

## 启动

从 GitHub 下载项目并进入目录：

```sh
git clone https://github.com/MichaelJiangg/harness.git
cd harness
```

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
[工具] read_file：已读取文件（…… 字符）。
DeepSeek > README 介绍了这个查询引擎的使用方式……
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
              └─ 工具调用 → 按名称执行工具 → 按 tool_call_id 回传结果 → DeepSeek
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
- 当前开放只读的 `read_file`、`grep` 和需逐次确认的 `write_file`、`bash`，工具描述随每次正常模型请求发送。
- 未知工具、参数错误与执行器错误转成工具结果，交给模型决定下一步。
- 一次用户提问或手动压缩最多实际请求模型 20 次，包含摘要和重试，每次请求超时为 120 秒；截断、空回答或循环达到上限均报错，不当作完成。
- 查询失败时保留已产生的用量，但不将未完成的对话加入后续历史。

`/exit` 和 Ctrl+C 会设置 `state.abort`，停止后续循环。Python 标准库的在途 HTTP 请求不会被这个标志立即中断，也不保证远端停止生成；直接退出时可能拿不到该请求的最终用量。命令行使用临时对话副本，只在查询成功后保存；直接调用 `query_loop(state)` 时由调用方管理失败状态。

## 工具机制

工具统一放在 `harness/tools/`，分成定义、注册、执行三层：

```text
harness/tools/
├── __init__.py      # 初始化注册表并导出公共接口
├── definition.py    # 工具定义和 DeepSeek 格式转换
├── registry.py      # 自动发现、注册、按名称查找
├── executor.py      # 参数校验、本地确认、执行分发、统一结果与错误
├── read_file.py     # 读文件的定义与具体实现
├── write_file.py    # 写文件的定义与具体实现
├── bash.py          # 命令执行、超时清理和两路输出收集
└── grep.py          # 关键词搜索、文件名筛选和结果限长
```

每个工具模块提供 `DEFINITION = ToolDefinition(...)`：

| 字段 | 含义 |
| --- | --- |
| `name` | 模型调用时使用的工具名，例如 `read_file` |
| `description` | 能做什么、何时使用、有哪些限制 |
| `input_schema` | 参数的 JSON Schema，包括类型、必填字段及取值约束 |
| `requires_confirmation` | 内部标志，默认 `False`；写文件和执行命令设为 `True`，不发送给模型 |
| `supports_cancellation` | 内部标志，默认 `False`；命令执行和搜索设为 `True`，接收会话取消事件，不发送给模型 |

模块还需提供 `execute(arguments, workspace)` 执行函数，成功时返回含简短 `message` 的结果字典，可预期错误使用 `ToolError(code, message)`。定义与执行函数分开，执行函数不会发送给模型。

声明支持取消的工具接受额外关键字参数 `abort=None`，由执行层传入会话取消事件。已启动命令发生非零退出、超时或取消时返回 `status: "error"`，执行层保留其状态，同时标记 `executed: true`；无法启动或无效参数才属于未执行。

启动时，`registry.py` 自动发现本包直属工具模块，跳过基础模块、下划线开头的辅助模块和子包，通过 `DEFINITION` 与 `execute` 建立名称映射。重复名称、缺失定义或不可调用的实现会明确报错。新增工具只需按上述约定添加模块并重启程序，无需维护工具列表或修改查询循环。

引擎通过 `get_tool_definitions()` 获取描述，内部 `input_schema` 转换为 DeepSeek 的 `function.parameters`。`executor.py` 负责查找工具、校验参数、取得必要的本地确认、调用实现并补充统一结果字段；当前支持本项目使用的对象、字符串、整数及必填、额外字段、最短长度和整数上下界规则，未支持的规则会拒绝执行。文件路径、类型、编码和大小等语义检查由具体工具负责。

读文件参数：

| 参数 | 规则 |
| --- | --- |
| `path` | 必填，非空字符串，文件的相对或绝对路径 |
| `offset` | 可选，从第几行开始，默认 `0`，必须为非负整数 |
| `limit` | 可选，最多读取多少行，必须为正整数；省略则读取剩余行 |

模型调用示例，读取第 11 行开始的最多 20 行：

```json
{"name": "read_file", "arguments": "{\"path\":\"README.md\",\"offset\":10,\"limit\":20}"}
```

读取成功的结果含 `status: "success"`、`executed: true`、`tool`、`path`、`content` 和 `message`；`path` 是相对工作目录的路径，`content` 是选定行的原文。失败时返回 `status: "error"`、`executed: false`、`tool`、`code` 和 `message`，例如 `not_found` 表示文件不存在，`permission_denied` 表示权限不足。结果通过原始 `tool_call_id` 回传，模型可据此修正路径继续调用。终端只显示 `[工具] read_file：……` 状态，内容交给模型用于回答。

读取规则：

- 根目录固定为命令行会话启动时的工作目录，`QueryState` 单独使用时在创建时固定；相对路径和目录内的绝对路径均可使用。
- 仅读取该目录及子目录内的普通文件；解析软链接后重新检查，拒绝越界及 `.env*`、`.git` 路径。
- 仅支持 UTF-8 文本，保留原文换行，空文件有效；目录、特殊文件、无效编码、含空字符的二进制内容均返回错误。
- `offset` 从 0 计数，超过文件末尾时成功返回空内容；`limit` 超过剩余行数时返回剩余全部内容。
- 整文件最大 1 MiB，按行读取也受此限制；大小检查后仍限制实际读取量，文件过大时返回 `file_too_large`，不尝试整文件载入。
- 读取成功后仍应用下述 6000 字符工具结果限制。文件内容作为分析资料处理，不作为新的用户指令。

### 关键词搜索

`grep` 使用与读文件相同的自动注册方式，不需要确认。可以直接输入「在 .py 文件中搜索 query_loop，告诉我文件名和行号」，模型会选择工具并根据搜索结果回答。

| 参数 | 规则 |
| --- | --- |
| `keyword` | 必填，非空、单行的字面关键词，区分大小写，不使用正则表达式 |
| `path` | 可选，搜索目录或单个文件，默认 `.`；目录会递归搜索 |
| `glob` | 可选，按文件名筛选，默认 `*`；例如 `*.py` 只搜索 Python 文件 |
| `max_results` | 可选，最多返回的匹配行数，默认 `100`，整数范围 `1`～`500` |

模型传入的参数示例：

```json
{"keyword": "query_loop", "path": "harness", "glob": "*.py", "max_results": 20}
```

结果的 `matches` 数组中，每项包含相对工作目录的 `path`、从 1 开始的 `line_number`、该行 `content` 和 `line_truncated`。同一行出现多次关键词也只返回一项，无匹配时返回空数组。`returned_count` 是本次返回条数，不代表所有文件中的总匹配数。

超过条数或长度预算时，保留完整匹配条目并设置 `truncated: true`，可缩小 `path` 或 `glob` 后重搜。包含执行层字段的结果保持在默认 6000 字符上限内；超长行保留关键词附近片段和完整关键词，并设 `line_truncated: true`。关键词本身过长、无法放入结果预算时会要求缩短。

搜索沿用会话启动目录限制，不读取 `.env*`、`.git` 或越界文件，不跟随目录软链接。仅搜索不超过 1 MiB 的 UTF-8 普通文件；不可读、二进制、编码无效或过大的文件会跳过，结果包含 `skipped_files` 和 `skipped_directories` 计数。指定的根路径不存在或不允许访问时，错误会回传模型。搜索期间支持会话取消，管道模式也能使用此只读工具。

### 写文件与确认

`write_file` 接受两个必填字符串参数：`path` 指定目标文件，`content` 是将写入的完整内容。内容允许为空，已有文件会覆盖全文，不追加。

在交互终端中，可以直接输入「帮我在 notes/hello.txt 写入你好」。模型提出写入后，程序展示目标路径、完整内容和新建／覆盖提示，等待您输入：

| 输入 | 行为 |
| --- | --- |
| `y` | 批准这一次写入，缺失的父目录自动创建 |
| `n` 或直接回车 | 拒绝，原文件和目录保持不变，拒绝结果回传给模型 |
| `/cost`、`/help` | 查看费用或帮助，继续等待确认 |
| `/exit` 或 Ctrl+C | 取消并退出，不执行待确认的写入 |

每次调用单独确认，没有自动批准或「全部同意」。预览不受工具结果 6000 字符截断限制；特殊控制字符以转义形式展示，避免影响终端显示，实际内容保持原样。输入流关闭也会拒绝等待中的写入；管道输入或重定向输出不提供确认，写工具返回未批准结果，读文件、搜索和普通回答仍可使用。

收到批准后，会重新核对目标路径和文件是否存在。如果等待期间目标文件被创建、删除，或软链接改了指向，本次确认失效并拒绝写入，需要重新发起请求。

写入 UTF-8 普通文本，最多 1 MiB，沿用会话启动目录及 `.env*`、`.git`、越界软链接限制。确认之后才创建目录和写入；成功结果包含相对路径、写入字节数和简短状态，不重复回传全文。参数、权限或写入失败均作为工具错误交给模型。

确认由本地执行层强制要求，模型不能传入 `approved` 等字段来批准自己。直接使用 `QueryState` 的默认执行器没有确认入口，会返回 `confirmation_required`；嵌入其他程序时可用 `create_tool_executor(workspace, confirm=callback, abort=event)` 注入本地交互。回调接收 `(name, arguments, workspace)`，应先展示待执行内容并获取用户批准，仅返回布尔值 `True` 才会执行。

### 终端命令

`bash` 与读写工具放在同一目录，通过 `DEFINITION` 与 `execute` 自动注册。可以向模型输入「运行 python3 --version」，或「执行项目测试，超时设为 60 秒」。执行前会展示完整命令、工作目录和超时，仅输入 `y` 批准本次执行；`n`、回车、EOF 或非交互模式均不批准。等待确认或命令运行期间，`/cost` 和 `/help` 仍可用。

| 参数 | 规则 |
| --- | --- |
| `command` | 必填，非空命令字符串，支持 Bash 管道和重定向 |
| `timeout` | 可选，超时秒数，默认 `30`，整数范围 `1`～`120` |

命令从会话启动目录运行，每次启动独立 `/bin/bash`，不读取 shell 启动脚本，标准输入关闭，不支持需要用户持续输入的终端程序。子进程环境不继承 Harness 的 `DEEPSEEK_API_KEY`，其余使用当前进程环境。当前实现支持 macOS／Linux；命令具有当前用户权限，起始工作目录不是文件访问沙箱，文件读写工具的路径隔离不适用于任意 shell 命令。

命令结果分别回传两路输出，即使退出码非零或超时，也保留已经产生的输出：

| 结果字段 | 含义 |
| --- | --- |
| `stdout`、`stderr` | 标准输出和错误输出，UTF-8 解码，异常字节以替换字符显示 |
| `exit_code` | 实际进程退出码，通常 `0` 表示成功；被信号终止时可为负数，无法取得时为 `null` |
| `timed_out`、`cancelled` | 是否因超时或会话取消而停止 |
| `status`、`executed` | 命令成功为 `success`，非零退出／超时／取消为 `error`；已经启动的命令仍是 `executed: true` |
| `stdout_truncated`、`stderr_truncated` | 对应输出是否被截断 |
| `stdout_bytes`、`stderr_bytes` | 实际读取的两路输出总字节数，包括被丢弃的部分 |

程序同时持续读取两路管道，每路最多保留 64 KiB，超限保留首尾并标明截断；回传模型前进一步按 6000 字符限制缩短两路输出，保留退出码和超时等字段。判断成功应同时检查退出码及超时／取消标记，不能仅凭有没有错误输出。

超时或 `/exit`、Ctrl+C 取消时，会终止同一进程组中的命令与子进程，并进行有限时间的输出收尾和进程回收。CLI 退出时最多等待后台任务 1 秒完成清理，不无限等待在途模型请求。当前不提供持久后台作业或操作系统沙箱，主动脱离进程组的程序不属于进程组清理范围。

进程管理采用 Python 标准库的 [subprocess](https://docs.python.org/3/library/subprocess.html) 和 [os.killpg](https://docs.python.org/3/library/os.html#os.killpg) 。

## 上下文压缩与工具结果限制

| 规则 | 默认值与行为 |
| --- | --- |
| 自动触发 | 每次正常模型请求前检查 `messages` 和 `tools` 的序列化长度，超过 24000 字符时压缩 |
| 保留历史 | 原始系统提示、最近 4 个完整用户轮次和当前未完成轮；工具调用与对应结果整组保留 |
| 摘要 | 独立调用 DeepSeek，总结目标、约束、关键事实与决策、进展和待办，最多 2000 字符 |
| 压缩次数 | 每次用户提问或 `/compact` 最多尝试 2 次，计数贯穿整个工具循环 |
| 长度仍超限 | 提示「太长了，建议开个新会话」；保留部分本身超限时直接提示 |
| 工具结果 | 每个结果序列化后最多 6000 字符；普通结果用 `head`／`tail` 保留首尾，命令结果分别缩短 `stdout`／`stderr` 并保留退出码等字段，搜索预先限长并保留完整匹配条目，均标注截断 |

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
| `harness/tools/definition.py` | `ToolDefinition` 及 DeepSeek 格式转换 |
| `harness/tools/registry.py` | 自动发现、注册和查找工具，生成描述列表 |
| `harness/tools/executor.py` | 统一参数校验、本地确认、固定工作目录、分发执行与 `ToolError` |
| `harness/tools/read_file.py` | 按行读取文件及路径、类型、大小和编码校验 |
| `harness/tools/write_file.py` | 确认后写入 UTF-8 文本、创建父目录及路径限制 |
| `harness/tools/bash.py` | 确认后运行命令、超时与取消清理、双路输出与退出码 |
| `harness/tools/grep.py` | 关键词搜索、文件名模式筛选、行号与内容回传及结果限长 |
| `harness/usage.py` | 逐请求 token 记录、模型费率与费用汇总 |
| `harness/cli.py` | 终端输入、回答和斜杠命令 |
| `tests/` | 模拟接口、工具、计费与 CLI 测试 |

## 验证

```sh
python3 -m unittest discover -s tests -v
```

测试覆盖纯文字回答、多工具与多轮调用、临时文件读写及错误恢复、关键词与文件类型搜索、行号及完整条目截断、写入和命令预览与逐次确认、拒绝和取消、命令两路输出与退出码、超时及同组进程清理、路径与软链接限制、连续对话、压缩与回滚、工具结果限长、递增重试与取消、限速显示、请求上限、token 和峰谷费用、`/cost`、`/compact`、启动及 `.env` 加载行为。文件和命令测试使用临时目录及无害本地子进程，接口与配置测试使用模拟内容，不读取实际密钥、不产生真实 API 费用。真实 API 联调尚未验证。

官方依据：[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 、[Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/) 、[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) 、[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/) 、[Error Codes](https://api-docs.deepseek.com/quick_start/error_codes/) 。
