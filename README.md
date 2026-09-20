# harness

一个类似 Claude Code 核心查询循环的最小命令行实现，模型使用 DeepSeek。Python 3.11+；查询、工具、记忆和笔记核心继续使用标准库，终端渲染使用 `rich`。

最近发布：**V0.7.1 — Add-钩子机制**（Git 标签 `v0.7.1`）。在 V0.6.3.1 智能记忆基础上增加生命周期 Hooks 配置。版本记录见 [CHANGELOG.md](CHANGELOG.md) 。

## 启动

从 GitHub 下载项目并进入目录：

```sh
git clone https://github.com/MichaelJiangg/harness.git
cd harness
```

首次使用，在项目根目录创建 `.env` 并填写密钥（该文件不会随仓库发布）：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
TAVILY_API_KEY=你的Tavily密钥
```

之后每次在项目目录直接运行：

```sh
python3 -m pip install "rich>=13.0"
python3 -m harness
```

启动时自动读取项目根目录 `.env`，无需每次 `export`。如果当前进程已设置 `DEEPSEEK_API_KEY`，环境变量优先于文件（包括已设置的空值）。支持单行值、单／双引号、注释和可选 `export` 前缀，不执行 shell 命令或变量插值；仅读取该密钥，不注入其他变量。`.gitignore` 已排除 `.env*`，不要将真实密钥提交到 Git。

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
| `/mode ask|auto [目录]` | 切换权限模式；auto 模式信任当前或指定工作目录 |
| `/memory [list\|show <id>\|delete <id> --yes\|clear --yes]` | 查看和管理本地会话记忆 |
| `/recall <query> [--limit N\|--threshold T]` | 搜索语义相关的历史会话记忆 |
| `/notes [append <text>\|replace --yes <text>\|clear --yes]` | 查看和编辑项目长期笔记 |
| `/activity [latest\|all\|clear]` | 查看后台工具、请求、压缩和编排活动 |
| `/help` | 查看帮助 |
| `/exit` | 退出，停止后续模型和工具调用 |

`python3 -m harness --help` 无需密钥即可查看帮助。支持单条管道输入，输入流结束后会等待回答；交互中一次处理一个问题，繁忙时的新问题会被提示稍后重发。

## 会话记忆

正常通过 `/exit` 或输入结束退出时，Harness 会调用 DeepSeek 把本次对话压缩成一条摘要，保存到当前工作区的 `.harness/memory.json`。下次启动时自动加载最近 5 条摘要，作为背景注入系统提示，模型据此回答“上次讨论了什么”之类的问题。

每条记录包含数字编号、UTC 日期、摘要和话题标签。摘要最多 1000 字，文件最多保留 20 条记录；文件采用 JSON 格式，权限为 `0600`，通过同目录临时文件原子替换，损坏时不会阻止 CLI 启动。摘要请求只提交用户和助手文字以及工具名称，不提交工具结果正文，避免把文件或命令输出写入记忆。

安装 ChromaDB 后，Harness 会把摘要写入本地向量索引，并在新问题时检索语义相关记忆：

```bash
python3 -m pip install chromadb
```

未安装 ChromaDB 时，记忆功能继续使用最近 N 条基础模式；`/recall <query>` 会退化为本地文本相关度搜索：

```text
你 > /recall 数据库性能
[0.92] 2026-09-20 — 数据库连接池配置
[0.85] 2026-09-19 — PostgreSQL 索引优化讨论
```

`/recall` 支持 `--limit N` 和 `--threshold T`。

系统提示按“项目笔记 → 最近会话 → 冷记忆”的顺序注入。冷记忆优先级最低；记忆总量超过预算时先裁剪冷记忆，再裁剪较旧的会话摘要，项目笔记始终保留。

```text
你 > /memory
[1] 2026-09-20 — 技术选型：FastAPI + PostgreSQL
    话题：技术选型、FastAPI、PostgreSQL

你 > /memory delete 1 --yes
已删除记忆 #1。
```

`delete` 和 `clear` 必须显式带 `--yes`。`python3 -m harness` 默认启用记忆；直接调用 `run_cli` 的嵌入场景默认关闭，避免第三方程序在退出时产生额外的模型请求。

记忆是本地明文 JSON，不是加密或防篡改存储；拥有当前用户权限的 Bash 命令仍可访问该文件。

## 项目笔记

项目根目录的 `HARNESS.md` 保存长期有效的项目知识，包括技术栈、编码约定、架构决定、已知问题和当前进展。CLI 启动时自动读取并注入系统提示，因此下个会话可以直接引用其中的规范。

AI 收到「记住：我们的 API 前缀统一用 /api/v2」这类明确要求，或在对话中发现值得长期保留的项目决定时，会调用 `notes_append` 追加 Markdown 内容。用户也可以在终端直接管理：

```text
你 > /notes
HARNESS.md：
# 编码规范
- 格式化工具：Black，行宽 88
- Lint 工具：ruff

你 > /notes append - 后端：FastAPI
已追加项目笔记（24 字节）。
```

`/notes replace --yes <text>` 替换整个文件，`/notes clear --yes` 清空文件。笔记固定指向会话工作区根目录的 `HARNESS.md`，不接收其他路径；文件最多 65536 字节，拒绝软链接、目录和二进制内容。普通追加默认放行，整篇替换需要用户确认，`deny` 规则仍可禁止。

## 终端渲染

交互终端使用 `rich` 渲染：AI 回答以 Markdown 面板显示，支持标题、粗体、斜体、列表、链接和代码块语法高亮；工具调用显示参数与结果摘要；用户输入、系统提示、错误、后台任务、委托和 Swarm 状态使用不同颜色。管道和非交互输出继续使用无 ANSI 的纯文本，方便脚本处理。

Harness 默认强制启用交互终端颜色；管道输出不会混入终端控制序列。

工具执行、逐请求用量和上下文压缩事件默认进入会话内后台活动日志，不打断对话；输入 `/activity` 可展开查看工具参数、结果摘要、用量、压缩、重试和编排事件。权限确认和错误仍直接显示。

`web_fetch(url)` 可以读取用户明确提供的公开网页，但不提供搜索。URL 仅限 80/443 端口，禁止本机、私网、链路本地、重定向和超过 1 MiB 的响应，网页脚本不会执行；默认需要确认。

`web_search(query, max_results=5)` 通过 Tavily Search API 搜索公开网页。API key 放在项目根目录 `.env`：

```dotenv
TAVILY_API_KEY=你的Tavily密钥
```

`.env*` 已被 `.gitignore` 排除；代码、工具参数、权限日志和 GitHub 发布清单不会包含该密钥。

## Hooks

用户可以在当前工作区 `.harness/hooks.json` 注册生命周期逻辑：

```json
{
  "hooks": [
    {
      "event": "session_start",
      "type": "shell",
      "name": "记录启动",
      "command": "echo started >> /tmp/harness.log"
    },
    {
      "event": "before_send_message",
      "type": "prompt",
      "name": "回答约束",
      "prompt": "回答保持简洁。"
    }
  ]
}
```

支持事件：`session_start`、`session_end`、`before_send_message`、`after_reply`、`before_tool`、`after_tool`。Hook 类型：

| 类型 | 说明 |
| --- | --- |
| `shell` | 执行一条 Bash 命令，独立超时，默认 30 秒，最长 120 秒 |
| `prompt` | 注入一段提示词；主要用于 session_start 和 before_send_message |
| `python` | 从 `.harness/hook_functions.py` 加载函数 |

Shell Hook 的环境会清除 DeepSeek/Tavily 密钥，并注入 `HARNESS_EVENT`、`HARNESS_WORKSPACE`、`HARNESS_TOOL`。Hook 异常不会中止查询。

## 项目配置

项目根目录的 `pyproject.toml` 集中管理元信息及非密钥默认配置。启动时通过 Python 标准库 `tomllib` 读取、验证并保存快照，修改配置后重启生效。配置文件位置固定在 Harness 源码根目录，不会从所操作的其他目录加载同名文件；当前仍使用 `python3 -m harness` 从源码运行。

| 配置位置 | 内容 |
| --- | --- |
| `[project]` | 项目名称、版本、描述、Python 版本与依赖 |
| `[tool.harness.model]` | 模型名称、HTTPS 接口地址、请求超时 |
| `[tool.harness.display]` | 逐字显示间隔，`character_delay = 0.02` 表示 20 毫秒 |
| `[tool.harness.engine]` | 请求上限、重试次数、首次等待与递增倍数 |
| `[tool.harness.background]` | 后台任务并发上限与默认超时 |
| `[tool.harness.security]` | 默认权限模式与 auto 模式信任目录 |
| `[tool.harness.swarm]` | Swarm 团队总请求上限与单角色请求上限 |
| `[tool.harness.context]` | 上下文与摘要长度、保留轮数、压缩次数、工具结果长度 |
| `[tool.harness.tools]` 及其 `bash`、`grep` 子表 | 文件大小、命令超时与输出、搜索条数与长度限制 |
| `[tool.harness.pricing]` 及其 `peak` 子表 | 现有费率、币种、核验信息及 UTC 高峰时段 |
| `[tool.harness.permissions]` | 工具的放行、询问和拒绝规则 |

运行配置在 `tool.harness` 下要求完整字段，不接受未知键、无效类型或超出对应范围的数值；默认值必须符合对应上限。搜索结果预算必须比总工具预算至少少 500 字符，且至少为单行字符限制的 6 倍加 400 字符，为 JSON 转义和结果字段留出空间。命令每路输出预算最少 128 字节，足以容纳首尾片段与截断提示。配置不存在或错误时停止启动并显示字段说明，不回退到更宽松的权限。

API key 继续放在 `.env` 或环境变量中，不能迁入 `pyproject.toml`。`CLAUDE.md` 保留开发规范，`ROADMAP.md` 保留进度。下文数值均为配置文件的初始默认值，修改后会同时更新实际执行、工具参数约束和相应说明。

## 权限检查

每次工具调用先校验参数，再由 `harness/permissions.py` 决定是否执行；权限判断、必要的本地确认及审计落盘都在调用工具实现之前完成，执行前再次复核权限。模型不能通过参数批准自己。

默认按工具和命令风险分为三级：

| 风险等级 | 工具 | 行为 |
| --- | --- | --- |
| 低风险：读取 | `read_file`、`grep`，明确识别的简单 `pwd`／`ls` 命令 | 直接执行，无需确认 |
| 中风险：写入或未知操作 | `write_file`、普通或未识别的 Bash 命令、新工具 | 预览操作，输入 `y` 后执行；已授权目录中的写入可免确认 |
| 高风险：破坏性命令 | 匹配破坏性启发式的 Bash 命令 | 先显示醒目文字警告及潜在影响，再展示完整命令、工作目录和超时，输入 `y` 后执行 |

高风险警告使用文字与醒目边框，不依赖终端颜色；高风险分类本身要求确认，不等同于永久禁止。必须永远拒绝的操作应配置 `deny` 规则。需要确认的 Bash 调用逐次询问，不记忆命令授权。

```toml
[tool.harness.permissions]
allow = ["read_file", "grep", "delegate", "notes_append"]
ask = ["write_file", "notes_replace", "web_fetch", "web_search"]
deny = []
rules = []
```

| 决策 | 行为 |
| --- | --- |
| `allow` | 直接执行；仍须满足工具自身的路径、类型和大小限制 |
| `ask` | 展示操作内容，等待本次明确输入 `y` 后执行 |
| `deny` | 直接返回 `permission_denied` 给模型，不进入确认或执行 |

工具名列表使用精确名称，优先级为 `deny` 高于 `ask` 高于 `allow`。未配置的工具按风险等级兜底：读取、搜索及明确识别的只读命令默认 `allow`，写入、其他 Bash 和新注册工具默认 `ask`；未注册工具仍返回 `unknown_tool`。将 `bash` 加入 `deny` 即可禁止所有命令，加入 `ask` 则连 `pwd` 也要求确认。

会话支持两种权限模式：

| 模式 | 行为 |
| --- | --- |
| `ask` | 默认模式，按规则和风险逐条确认 |
| `auto` | 信任当前工作目录或 `/mode auto <dir>` 指定目录；读取、写入、验证和常见 Node/Python/Perl 脚本自动放行 |

auto 模式不覆盖 `deny`，也不放行网络下载、`rm`、进程管理、写敏感路径、越界路径或任意重定向。安全的管道、`2>&1`、`2>/dev/null`、`$(pwd)` 和 heredoc 脚本在信任目录内可自动执行。所有自动放行的操作仍记录审计日志，确认状态为 `auto`。

仅将 `write_file` 或 `bash` 加入工具名 `allow` 列表，不会免除写入或非只读命令的确认；写入可以通过目录规则或当前会话的目录授权放行。需要确认的新工具默认显示中风险、工具名和完整参数 JSON；读取工具或只读命令配置为 `ask` 时仍显示低风险，但必须确认。

拒绝、空输入、EOF、取消或没有确认入口时，均不执行需要确认的工具；管道模式仍可使用被允许的工具。配置在当前进程内不会热加载，修改 TOML 不会让当前会话自动获得新权限。

拒绝结果通过对应的 `tool_call_id` 回传模型，说明「哪个工具被拒绝」「配置规则、未获批准等具体原因」「可采取的安全下一步」。消息不会复述文件正文或原始命令参数，并要求模型向用户说明限制，不得改用其他工具绕过禁止。

### Bash 启发式检测

`classify_bash_risk` 返回 `read_only`、`write` 或 `destructive`，对应上述低、中、高风险。它检查以下五类信号：

| 检查 | 示例与处理 |
| --- | --- |
| 危险关键词 | 递归删除、`mkfs`、`dd if=`、`chmod -R 777`、写入磁盘设备及 Git 强制推送，提升到高风险 |
| 下载并执行管道 | `curl url \| bash`、`wget url -O- \| sh` 等，标为高风险 |
| 敏感路径 | 命令参数涉及 `/etc/`、`/usr/`、`/System/`、`~/.ssh/` 等，至少中风险 |
| 命令组合 | 对 `&&`、`\|\|` 连接的简单命令分别判断，取最高风险；重定向、复杂语法等至少中风险 |
| 环境修改 | 赋值或 `export`，包括 `PATH`、`LD_PRELOAD` 等变量，至少中风险 |

仅明确识别的 `pwd`、`ls` 及认可参数可以自动放行。未知命令、解释器、包装命令、变量展开、脚本和无法可靠识别的语法不会默认只读。检测是保守启发式，不是完整 Shell 解析或操作系统沙箱；文本可能误报，也不能穷尽变量拼接或外部脚本中的行为。

### 按目录和命令配置规则

项目默认 `rules = []`，不预先启用目录放行。启用以下示例时，删除空的 `rules = []`，在 TOML 中增加这两段数组表；不能同时保留同名空数组和数组表：

```toml
[[tool.harness.permissions.rules]]
name = "allow-tests-writes"
tool = "write_file"
action = "allow"
directory = "tests"
priority = 100

[[tool.harness.permissions.rules]]
name = "deny-recursive-force-remove"
tool = "bash"
action = "deny"
command_pattern = '\brm\s+-(?:rf|fr)\b'
priority = 200
```

第一条让会话启动目录中的 `tests/` 及子目录写入免确认；第二条禁止命令原文中匹配正则的调用，包括复合命令中的 `rm -rf`、`rm -fr`，禁止结果直接回传模型，不显示批准提示。

| 字段 | 规则 |
| --- | --- |
| `tool` | 必填，精确工具名 |
| `action` | 必填，`allow` 放行、`ask` 确认、`deny` 拒绝 |
| `priority` | 可选整数，默认 `0`；同类规则数值越大越优先，同值按声明顺序 |
| `name` | 可选，最多 128 个可显示字符，不得与其他规则重名，用于拒绝说明与审计定位；省略时生成稳定的规则标识 |
| `directory` | 可选，仅用于 `read_file`／`write_file`，相对目录及全部子目录；不允许绝对路径或 `..` |
| `command_pattern` | 可选，仅用于 Bash，用 Python 正则搜索完整命令原文 |

匹配核心位于 `harness/permissions.py`：`PermissionPolicy` 先检查工具名 `deny` 列表；随后 `check_permission` 按「禁止规则在前、各类内部按 `priority` 降序、同级按声明顺序」遍历规则，第一条匹配决定结果。没有匹配时才使用工具名列表和风险默认策略。因此高数值的 `allow` 不能压过匹配的 `deny`，但可以覆盖工具名 `ask` 的默认确认。写入 `allow` 必须带 `directory`；Bash 不接受强制 `allow` 规则，未命中规则时按命令风险决定是否确认。

目录按路径组件匹配，`tests2/` 不属于 `tests/`。放行要求标准化路径与解析软链接后的实际路径都在授权目录内；询问／禁止只需任一路径命中，且保守覆盖大小写或 Unicode 规范形式不同的目录名，包括尚未创建的目录。在区分大小写的文件系统中，这些不同拼写也会一起受限；放行仍要求精确路径拼写。`grep` 递归搜索仅支持工具级规则，不支持 `directory` 条件。

已有工作目录、`.env*`、`.git`、文件类型与大小限制继续有效，新增 `.harness/` 审计目录保护。执行前会再次检查规则，避免等待确认时软链接改指向禁止目录；权限条件变化时要求重新发起。路径检查异常直接返回 `permission_check_failed`，不执行工具。路径检查与文件操作并非抵御并发外部修改的原子沙箱。

命令规则只匹配文本，不解析 Shell：示例也会拦截 `echo 'rm -rf tmp'`，但不识别变量拼接、外部脚本等任意等价写法；未命中禁止规则也不代表自动放行，仍需经过风险判断。规则配置由用户维护，错误字段、类型或无效正则会阻止启动，修改配置后需重新启动会话。

### 会话授权与审计

CLI 首次确认写入时，除完整内容外，还会显示「本会话授权目录」，范围为目标文件的父目录及全部子目录。输入 `y` 批准且写入成功后，同一 CLI 会话内后续写入该范围不再询问；失败、拒绝、取消或确认期间目标改变不新增授权。权限缓存只在内存中保存，退出或新建执行器后失效；任何匹配的禁止规则仍优先于缓存。Bash 不复用确认。

权限决策与确认结果保存在会话工作目录的 `.harness/permission.log`，每行一个 JSON 对象。记录包含 UTC 时间、`source = "tool_executor"`、`session_id`、执行器生成的 `call_id`、工具名、风险、`matched_rule`、决策、事件类型和 `confirmation` 状态，便于区分规则放行、目录授权复用与用户批准／拒绝。

参数采用白名单脱敏：路径最多 512 个字符，保留必要的分页／超时等数值；正文、关键词和原始命令仅保留长度，命令另存 SHA-256 指纹。未知字段和完整内容不进入日志。日志目录使用 `0700`、文件使用 `0600`，拒绝软链接、硬链接和特殊文件；决策与确认结果必须在工具执行前写入并同步，失败返回 `audit_failed` 并阻止工具运行。

`.harness/` 已加入 `.gitignore`，内置读文件、写文件和搜索工具也将其视为受保护路径。审计是本地诊断记录，不是防篡改存储，也不限制获得当前用户权限的 Bash 命令访问该目录。

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
- 当前开放只读的 `read_file`、`grep`、`background_check`，默认需确认且可按目录授权的 `write_file`、按命令风险决定确认的 `bash`、受限验证 `run_verify`、后台任务提交 `background_submit`、多角色团队协作 `swarm`，以及独立子任务 `delegate`；工具描述随每次正常模型请求发送。
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
├── executor.py      # 参数校验、权限检查、本地确认、执行分发
├── read_file.py     # 读文件的定义与具体实现
├── write_file.py    # 写文件的定义与具体实现
├── bash.py          # 命令执行、超时清理和两路输出收集
├── grep.py          # 关键词搜索、文件名筛选和结果限长
├── delegate.py      # 独立子任务
├── background_submit.py  # 后台任务提交
├── background_check.py   # 后台任务状态查询
├── run_verify.py    # 受限 Node／Python 验证
└── swarm.py         # 多角色团队协作
```

每个工具模块提供 `DEFINITION = ToolDefinition(...)`：

| 字段 | 含义 |
| --- | --- |
| `name` | 模型调用时使用的工具名，例如 `read_file` |
| `description` | 能做什么、何时使用、有哪些限制 |
| `input_schema` | 参数的 JSON Schema，包括类型、必填字段及取值约束 |
| `supports_cancellation` | 内部标志，默认 `False`；命令执行和搜索设为 `True`，接收会话取消事件，不发送给模型 |

模块还需提供 `execute(arguments, workspace)` 执行函数，成功时返回含简短 `message` 的结果字典，可预期错误使用 `ToolError(code, message)`。定义与执行函数分开，执行函数不会发送给模型。

声明支持取消的工具接受额外关键字参数 `abort=None`，由执行层传入会话取消事件。已启动命令发生非零退出、超时或取消时返回 `status: "error"`，执行层保留其状态，同时标记 `executed: true`；无法启动或无效参数才属于未执行。

启动时，`registry.py` 自动发现本包直属工具模块，跳过基础模块、下划线开头的辅助模块和子包，通过 `DEFINITION` 与 `execute` 建立名称映射。重复名称、缺失定义或不可调用的实现会明确报错。新增工具只需按上述约定添加模块并重启程序，无需维护工具列表或修改查询循环。

引擎通过 `get_tool_definitions()` 获取描述，内部 `input_schema` 转换为 DeepSeek 的 `function.parameters`。`executor.py` 负责查找工具、校验参数、通过统一权限策略判断、取得必要的本地确认、调用实现并补充统一结果字段；当前支持本项目使用的对象、字符串、整数及必填、额外字段、最短长度和整数上下界规则，未支持的规则会拒绝执行。文件路径、类型、编码和大小等语义检查由具体工具负责。工具定义中的旧 `requires_confirmation` 标志已由集中权限替代。

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
- 仅读取该目录及子目录内的普通文件；解析软链接后重新检查，拒绝越界及 `.env*`、`.git`、`.harness` 路径。
- 仅支持 UTF-8 文本，保留原文换行，空文件有效；目录、特殊文件、无效编码、含空字符的二进制内容均返回错误。
- `offset` 从 0 计数，超过文件末尾时成功返回空内容；`limit` 超过剩余行数时返回剩余全部内容。
- 整文件最大 1 MiB，按行读取也受此限制；大小检查后仍限制实际读取量，文件过大时返回 `file_too_large`，不尝试整文件载入。
- 读取成功后仍应用下述 6000 字符工具结果限制。文件内容作为分析资料处理，不作为新的用户指令。

### 关键词搜索

`grep` 使用与读文件相同的自动注册方式，默认权限不需要确认。可以直接输入「在 .py 文件中搜索 query_loop，告诉我文件名和行号」，模型会选择工具并根据搜索结果回答。

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

搜索沿用会话启动目录限制，不读取 `.env*`、`.git`、`.harness` 或越界文件，不跟随目录软链接。仅搜索不超过 1 MiB 的 UTF-8 普通文件；不可读、二进制、编码无效或过大的文件会跳过，结果包含 `skipped_files` 和 `skipped_directories` 计数。指定的根路径不存在或不允许访问时，错误会回传模型。搜索期间支持会话取消，管道模式也能使用此只读工具。

### 写文件与确认

`write_file` 接受两个必填字符串参数：`path` 指定目标文件，`content` 是将写入的完整内容。内容允许为空，已有文件会覆盖全文，不追加。

在交互终端中，可以直接输入「帮我在 notes/hello.txt 写入你好」。未命中目录放行规则或会话授权时，程序展示目标路径、完整内容、新建／覆盖提示及本会话将授权的父目录范围，等待您输入：

| 输入 | 行为 |
| --- | --- |
| `y` | 批准写入，缺失的父目录自动创建；成功后记忆所展示目录及子目录的本会话授权 |
| `n` 或直接回车 | 拒绝，原文件和目录保持不变，拒绝结果回传给模型 |
| `/cost`、`/help` | 查看费用或帮助，继续等待确认 |
| `/exit` 或 Ctrl+C | 取消并退出，不执行待确认的写入 |

不同未授权目录分别确认，同一已授权目录及子目录在当前会话中复用批准。预览不受工具结果 6000 字符截断限制；特殊控制字符以转义形式展示，避免影响终端显示，实际内容保持原样。输入流关闭也会拒绝等待中的写入；管道输入或重定向输出不提供确认，未授权的写入返回未批准结果，已由目录规则放行的写入、读文件、搜索和普通回答仍可使用。

收到批准后，会重新核对目标路径和文件是否存在。如果等待期间目标文件被创建、删除，或软链接改了指向，本次确认失效并拒绝写入，需要重新发起请求。

写入 UTF-8 普通文本，最多 1 MiB，沿用会话启动目录及 `.env*`、`.git`、`.harness`、越界软链接限制。规则放行或用户确认之后才创建业务目录和写入；成功结果包含相对路径、写入字节数和简短状态，不重复回传全文。参数、权限或写入失败均作为工具错误交给模型。

确认由本地权限与执行层强制要求，模型不能传入 `approved` 等字段来批准自己。直接使用 `QueryState` 的默认执行器没有确认入口，需要询问时会返回 `confirmation_required`；嵌入其他程序时可用 `create_tool_executor(workspace, confirm=callback, abort=event, permissions=policy)` 注入本地交互和 `PermissionPolicy`，省略策略则使用启动配置。回调接收 `(name, arguments, workspace)`，应先展示待执行内容并获取用户批准，仅返回布尔值 `True` 才会执行。

直接嵌入默认仍是单次确认，不会把已有回调自动扩大为目录授权。需要会话记忆时，显式传入 `session_cache=SessionPermissionCache()`，并由确认回调清楚展示父目录及子目录的授权范围；为每个新会话创建独立缓存。

### 终端命令

`bash` 与读写工具放在同一目录，通过 `DEFINITION` 与 `execute` 自动注册。可以向模型输入「运行 python3 --version」，或「执行项目测试，超时设为 60 秒」。明确识别的简单只读命令默认直接执行；其他调用会展示风险、完整命令、工作目录和超时，高风险另加醒目警告，仅输入 `y` 批准本次执行。需要确认时，`n`、回车、EOF 或非交互模式均不批准。等待确认或命令运行期间，`/cost` 和 `/help` 仍可用。

| 参数 | 规则 |
| --- | --- |
| `command` | 必填，非空命令字符串，支持 Bash 管道和重定向 |
| `timeout` | 可选，超时秒数，默认 `30`，整数范围 `1`～`120` |

命令从会话启动目录运行，每次启动独立 `/bin/bash`，不读取 shell 启动脚本，标准输入关闭，不支持需要用户持续输入的终端程序。明确识别的只读命令使用受限环境及固定系统 `PATH`（`/usr/bin:/bin:/usr/sbin:/sbin`），不导入函数或动态加载器设置。其他命令保留更多当前环境，但同样清除 `DEEPSEEK_API_KEY`、`BASH_ENV`、`ENV`、`BASH_FUNC_*`、`LD_*` 与 `DYLD_*`。当前实现支持 macOS／Linux；命令具有当前用户权限，起始工作目录不是文件访问沙箱，文件读写工具的路径隔离不适用于任意 shell 命令。

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

## Agent 编排

V0.5 提供三种编排方式：

| 工具 | 用途 |
| --- | --- |
| `delegate` | 启动同步独立子查询，完成后把报告交给主 AI |
| `background_submit` | 把慢任务提交到会话级后台队列，之后用 `background_check` 查询 |
| `swarm` | 让多个角色按交接协议接力，例如 Coder、Reviewer、Tester |

委托和 Swarm 角色拥有独立上下文，只继承当前会话可见工具及权限、确认、审计和工作区。后台 Bash 任务支持最多 5 个并发、默认 300 秒超时；后台分析使用独立请求预算。Swarm 默认团队总预算 120 次、单角色 40 次，主查询自己的 20 次额度不共享。

`run_verify` 用于运行工作区内明确存在的验证脚本，支持 `node`、`python3`、`unittest`、`node-test`、`npm-test`。首次批准某目录后，当前会话内同目录及子目录的验证复用授权；写文件和危险 Bash 仍逐次确认。

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

费率及 UTC 时段保存在 `pyproject.toml` 的 `tool.harness.pricing` 中，不自动联网刷新。`peak_weekdays` 使用周一为 `0`、周日为 `6` 的编号；`peak_hours_utc` 使用左闭右开的小时区间。官方费率变化时需更新配置和核验信息。本次仅迁移现有数值，未重新核价；统计只保存在当前进程内存中，退出后清空，不代表账户全部历史消费。

如果缓存明细缺失或不一致，按全部输入未命中缓存保守估算并标注；`usage` 缺失或非法、连接失败时，不会虚构零费用，会标记用量未知。汇总此时只代表已知小计，请以账单核对未知请求。

## 文件结构与扩展点

| 文件 | 职责 |
| --- | --- |
| `pyproject.toml` | 项目元信息和非密钥运行默认配置 |
| `harness/__main__.py` | 读取配置并启动程序 |
| `harness/config.py` | 校验并缓存 TOML 配置；密钥优先读环境变量，再读取 `.env` |
| `harness/permissions.py` | 规则匹配、风险默认策略、拒绝说明与会话目录授权 |
| `harness/bash_risk.py` | Bash 启发式检测及保守只读命令识别 |
| `harness/audit.py` | 脱敏权限审计、私有日志写入与失败反馈 |
| `harness/client.py` | DeepSeek HTTP 请求、超时与错误分类 |
| `harness/engine.py` | 查询循环、有限重试、摘要压缩与逐次记账 |
| `harness/orchestration.py` | 独立委托、后台分析和 Swarm 角色编排 |
| `harness/background.py` | 后台任务状态、排队、并发限制与超时 |
| `harness/memory/` | 本地 JSON 记忆、启动注入、退出摘要和记忆命令；`session.py` 保存核心，`injection.py` 负责上下文注入 |
| `harness/memory/search.py` | 可选 ChromaDB 向量召回、文本降级和 `/recall` 相关度计算 |
| `harness/hooks.py` | 加载并执行会话、消息和工具生命周期 Hooks |
| `harness/notes.py` | HARNESS.md 读取、注入、追加、替换和路径保护 |
| `harness/context.py` | 字符预算、完整轮次切分、摘要资料和工具结果截断 |
| `harness/tools/definition.py` | `ToolDefinition`、取消能力及 DeepSeek 格式转换 |
| `harness/tools/registry.py` | 自动发现、注册和查找工具，生成描述列表 |
| `harness/tools/executor.py` | 统一参数校验、本地确认、固定工作目录、分发执行与 `ToolError` |
| `harness/tools/read_file.py` | 按行读取文件及路径、类型、大小和编码校验 |
| `harness/tools/write_file.py` | 确认后写入 UTF-8 文本、创建父目录及路径限制 |
| `harness/tools/bash.py` | 受控环境运行命令、超时与取消清理、双路输出与退出码 |
| `harness/tools/grep.py` | 关键词搜索、文件名模式筛选、行号与内容回传及结果限长 |
| `harness/tools/delegate.py` | 独立子任务 Schema 与运行入口 |
| `harness/tools/background_submit.py` | 后台命令或后台分析的提交入口 |
| `harness/tools/background_check.py` | 后台任务状态与结果查询 |
| `harness/tools/run_verify.py` | 工作区内 Node／Python 验证脚本运行入口 |
| `harness/tools/swarm.py` | 多角色团队协作 Schema 与默认团队 |
| `harness/tools/notes_read.py` | 读取 HARNESS.md 项目长期笔记 |
| `harness/tools/notes_append.py` | 向 HARNESS.md 追加长期知识 |
| `harness/tools/notes_replace.py` | 替换 HARNESS.md 全部内容 |
| `harness/tools/web_fetch.py` | 读取公开网页并提取文本，拒绝私网、重定向和危险内容 |
| `harness/tools/web_search.py` | 使用 Tavily 搜索公开网页并返回标题、URL 和摘要 |
| `harness/usage.py` | 逐请求 token 记录、模型费率与费用汇总 |
| `harness/cli.py` | 终端输入、回答和斜杠命令 |
| `tests/` | 模拟接口、工具、计费与 CLI 测试 |

## 验证

```sh
python3 -m unittest discover -s tests -v
```

测试覆盖纯文字回答、多工具与多轮调用、委托与子查询、后台任务排队与跨轮查询、Swarm 多角色交接与回退、受限验证脚本、临时文件读写及错误恢复、关键词与文件类型搜索、行号及完整条目截断、写入和命令预览与逐次确认、拒绝和取消、命令两路输出与退出码、超时及同组进程清理、路径与软链接限制、连续对话、压缩与回滚、工具结果限长、递增重试与取消、限速显示、请求上限、token 和峰谷费用、`/cost`、`/compact`、启动及 `.env` 加载行为。文件和命令测试使用临时目录及无害本地子进程，接口与配置测试使用模拟内容，不读取实际密钥、不产生真实 API 费用。真实 API 联调尚未验证。

V0.4 权限离线测试通过，覆盖规则冲突与路径边界、五类 Bash 风险信号、环境清理、会话授权复用与隔离、禁止优先、拒绝说明，以及审计脱敏、特殊文件拒绝和日志故障时阻止执行。

V0.5 Agent 编排离线测试通过，覆盖独立预算、后台生命周期、Swarm 交接和受限验证目录授权。

本地记忆测试通过，覆盖 JSON 持久化、最近摘要加载、删除与清空确认、退出摘要、工具结果隔离和嵌入调用默认关闭。

项目笔记测试通过，覆盖 Markdown 持久化、启动注入、自动追加、整篇替换确认、路径保护和 CLI 管理。

终端渲染测试通过，覆盖 Markdown 标题、代码块、列表、链接、粗斜体、工具参数与结果面板，以及交互终端和管道输出的不同行为。

官方依据：[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) 、[Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/) 、[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) 、[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/) 、[Error Codes](https://api-docs.deepseek.com/quick_start/error_codes/) 。
