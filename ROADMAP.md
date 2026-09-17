# Harness 进度

## 当前阶段

V0.2 query engine 已完成本地发布准备，目标仓库为 `MichaelJiangg/harness`，拟使用标签 `v0.2`；GitHub 上传与 Release 尚未执行。Python 查询引擎已支持限速流式显示、自动与手动上下文压缩、工具结果截断及有限重试，已通过离线验证。主入口为 `python3 -m harness`，核心函数为 `harness/engine.py` 的 `query_loop(state)`。

## 已完成

- 检查工作空间：本项目目录为空；本机 Node.js v22.23.2，原生 fetch、readline 和 node:test 可用。
- 用户确认 Node.js 零第三方依赖方案及先补规范再实现的顺序。
- 用户随后提供 Python 核心骨架，已先调整 CLAUDE.md，再将实现切换为 Python 标准库版本。
- 已核验官方 DeepSeek 模型、Chat Completions 工具调用消息格式、usage 字段与峰谷费率。
- 实现模型回复与工具调用循环，保存 assistant 工具请求并按 tool_call_id 回传全部工具结果。
- 实现 read_file 和 run_command 占位，明确返回未执行；参数错误作为工具结果回传。
- 实现会话历史、错误反馈、请求超时和每次查询最多 20 次模型请求。
- 实现逐请求 token 记录、缓存拆分、峰谷 USD 预估费用及用量未知标记。
- 实现 `/cost`、`/help`、`/exit`，等待模型期间可查询已返回请求的用量。
- 完成 README 使用说明、协议差异、工具扩展点、费用口径与取消限制。
- 42 项 Python 测试全部通过，覆盖 HTTP 协议、模型和工具循环、异常恢复、用量统计与 CLI。
- 用户确认后已删除本次创建的 JavaScript 草稿 `src/`、`test/`、`package.json`，仅保留 Python 实现；清理后测试与帮助入口再次验证通过。
- 新增 `.env` 密钥读取，环境变量优先；支持单行值、引号、注释和可选 export 前缀，不执行内容、不修改进程环境。
- 经用户单独确认创建仅含 `DEEPSEEK_API_KEY=` 的 `.env` 空模板，文件权限为 0600；`git check-ignore .env` 确认被排除。
- 增加 8 项配置测试，全部 50 项测试通过；配置测试使用模拟文件内容，不读取实际密钥。
- 回答改用 DeepSeek SSE 流式接口，文字片段收到后立即刷新，结束时不重复打印；原有工具循环、配置与费用口径保持不变。
- 流式工具参数按 index 拼接完整后执行；末块 usage 每次请求记录一次，断流保留已知用量或标为未知。
- 增加流式客户端和命令行集成验证，全部 61 项测试通过；确认尾块到达前已显示中文首段，流中 `/cost` 可用，失败对话不进入后续历史。
- 终端默认每字符等待 20 毫秒，等待可取消且不持输出锁；管道输出不限速，输入提示不会插入逐字回答中。
- 超过 24000 字符的上下文自动摘要，保留系统提示、最近 4 个完整用户轮次及当前轮；摘要最多 2000 字符，失败或无效时不替换原历史。
- 每次提问最多压缩 2 次，计数跨工具循环累计；保留部分仍过长或压缩无法达标时提示「太长了，建议开个新会话」。
- 新增空闲时 `/compact` 手动压缩；压缩期间 `/cost` 和退出仍可用，失败保留原会话。
- 工具结果最终序列化长度最多 6000 字符，首尾截断封装保留原始长度与标记；终端不打印原始超长结果。
- 暂时性 API 错误最多重试 3 次，等待 1、2、4 秒，部分输出会标注未完成后另起回答；永久错误不重试。
- 摘要、重试与普通请求统一逐次记账，并共用单次操作 20 次实际请求上限；用量未知时保持未知，不重跑已执行工具。
- 全部 101 项测试通过，覆盖新增上下文、截断、重试、压缩回滚与终端限速交互。
- 整理 V0.2 query engine 版本说明，README 明确首次使用需自行创建 `.env`；本项目已初始化独立 `main` 仓库，不携带父级 `sushu` 历史。
- 发布清单包含 24 个源码、测试和文档文件；`.env` 与 Python 缓存已排除，常见凭据模式扫描未发现匹配。

## 进行中

- 用户已确认公开发布：将 `main` 上传至 GitHub 的 `MichaelJiangg/harness`，创建 `v0.2` 标签及 `V0.2 query engine` Release；等待 CLI 登录完成。

## 待办

- 配置 API 密钥后进行一次真实接口联调。
- 后续独立实现真实工具执行。

## 阻塞与待确认

- GitHub 连接器可识别账号 `MichaelJiangg`，但查询 `MichaelJiangg/harness` 返回 404，未找到可访问的同名仓库；用户已确认新建公开仓库。
- 通过本机已启用的代理恢复了 CLI 的 GitHub 连接，已发起设备登录，等待用户完成 `gh` 授权。连接器不提供创建仓库或发布 Release 的接口。
- 已配置本项目 `origin` 为 `https://github.com/MichaelJiangg/harness.git`；仓库创建、推送、标签和 Release 尚未执行。
- 真实 API 联调待用户在本地 `.env` 或环境变量中配置密钥后验证，不影响离线测试。
- `/exit` 停止后续循环，不保证中断在途 HTTP 或远端生成；退出时可能无法获取该请求最终用量。

## 最近验证

- 2026-09-18：V0.2 发布准备阶段重新执行 `python3 -m unittest discover -s tests -q`，101 项全部通过；`python3 -m harness --help` 正常退出。独立仓库的 `git check-ignore .env` 返回 `.env`，发布文件清单扫描通过，未读取实际密钥或调用真实模型 API。
- 2026-09-18：执行 `python3 -m unittest discover -s tests -q`，101 项全部通过。验证最近轮次与工具链完整保留、压缩次数跨轮累计、摘要与重试共同计费、长结果严格限长、限速等待期间命令响应，以及 `/compact` 提交和失败回滚；未访问真实 API 或读取实际密钥。
- 2026-09-18：流式显示改造后执行 `python3 -m unittest discover -s tests -q`，61 项全部通过。使用模拟 SSE 流验证即时刷新、多工具分片、断流与用量记录；未调用真实 API。
- 2026-09-18：接入 `.env` 后执行 `python3 -m unittest discover -s tests -q`，50 项全部通过；`git check-ignore .env` 返回 `.env`。
- 2026-09-18：Python 3.14.6 执行 `python3 -m unittest discover -s tests -v`，42 项测试全部通过。
- 清理 JavaScript 草稿后执行 `python3 -m unittest discover -s tests -q`，42 项通过；`python3 -m harness --help` 正常退出。
- 启动测试确认 `python3 -m harness --help` 无密钥可用；缺失密钥时正常启动返回退出码 1，提示配置环境变量。
- 已模拟 HTTP 响应提前断开，确认请求用量标为未知，不漏记、不输出响应正文。
- 未调用真实 DeepSeek API，未产生测试 API 费用。
