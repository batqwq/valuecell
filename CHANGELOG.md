# 变更日志

本文件记录本项目自创建以来的所有重要变更。

格式遵循 Keep a Changelog，并遵守语义化版本（SemVer）。

## [Unreleased]

### 添加
- 前端与后端的中文化收尾：继续清理零散英文硬编码，保证全中文体验。
- Telegram 指令改进：`/agent` 无参数时提供可点击的代理列表菜单。
- Telegram 长文分段与流式编辑：对超长回复按块编辑，提升可读性和“正在输入”连贯性。
- RemoteConnections 自动拉起本地 Agent：当 agent card 指向 `localhost` 且未就绪时，自动以子进程启动对应模块（日志输出到 `python/logs/agents/*.autospawn.log`），随后重试连接。
- 脱敏与发布支持：
  - 新增 `.githooks/pre-commit` 敏感信息扫描（安装：`bash scripts/setup_git_hooks.sh`）。
  - 新增 `scripts/clean_for_release.sh` 一键清理日志/数据库/知识库等运行产物。
  - 新增文档 `docs/DEPLOY_TO_GITHUB.md`，指导推送到 `github.com/batqwq/valuecell` 的脱敏流程与历史清洗。

### 变更
- 模型供应商与模型名展示逻辑文档化，确保通过 OpenRouter 使用 DeepSeek/Qwen 时，Telegram 与 Web 聊天都能正确标注“供应商/模型/开始时间/生成耗时”。
- Telegram 连接失败提示改为中文并说明“已尝试自动启动”，同时给出手动启动命令。
- 当模型 ID 缺失且已配置 OpenRouter 的对话密钥时，默认回退 `DEFAULT_CHAT_MODEL_ID`（默认 `deepseek/deepseek-v3.2-exp`），避免显示 `unknown`。
- Telegram 聊天输出全面中文化：过滤任务状态 JSON、补充中文态消息、交易实例描述/错误提示本地化，并将 HTTP 503 错误转换为中文说明。
- `start.sh` 在启动 Telegram 轮询与 Grok watcher 前会等待后端健康检查，避免初始阶段的连接拒绝与超时日志刷屏。
- Telegram 轮询脚本将 webhook POST 超时时间提升至 120 秒，并针对 ReadTimeout 做优雅重试，避免误报错误日志。
- start.sh 在所有模块成功拉起后输出汇总提示（PID/访问地址），便于确认启动状态。
- `.gitignore` 扩充：忽略 `node_modules/`、`frontend/node_modules/`、`frontend/.cache/` 与 `python/logs/`。
- Telegram `/help` 与 `/agent` 命令增加中文智能体简介，同时提供内联按钮以快速切换目标智能体，并新增“🔄 重启所有服务”菜单项给出操作指引。

### 修复
- LanceDB 向量库在大文本插入时偶发的 Arrow FixedSizeListType 转换报错，计划通过更稳健的 schema 与分块策略规避。
- 修复 AutoTradingAgent 因循环依赖导致无法启动的问题：`exchanges/base_exchange.py` 不再直接导入 `TradeType`，改为 `TYPE_CHECKING` 与字符串前向引用，解除与 `models.py` 的循环依赖；随后自动拉起与连接重试可正常工作。
- 当 LanceDB 表中的向量维度与当前嵌入模型不一致时，自动检测并覆盖重建表，避免 FixedSizeListType 转换错误（research_agent/knowledge.py）。
- 修复 AutoTradingAgent 解析结果为字符串导致 `'str' object has no attribute 'agent_models'`：
  - `_parse_trading_request` 增强为从 LLM 文本中稳健提取/解析 JSON，兼容 `agent_model` 与 `agent_models` 两种字段，并规范 `crypto_symbols` 的类型（agent.py）。
- AutoTradingAgent 对非交易意图的自然语言输入改为友好提示，不再报解析错误；解析失败时返回全中文说明，避免在 Telegram 中出现英文报错信息（agent.py）。
- 修复 searchXagent 嵌入器配置报错：`vdb.py` 优先使用 `.env` 中的 `EMBEDDER_*` 显式配置（OpenAI Embedder），缺失时再回退到集中化 provider 工厂，避免误用 OpenRouter 作为嵌入提供商。

### 日志
- Telegram 服务新增滚动日志到 `python/logs/telegram/telegram_service.log`，可通过 `TELEGRAM_LOG_FILE` 或 `VALUECELL_LOG_DIR` 覆盖路径。
- 自动拉起的内置 Agent 写入 `python/logs/agents/<AgentName>.autospawn.log`。

### 安全
- 补充基于 Nginx/Caddy 的反向代理与 TLS/HTTPS 部署指引，保持后端仅监听 127.0.0.1。

---

## [0.3.0] - 2025-10-30

本次为“OKX 交易所适配与一键启动整合”的里程碑版本，聚焦交易接入、Telegram 交互、Grok 市场巡检、中文体验与统一启动脚本。

### 添加
- 交易所与执行
  - 新增 OKX 交易所适配（基于 ccxt）：余额、行情、下单、订单/持仓、精度与限额校验、品种归一化（现货/永续）、`tdMode`/`reduceOnly`/`posSide` 支持。
  - 交易执行器改为异步：先纸交易仿真，再按会话状态镜像到实盘（具备精度/限额保护）。
  - 会话级“纸交易/真仓”热切换，支持中文同义词，二次确认按钮。

- Telegram 机器人与接口
  - 新增 `/api/v1/telegram/webhook` 路由与长轮询脚本，二者复用同一处理逻辑。
  - 全中文 UX：`/start`、`/help`、`/menu`、`/agent <name>`、`/paper`、`/live` 等指令与按钮。
  - 白名单与二次确认：仅允许指定用户操作关键切换（例：`@bbb9_t`，ID: `5067260604`）。
  - “正在输入”指示：在生成回复期间向用户显示 typing 状态。
  - 回复头自动标注“模型供应商/模型/开始时间”，并在结尾附“生成耗时”。

- Grok 10 分钟市场巡检（searchXagent）
  - 新增 xAI Grok-4-Fast 异步客户端与周期性任务（默认每 10 分钟）。
  - 将可信、不重复的市场资讯写入 `.knowledge/market_updates/*.md`，并尝试入库至 LanceDB 以供检索。
  - 提供工具函数 `searchXagent(custom_query, immediate)`，允许 ResearchAgent、Planner、SEC 分析在任意时刻触发即时或周期性检索。

- Web/SSE 聊天
  - SSE 流水线统一中文标签；会话开头标注“供应商/模型/开始时间”，结尾追加“生成耗时”。

- 启动与脚本
  - 新增 `start.sh` 一键启动脚本：
    - 自动加载 `.env`。
    - 启动后端服务与智能体（通过 `scripts/launch.py`）。
    - 启动 Telegram 长轮询与 Grok 巡检 watcher。
    - 支持 `--no-frontend/--no-backend/--no-telegram/--no-search` 等开关与 PID 清理。

- 配置与限制
  - 嵌入向量（Embedding）配置：默认 `text-embedding-3-large` 与维度设置；修正提示词需使用引号避免 `uv` 解析问题。
  - 强化 OpenRouter 使用路径：除 Embedding 外禁止直接调用 OpenAI LLM，统一通过 OpenRouter 调用 DeepSeek/Qwen。
  - 文档与运行指引：端口转发建议（`1420` 前端、`8000` API/SSE）、仅本地监听策略，确保网页前端与后台不暴露公网。

### 变更
- i18n
  - 默认语言改为简体中文（`zh-Hans`），增加 `VITE_USER_LANGUAGE` 环境变量接入前端。
  - 前端主要导航、Agent 卡片标题、占位符、流式指示文案等改为中文。

- 模型与调用
  - 统一在 Telegram 与 Web 聊天中展示模型供应商/模型名与耗时；提供更稳健的供应商推断（DeepSeek/Qwen → OpenRouter）。
  - 规范 `.env` 与 `python/third_party/TradingAgents/.env` 的示例与中文说明，便于部署与排错。

- 启动与运维
  - 将多个独立启动流程整合为 `start.sh`，提升开箱即用与可观测性。

### 修复
- 解决 `.env` 中多行提示词导致 `uv run --env-file` 解析失败的问题（建议对复杂提示词加引号）。
- 修正 Telegram 长轮询在服务未启动时的连接拒绝问题：增加合理的启动顺序与重试提示。
- 修正日志格式中 `logger.info("Stored market update at %s")` 未插值的问题（采用结构化/新式格式化）。

### 移除
- 移除（或禁用）除 Embedding 外的直接 OpenAI LLM 调用路径，改为统一使用 OpenRouter 访问 DeepSeek/Qwen。

### 安全
- 后端仅绑定 `127.0.0.1`；结合 SSH 隧道（如 `ssh -L 1420:localhost:1420`）进行远程访问，避免公网暴露。
- Telegram 关键操作启用白名单与二次确认，降低误操作风险。

### 已知问题
- LanceDB 在特定大文本/嵌入尺寸下仍可能出现 Arrow 类型转换告警；已纳入后续优化计划。

---

## [0.2.0] - 2025-10-29

聚焦中文体验与基础设施完善。

### 添加
- 前端中文化基础文案与菜单项：主页/市场/设置/自选/搜索/添加等。
- SSE 流式“AI 正在回复…”指示文案。
- TradingAgents 中文示例环境文件与说明，便于快速配置。

### 变更
- 默认时区与语言在后端 `i18n` 配置中初始化并打印，便于审计（如 `language=zh-Hans, timezone=Asia/Shanghai`）。

### 修复
- 增强 CORS/Origin 兼容性（`127.0.0.1/localhost`）。

---

## [0.1.0] - 2025-10-28

项目初始版本。

### 添加
- 后端服务框架与 API 基础路由（健康检查等）。
- 资产数据适配器接入与注册：Yahoo Finance、AKShare。
- 基本 Agent 流水线与日志设施。

---

备注
- 本日志力求“全面、合并重复、可审阅”。如需按提交记录进一步细化到每一次改动，请在仓库具备可用的提交历史后启用自动汇总脚本进行补充。
