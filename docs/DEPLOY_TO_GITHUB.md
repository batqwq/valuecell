# 部署到你自己的 GitHub 仓库（脱敏发布）

目标仓库：`https://github.com/batqwq/valuecell`

## 1. 本地脱敏与清理

1) 确认 .gitignore 已忽略敏感与大文件：`.env`、`logs/`、`python/logs/`、`lancedb/`、`.knowledge/`、`*.db`、`node_modules/`。

2) 启用提交前敏感扫描钩子（阻止 API Key 被提交）：

```bash
bash scripts/setup_git_hooks.sh
```

3) 清理运行产物（日志/数据库/知识库等）：

```bash
bash scripts/clean_for_release.sh
```

4) 确认 `.env` 未被提交；如需示例配置，请更新并提交 `.env.example`（仅占位符）。

## 2. 初始化并推送到你的仓库

建议在项目根目录执行：

```bash
git init
git add .
git commit -m "chore: initial import (sanitized)"
git branch -M main
git remote add origin git@github.com:batqwq/valuecell.git
git push -u origin main
```

如使用 HTTPS：

```bash
git remote add origin https://github.com/batqwq/valuecell.git
git push -u origin main
```

## 3. 若仓库已有历史且含敏感信息

使用 `git filter-repo` 或 BFG 清洗历史：

```bash
# 安装 git-filter-repo 后
git filter-repo --path .env --invert-paths || true
git filter-repo --replace-text <(cat <<'EOF'
OPENAI_API_KEY==>OPENAI_API_KEY_PLACEHOLDER
OPENROUTER_API_KEY==>OPENROUTER_API_KEY_PLACEHOLDER
TELEGRAM_BOT_TOKEN==>TELEGRAM_BOT_TOKEN_PLACEHOLDER
OKX_API_KEY==>OKX_API_KEY_PLACEHOLDER
OKX_SECRET==>OKX_SECRET_PLACEHOLDER
OKX_PASSPHRASE==>OKX_PASSPHRASE_PLACEHOLDER
EOF
)
git push --force origin main
```

请随后在各平台（OpenRouter / Telegram / OKX / xAI）主动轮换密钥。

## 4. GitHub 仓库设置建议

- “Settings → Secrets and variables → Actions” 中配置运行所需的 Token（如需 CI）。
- 开启分支保护规则（main 需 PR 才能合并）。
- 打开 Dependabot 安全更新与漏洞告警。

## 5. 常见问题

- 提交被 pre-commit 拦截：请移除或用占位符替换明文密钥，然后再提交；真实密钥放到 `.env`。
- 大体积文件未忽略：更新 `.gitignore` 后，若已提交过历史，需用 `git filter-repo` 清除历史记录。

