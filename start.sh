#!/bin/bash
set -Eeuo pipefail

# Simple project launcher with auto-install for bun and uv
# - macOS: use Homebrew to install missing tools
# - other OS: print guidance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
PY_DIR="$SCRIPT_DIR/python"
ENV_FILE="$SCRIPT_DIR/.env"

BACKEND_PID=""
FRONTEND_PID=""
TELEGRAM_PID=""
SEARCH_XAGENT_PID=""

info()  { echo "[INFO]  $*"; }
success(){ echo "[ OK ]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERR ]  $*" 1>&2; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  warn ".env file not found at $ENV_FILE"
fi

ensure_brew_on_macos() {
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    if ! command_exists brew; then
      error "Homebrew is not installed. Please install Homebrew: https://brew.sh/"
      error "Example install: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
      exit 1
    fi
  fi
}

ensure_tool() {
  local tool_name="$1"; shift
  local brew_formula="$1"; shift || true

  if command_exists "$tool_name"; then
    success "$tool_name is installed ($($tool_name --version 2>/dev/null | head -n1 || echo version unknown))"
    return 0
  fi

  case "$(uname -s)" in
    Darwin)
      ensure_brew_on_macos
      info "Installing $tool_name via Homebrew..."
      brew install "$brew_formula"
      ;;
    Linux)
      info "Detected Linux, auto-installing $tool_name..."
      if [[ "$tool_name" == "bun" ]]; then
        curl -fsSL https://bun.sh/install | bash
        # Add Bun default install dir to PATH (current process only)
        if ! command_exists bun && [[ -x "$HOME/.bun/bin/bun" ]]; then
          export PATH="$HOME/.bun/bin:$PATH"
        fi
      elif [[ "$tool_name" == "uv" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv default install dir to PATH (current process only)
        if ! command_exists uv && [[ -x "$HOME/.local/bin/uv" ]]; then
          export PATH="$HOME/.local/bin:$PATH"
        fi
      else
        warn "Unknown tool: $tool_name"
      fi
      ;;
    *)
      warn "$tool_name not installed. Auto-install is not provided on this OS. Please install manually and retry."
      exit 1
      ;;
  esac

  if command_exists "$tool_name"; then
    success "$tool_name installed successfully"
  else
    error "$tool_name installation failed. Please install manually and retry."
    exit 1
  fi
}

wait_for_backend() {
  local url="http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}/api/v1/system/health"
  local attempts=0
  local max_attempts=30
  info "等待后端健康检查: $url"
  while ! curl -fsS "$url" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if (( attempts >= max_attempts )); then
      warn "后端未在预期时间内完成启动，后续服务可能会自行重试。"
      return 1
    fi
    sleep 1
  done
  success "后端健康检查通过"
  return 0
}

compile() {
  # Backend deps
  if [[ -d "$PY_DIR" ]]; then
    info "Sync Python dependencies (uv sync)..."
    (cd "$PY_DIR" && bash scripts/prepare_envs.sh && uv run --env-file "$ENV_FILE" valuecell/server/db/init_db.py)
    success "Python dependencies synced"
  else
    warn "Backend directory not found: $PY_DIR. Skipping"
  fi

  # Frontend deps
  if [[ -d "$FRONTEND_DIR" ]]; then
    info "Install frontend dependencies (bun install)..."
    (cd "$FRONTEND_DIR" && bun install)
    success "Frontend dependencies installed"
  else
    warn "Frontend directory not found: $FRONTEND_DIR. Skipping"
  fi
}

start_backend() {
  if [[ ! -d "$PY_DIR" ]]; then
    warn "Backend directory not found; skipping backend start"
    return 0
  fi
  info "Starting backend与核心智能体 (scripts/launch.py)..."
  (
    cd "$PY_DIR" && uv run --env-file "$ENV_FILE" scripts/launch.py
  ) & BACKEND_PID=$!
  info "Launch manager PID: $BACKEND_PID"
}

start_frontend() {
  if [[ ! -d "$FRONTEND_DIR" ]]; then
    warn "Frontend directory not found; skipping frontend start"
    return 0
  fi
  info "Starting frontend dev server (bun run dev)..."
  (
    cd "$FRONTEND_DIR" && bun run dev
  ) & FRONTEND_PID=$!
  info "Frontend PID: $FRONTEND_PID"
}

start_telegram_bot() {
  if [[ ! -d "$PY_DIR" ]]; then
    warn "Backend directory not found; skipping Telegram bot"
    return 0
  fi
  if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    warn "TELEGRAM_BOT_TOKEN not set; skipping Telegram bot"
    return 0
  fi
  info "Starting Telegram long-polling bot..."
  (
    cd "$PY_DIR" && uv run --env-file "$ENV_FILE" scripts/telegram_polling.py
  ) & TELEGRAM_PID=$!
  info "Telegram bot PID: $TELEGRAM_PID"
}

start_search_xagent() {
  if [[ ! -d "$PY_DIR" ]]; then
    warn "Backend directory not found; skipping searchXagent"
    return 0
  fi
  if [[ -z "${XAI_API_KEY:-}" ]]; then
    warn "XAI_API_KEY not set; skipping searchXagent"
    return 0
  fi
  info "Starting Grok 10-min watcher (searchXagent)..."
  (
    cd "$PY_DIR" && uv run --env-file "$ENV_FILE" -m valuecell.agents.research_agent.search_x_agent
  ) & SEARCH_XAGENT_PID=$!
  info "searchXagent PID: $SEARCH_XAGENT_PID"
}
cleanup() {
  echo
  info "Stopping services..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$TELEGRAM_PID" ]] && kill -0 "$TELEGRAM_PID" 2>/dev/null; then
    kill "$TELEGRAM_PID" 2>/dev/null || true
  fi
  if [[ -n "$SEARCH_XAGENT_PID" ]] && kill -0 "$SEARCH_XAGENT_PID" 2>/dev/null; then
    kill "$SEARCH_XAGENT_PID" 2>/dev/null || true
  fi
  success "Stopped"
}

trap cleanup EXIT INT TERM

print_usage() {
  cat <<'EOF'
Usage: ./start.sh [options]

Description:
  - Checks whether bun and uv are installed; on macOS, missing tools will be auto-installed via Homebrew.
  - Then installs backend and frontend dependencies and starts services.

Options:
  --no-frontend   Start backend only
  --no-backend    Start frontend only
  --no-telegram   Do not start Telegram long-polling bot
  --no-search     Do not start Grok 10-min watcher (searchXagent)
  -h, --help      Show help
EOF
}

main() {
  local start_frontend_flag=1
  local start_backend_flag=1
  local start_telegram_flag=1
  local start_search_flag=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-frontend) start_frontend_flag=0; shift ;;
      --no-backend)  start_backend_flag=0; shift ;;
      --no-telegram) start_telegram_flag=0; shift ;;
      --no-search)   start_search_flag=0; shift ;;
      -h|--help)     print_usage; exit 0 ;;
      *) error "Unknown argument: $1"; print_usage; exit 1 ;;
    esac
  done

  # Ensure tools
  ensure_tool bun oven-sh/bun/bun
  ensure_tool uv uv

  compile

  if (( start_frontend_flag )); then
    start_frontend
    sleep 5  # Give frontend a moment to start
  fi

  if (( start_backend_flag )); then
    start_backend
    sleep 2
    wait_for_backend || true
  fi

  if (( start_telegram_flag )); then
    start_telegram_bot
  fi
  if (( start_search_flag )); then
    start_search_xagent
  fi

  success "所有模块已启动：frontend=${FRONTEND_PID:--} backend_manager=${BACKEND_PID:--} telegram=${TELEGRAM_PID:--} searchXagent=${SEARCH_XAGENT_PID:--}"
  info "访问前端: http://127.0.0.1:1420  | API: http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}"

  # Wait for background jobs
  wait
}

main "$@"
