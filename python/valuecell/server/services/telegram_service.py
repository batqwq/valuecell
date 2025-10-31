"""Telegram integration service.

Provides helpers to process Telegram updates, route text to the
AgentOrchestrator, and send replies using the Bot API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import requests

from valuecell.core.coordinate.orchestrator import AgentOrchestrator
from valuecell.core.types import (
    BaseResponse,
    ConversationItemEvent,
    NotifyResponseEvent,
    StreamResponseEvent,
    TaskStatusEvent,
    UnifiedResponseData,
    UserInput,
    UserInputMetadata,
)

logger = logging.getLogger(__name__)


def _configure_logger() -> None:
    """Ensure Telegram service logs也落到本地文件，便于排查。"""

    log_file = os.getenv("TELEGRAM_LOG_FILE")
    if not log_file:
        base_dir = os.getenv("VALUECELL_LOG_DIR", os.path.join(os.getcwd(), "logs"))
        log_file = os.path.join(base_dir, "telegram", "telegram_service.log")

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler_exists = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        if not handler_exists:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)
    except Exception as exc:  # pragma: no cover - logging 自检
        logger.warning("无法初始化 Telegram 日志文件：%s", exc)


_configure_logger()


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


class TelegramService:
    """Service to handle Telegram updates and send messages."""

    # In-memory mapping from chat_id -> selected agent name
    _chat_agent: Dict[int, str] = {}
    _pending_confirm: Dict[int, str] = {}

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set. Telegram integration disabled.")

        allowed = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
        self.allowed_user_ids: Optional[List[int]] = (
            [int(x) for x in allowed.split(",") if x.strip().isdigit()] if allowed else None
        )

        self.default_agent = os.getenv("TELEGRAM_DEFAULT_AGENT", "ResearchAgent")
        self.webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

        self._orchestrator = AgentOrchestrator()

    # --------------------- Public API ---------------------

    def _resolve_model_info(self, agent_name: Optional[str]) -> tuple[str, str]:
        env = os.getenv
        if not agent_name:
            mid = env("RESEARCH_AGENT_MODEL_ID", "unknown")
        else:
            name = agent_name.lower()
            if "research" in name:
                mid = env("RESEARCH_AGENT_MODEL_ID", "unknown")
            elif "auto" in name and "trading" in name:
                mid = env("TRADING_PARSER_MODEL_ID", env("TRADING_PARSER_MODEL_ID", "unknown"))
            elif "sec" in name:
                mid = env("SEC_ANALYSIS_MODEL_ID", env("SEC_PARSER_MODEL_ID", "unknown"))
            else:
                mid = env("RESEARCH_AGENT_MODEL_ID", "unknown")

        # Fallback default for model id via OpenRouter if not set
        if mid == "unknown" and env("OPENROUTER_API_KEY"):
            mid = env("DEFAULT_CHAT_MODEL_ID", "deepseek/deepseek-v3.2-exp")

        provider = "未知供应商"
        openrouter_key = env("OPENROUTER_API_KEY")
        openai_key = env("OPENAI_API_KEY")
        if openrouter_key:
            provider = "OpenRouter"
        if mid.startswith("google/") or mid.startswith("gemini"):
            provider = "Google"
        elif mid.startswith("openai/") or openai_key:
            provider = "OpenAI"
        elif mid.startswith("anthropic/"):
            provider = "Anthropic"
        elif mid.startswith("deepseek/") and openrouter_key:
            provider = "OpenRouter"
        elif mid.startswith("deepseek/"):
            provider = "DeepSeek"
        return provider, mid

    def is_ready(self) -> bool:
        return bool(self.bot_token)

    def validate_secret(self, provided: Optional[str]) -> bool:
        if not self.webhook_secret:
            return True  # no secret configured
        return provided == self.webhook_secret

    async def handle_update(self, update: dict) -> None:
        """Process a single Telegram Update payload."""
        try:
            # Handle inline keyboard callbacks first
            callback = update.get("callback_query")
            if callback:
                await self._handle_callback(callback)
                return

            message = update.get("message") or update.get("edited_message")
            if not message:
                # Could support callbacks, channel posts, etc. Ignore for now
                return

            chat = message.get("chat", {})
            chat_id = chat.get("id")
            from_user = message.get("from", {})
            user_id = from_user.get("id")

            if chat_id is None or user_id is None:
                return

            if self.allowed_user_ids is not None and int(user_id) not in self.allowed_user_ids:
                await self._send_message(chat_id, "⛔ 你没有权限使用此机器人。")
                return

            text = (message.get("text") or "").strip()
            if not text:
                await self._send_message(chat_id, "⚠️ 暂不支持该类型消息，请发送文本。")
                return

            logger.info(
                "收到消息 chat_id=%s user_id=%s text=%s",
                chat_id,
                user_id,
                text,
            )

            # Confirmation flow: expecting '确认 xxx'
            if text.startswith("确认") and int(chat_id) in self._pending_confirm:
                desired = self._pending_confirm.get(int(chat_id), "").lower()
                confirm_text = text.replace("确认", "").strip().lower()
                if confirm_text in {desired, f"{desired}模式", f"切换{desired}"}:
                    # Dispatch switch command to AutoTradingAgent
                    del self._pending_confirm[int(chat_id)]
                    logger.info(
                        "收到切换确认 chat_id=%s mode=%s", chat_id, desired
                    )
                    await self._send_message(chat_id, f"🔐 正在切换至 {desired.upper()}，请稍等…")
                    await self._dispatch_switch_exchange(chat_id, user_id, desired)
                    return
                else:
                    await self._send_message(chat_id, "❌ 目标不匹配，已取消。")
                    del self._pending_confirm[int(chat_id)]
                    return

            # Basic commands
            if text.startswith("/start"):
                logger.info("执行 /start chat_id=%s", chat_id)
                await self._send_message(
                    chat_id,
                    "👋 欢迎使用 ValueCell 机器人。\n"
                    "你可以直接发送请求（例如：‘分析 BTC-USD’），或用 /agent <名称> 切换当前智能体。\n"
                    f"当前智能体：{self._get_agent(chat_id)}",
                )
                return
            if text.startswith("/help"):
                logger.info("执行 /help chat_id=%s", chat_id)
                help_text = (
                    "命令：\n"
                    "/start - 开始并查看欢迎信息\n"
                    "/help - 查看命令说明\n"
                    "/menu - 打开快捷菜单，快速执行切换操作\n"
                    "/agent <名称> - 切换当前智能体，例如 /agent ResearchAgent\n"
                    "/status - 查询自动交易状态（等同于菜单中的状态按钮）\n"
                    "发送任意其它文本将交给当前智能体处理。\n\n"
                    "🤖 当前可用智能体：\n"
                    f"{self._format_agent_summary()}"
                )
                await self._send_message(chat_id, help_text, reply_markup=self._menu_keyboard())
                return
            if text.startswith("/menu") or text in {"菜单", "menu", "/memu", "memu"}:
                logger.info("打开菜单 chat_id=%s", chat_id)
                await self._send_message(chat_id, "请选择动作：", reply_markup=self._menu_keyboard())
                return
            if text.startswith("/agent"):
                parts = text.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    agent_name = parts[1].strip()
                    self._chat_agent[int(chat_id)] = agent_name
                    logger.info(
                        "切换智能体 chat_id=%s target_agent=%s",
                        chat_id,
                        agent_name,
                    )
                    await self._send_message(
                        chat_id,
                        f"✅ 已切换当前智能体为：{self._get_agent(chat_id)}",
                        reply_markup=self._menu_keyboard(),
                    )
                else:
                    await self._send_agent_overview(chat_id)
                return

            if text in {"/status", "status", "状态", "摘要", "📊 状态"}:
                await self._send_status(chat_id, user_id)
                return

            # Trading mode commands (paper/live)
            lowered = text.lower()
            if lowered in {"/paper", "paper", "模拟", "模拟盘"}:
                self._pending_confirm[int(chat_id)] = "paper"
                logger.info("请求切换到 PAPER chat_id=%s", chat_id)
                await self._send_message(
                    chat_id,
                    "⚠️ 你正在请求切换到 模拟(PAPER) 模式。\n"
                    "为避免误操作，请回复：\n"
                    "确认 paper",
                )
                return
            if lowered in {"/live", "/okx", "okx", "真仓", "实盘", "live"}:
                self._pending_confirm[int(chat_id)] = "okx"
                logger.info("请求切换到 OKX chat_id=%s", chat_id)
                await self._send_message(
                    chat_id,
                    "⚠️ 你正在请求切换到 真实(OKX) 模式。\n"
                    "为避免误操作，请回复：\n"
                    "确认 okx",
                )
                return

            # Forward to orchestrator（开启“正在输入”指示）
            agent_name = self._get_agent(chat_id)
            conversation_id = f"tg_{chat_id}"
            meta = UserInputMetadata(user_id=str(user_id), conversation_id=str(conversation_id))
            user_input = UserInput(query=text, target_agent_name=agent_name, meta=meta)

            # 打字指示器：每 ~4s 刷一次，直到生成完成
            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(self._typing_indicator(chat_id, stop_typing))

            response_text = await self._collect_response_text(user_input)

            logger.info(
                "完成调用 target_agent=%s chat_id=%s text_len=%s",
                agent_name,
                chat_id,
                len(response_text or ""),
            )

            stop_typing.set()
            try:
                await typing_task
            except Exception:
                pass
            if not response_text:
                response_text = "(No response)"

            # Telegram messages limited to ~4096 chars, split if needed
            for chunk in self._chunk_text(response_text, 3500):
                await self._send_message(chat_id, chunk)

        except Exception as e:
            logger.exception("Failed to handle Telegram update: %s", e)

    # --------------------- Internals ---------------------

    async def _collect_response_text(self, user_input: UserInput) -> str:
        """Run orchestrator and collect a single text response."""
        provider, model_id = self._resolve_model_info(user_input.target_agent_name)
        start_ts = time.monotonic()
        start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines: List[str] = [
            f"模型供应商：{provider}",
            f"模型：{model_id}",
            f"开始时间：{start_human}",
            "",
        ]
        async for resp in self._orchestrator.process_user_input(user_input):
            s = self._response_to_text(resp)
            if s:
                lines.append(s)
        elapsed = time.monotonic() - start_ts
        lines.append("")
        lines.append(f"生成完成，用时：{elapsed:.2f} 秒")
        return "\n".join(_coalesce_lines(lines))

    async def _handle_callback(self, callback: dict) -> None:
        """Process Telegram inline keyboard callback_query."""
        try:
            data = (callback.get("data") or "").strip()
            cb_id = callback.get("id")
            message = callback.get("message") or {}
            chat_id = (message.get("chat") or {}).get("id")
            from_user = callback.get("from") or {}
            user_id = from_user.get("id")
            if chat_id is None or user_id is None:
                if cb_id:
                    await self._answer_callback(cb_id, "Invalid callback")
                return

            if data == "menu":
                await self._send_message(chat_id, "请选择动作：", reply_markup=self._menu_keyboard())
                if cb_id:
                    await self._answer_callback(cb_id)
                return

            if data in {"switch_request:paper", "switch_request:okx"}:
                mode = data.split(":", 1)[1]
                self._pending_confirm[int(chat_id)] = mode
                await self._send_message(
                    chat_id,
                    f"⚠️ 你正在请求切换到 {'PAPER' if mode=='paper' else 'OKX'} 模式。\n为避免误操作，请确认：",
                    reply_markup=self._confirm_keyboard(mode),
                )
                if cb_id:
                    await self._answer_callback(cb_id)
                return

            if data in {"confirm_switch:paper", "confirm_switch:okx"}:
                mode = data.split(":", 1)[1]
                self._pending_confirm.pop(int(chat_id), None)
                await self._send_message(chat_id, f"🔐 正在切换至 {mode.upper()}，请稍等…")
                await self._dispatch_switch_exchange(chat_id, user_id, mode)
                if cb_id:
                    await self._answer_callback(cb_id, "已执行")
                return

            if data == "cancel_switch":
                self._pending_confirm.pop(int(chat_id), None)
                await self._send_message(chat_id, "已取消")
                if cb_id:
                    await self._answer_callback(cb_id, "已取消")
                return

            if data == "restart_services":
                restart_text = (
                    "🔄 要重启所有服务，请在服务器上执行：\n"
                    "`cd /opt/valuecell && ./start.sh`\n\n"
                    "只想重启后端/智能体，可执行：\n"
                    "`cd /opt/valuecell && ./start.sh --no-frontend`\n\n"
                    "执行后请等待 1-2 分钟，确保服务完全启动。"
                )
                await self._send_message(chat_id, restart_text, reply_markup=self._menu_keyboard())
                if cb_id:
                    await self._answer_callback(cb_id)
                return

            if data.startswith("choose_agent:"):
                agent = data.split(":", 1)[1].strip()
                if agent:
                    self._chat_agent[int(chat_id)] = agent
                    await self._send_message(
                        chat_id,
                        f"✅ 已切换当前智能体为：{agent}",
                        reply_markup=self._menu_keyboard(),
                    )
                if cb_id:
                    await self._answer_callback(cb_id)
                return

            if data == "noop":
                if cb_id:
                    await self._answer_callback(cb_id)
                return

            if cb_id:
                await self._answer_callback(cb_id)
        except Exception as e:
            logger.exception("Callback handling error: %s", e)

    async def _dispatch_switch_exchange(self, chat_id: int, user_id: int, desired: str) -> None:
        """Send a control command to AutoTradingAgent to switch exchange.

        This does not depend on the user's currently selected agent to ensure
        the switch is applied to active trading sessions.
        """
        conversation_id = f"tg_{chat_id}"
        meta = UserInputMetadata(user_id=str(user_id), conversation_id=str(conversation_id))
        user_input = UserInput(
            query=f"switch_exchange {desired}",
            target_agent_name="AutoTradingAgent",
            meta=meta,
        )
        # 打字指示器：每 ~4s 刷一次，直到执行完成
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self._typing_indicator(chat_id, stop_typing))

        logger.info("执行模式切换命令 chat_id=%s desired=%s", chat_id, desired)

        text = await self._collect_response_text(user_input)

        stop_typing.set()
        try:
            await typing_task
        except Exception:
            pass
        if not text:
            text = "(No response)"
        else:
            logger.info("模式切换结果 chat_id=%s length=%s", chat_id, len(text))
        for chunk in self._chunk_text(text, 3500):
            await self._send_message(chat_id, chunk)

    async def _send_agent_overview(self, chat_id: int) -> None:
        summary = self._format_agent_summary()
        overview_text = (
            "🤖 当前可用智能体：\n"
            f"{summary}\n\n"
            "提示：点击下方按钮可直接切换，也可以输入 /agent <名称> 手动切换。"
        )
        keyboard = self._agent_list_keyboard()
        await self._send_message(chat_id, overview_text, reply_markup=keyboard if keyboard else None)

    @staticmethod
    def _response_to_text(resp: BaseResponse) -> Optional[str]:
        try:
            event = getattr(resp, "event", None)
            data = getattr(resp, "data", None)

            # For状态事件，直接转中文提示，避免裸 JSON
            formatted_event = TelegramService._format_event_message(event, data)
            if formatted_event:
                return formatted_event

            payload = getattr(data, "payload", None)

            text_content: Optional[str] = None
            if payload is not None:
                if hasattr(payload, "content"):
                    text_content = getattr(payload, "content")
                elif isinstance(payload, dict):
                    text_content = payload.get("content") or payload.get("data")
                elif hasattr(payload, "data"):
                    candidate = getattr(payload, "data")
                    if isinstance(candidate, str):
                        text_content = candidate

            if not text_content:
                legacy = getattr(resp, "content", None)
                if isinstance(legacy, str):
                    text_content = legacy

            if text_content and isinstance(text_content, str):
                return TelegramService._localize_message(
                    text_content,
                    event,
                )

            # As fallback, show JSON（保留用于调试，但不阻止正常内容展示）
            data_dict = resp.model_dump(exclude_none=True)
            return json.dumps(data_dict, ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _localize_message(text: str, event: Optional[ConversationItemEvent]) -> str:
        normalized = text.strip()
        lowered = normalized.lower()

        if "failed to resolve agent card" in lowered:
            return (
                "⚠️ 无法连接到 AutoTradingAgent 服务（已尝试自动启动）。\n"
                "若仍失败，请手动启动并重试：\n"
                "`cd python && uv run --env-file ../.env -m valuecell.agents.auto_trading_agent`"
            )

        if normalized.startswith("(Error)"):
            # 去掉英文错误前缀，保持清晰中文提示
            cleaned = normalized.replace("(Error)", "⚠️ 错误", 1).strip()
            return cleaned

        replacements = {
            "Parsing trading request": "解析交易请求",
            "Creating": "正在创建",
            "Trading Instance Created": "交易实例已创建",
            "Configuration": "配置",
            "Trading Symbols": "交易品种",
            "Initial Capital": "初始资金",
            "Exchange": "交易所",
            "Check Interval": "检查频率",
            "Risk Per Trade": "单笔风险",
            "Max Positions": "最大持仓数",
            "AI Signals": "AI 信号",
            "Session ID": "会话 ID",
            "Total Active Instances in Session": "会话中的活跃实例数",
            "Starting continuous trading for all instances": "开始对所有实例进行持续交易",
            "All instances will run continuously until stopped.": "所有实例将持续运行，直到手动停止。",
            "Starting monitoring loop for all instances": "启动所有实例的监控循环",
            "Monitoring loop started": "监控循环已启动",
            "Model": "模型",
            "Instance ID": "实例 ID",
            "trading instance(s)": "个交易实例",
            "AI Signals: ✅ Enabled": "AI 信号：✅ 已启用",
            "AI Signals: ❌ Disabled": "AI 信号：❌ 已关闭",
            "DEFAULT_AGENT_MODEL": "默认模型（DEFAULT_AGENT_MODEL）",
            "Parse Error": "解析错误",
            "Could not parse trading configuration": "无法从你的消息中解析交易配置",
        }

        cleaned_text = normalized
        for eng, zh in replacements.items():
            cleaned_text = cleaned_text.replace(eng, zh)

        cleaned_text = cleaned_text.replace("- 交易品种:", "- 交易品种：")
        cleaned_text = cleaned_text.replace("- 初始资金:", "- 初始资金：")
        cleaned_text = cleaned_text.replace("- 交易所:", "- 交易所：")
        cleaned_text = cleaned_text.replace("- 检查频率:", "- 检查频率：")
        cleaned_text = cleaned_text.replace("- 单笔风险:", "- 单笔风险：")
        cleaned_text = cleaned_text.replace("- 最大持仓数:", "- 最大持仓数：")
        cleaned_text = cleaned_text.replace("- AI 信号:", "- AI 信号：")

        if "HTTP Error 503" in cleaned_text:
            cleaned_text = (
                "⚠️ AutoTradingAgent 返回 HTTP 503，网络通道被对端中断。"
                "请检查代理服务的运行状态或稍后重试。"
            )

        # 加粗表示统一改为中文冒号
        cleaned_text = cleaned_text.replace("**配置:**", "**配置：**")
        cleaned_text = cleaned_text.replace("**模型:**", "**模型：**")
        cleaned_text = cleaned_text.replace("**实例 ID:**", "**实例 ID：**")
        cleaned_text = cleaned_text.replace("**会话 ID:**", "**会话 ID：**")

        return cleaned_text

    @staticmethod
    def _format_event_message(event: Optional[ConversationItemEvent], data: Optional[UnifiedResponseData]) -> Optional[str]:
        if not isinstance(event, TaskStatusEvent):
            return None

        agent_name = getattr(data, "agent_name", None) or "目标智能体"
        task_id = getattr(data, "task_id", None)

        if event == TaskStatusEvent.TASK_STARTED:
            return f"🟡 {agent_name} 开始执行任务{(' ' + task_id) if task_id else ''}。"
        if event == TaskStatusEvent.TASK_COMPLETED:
            return f"✅ {agent_name} 已完成任务{(' ' + task_id) if task_id else ''}。"
        if event == TaskStatusEvent.TASK_FAILED:
            payload = getattr(data, "payload", None)
            reason = None
            if payload and hasattr(payload, "content"):
                reason = getattr(payload, "content")
            if reason:
                reason = TelegramService._localize_message(str(reason), event)
            else:
                reason = "发生未知错误"
            return f"⚠️ {agent_name} 执行失败：{reason}"
        return None

    def _get_agent(self, chat_id: int) -> str:
        return self._chat_agent.get(int(chat_id), self.default_agent)

    async def _send_message(self, chat_id: int, text: str, *, reply_markup: Optional[dict] = None) -> None:
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set, skip send_message")
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            # Non-blocking requests via thread executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(url, json=payload, timeout=15))
        except Exception as e:
            logger.error("sendMessage failed: %s", e)

    async def _answer_callback(self, callback_id: str, text: Optional[str] = None, show_alert: bool = False) -> None:
        if not self.bot_token:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
        payload = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(url, json=payload, timeout=10))
        except Exception as e:
            logger.error("answerCallbackQuery failed: %s", e)

    @staticmethod
    def _menu_keyboard() -> dict:
        return {
            "inline_keyboard": [
                [
                    {"text": "切换到模拟(PAPER)", "callback_data": "switch_request:paper"},
                ],
                [
                    {"text": "切换到真仓(OKX)", "callback_data": "switch_request:okx"},
                ],
                [
                    {"text": "🔄 重启所有服务", "callback_data": "restart_services"},
                ],
            ]
        }

    def _agent_list_keyboard(self) -> Optional[dict]:
        agents = self._list_available_agents()
        if not agents:
            return {
                "inline_keyboard": [
                    [{"text": "暂无可选智能体", "callback_data": "noop"}],
                ]
            }
        rows: List[List[dict]] = []
        current_row: List[dict] = []
        for agent in agents:
            current_row.append({"text": agent, "callback_data": f"choose_agent:{agent}"})
            if len(current_row) == 2:
                rows.append(current_row)
                current_row = []
        if current_row:
            rows.append(current_row)
        return {"inline_keyboard": rows}

    def _list_available_agents(self) -> List[str]:
        try:
            agents = self._orchestrator.agent_connections.list_available_agents()
        except Exception:
            agents = []
        fallback = [
            self.default_agent,
            "ResearchAgent",
            "AutoTradingAgent",
            "NewsAgent",
            "ValueCellAgent",
        ]
        seen: Dict[str, bool] = {}
        ordered: List[str] = []
        for name in agents + fallback:
            if not name:
                continue
            if name in seen:
                continue
            seen[name] = True
            ordered.append(name)
        return ordered

    def _format_agent_summary(self) -> str:
        agents = self._list_available_agents()
        if not agents:
            return "暂无可用智能体，稍后再试。"
        descriptions = self._agent_descriptions()
        lines = []
        for idx, agent in enumerate(agents, start=1):
            desc = descriptions.get(agent, "暂无简介，欢迎直接体验。")
            lines.append(f"{idx}. {agent} —— {desc}")
        return "\n".join(lines)

    @staticmethod
    def _agent_descriptions() -> Dict[str, str]:
        return {
            "ResearchAgent": "调研分析专家，整理 SEC 等监管文件并生成结构化洞察。",
            "AutoTradingAgent": "自动交易执行，支持模拟盘与 OKX 真仓切换，基于 AI 信号生成策略。",
            "NewsAgent": "实时跟踪加密及传统金融快讯，提供新闻摘要与影响分析。",
            "ValueCellAgent": "总控协调智能体，统筹多智能体任务流程。",
        }

    @staticmethod
    def _confirm_keyboard(mode: str) -> dict:
        label = "PAPER" if mode == "paper" else "OKX"
        return {
            "inline_keyboard": [
                [
                    {"text": f"✅ 确认切换 {label}", "callback_data": f"confirm_switch:{mode}"},
                    {"text": "取消", "callback_data": "cancel_switch"},
                ]
            ]
        }

    async def _send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        if not self.bot_token:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendChatAction"
        payload = {"chat_id": chat_id, "action": action}
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: requests.post(url, json=payload, timeout=10))
        except Exception as e:
            logger.debug("sendChatAction failed: %s", e)

    async def _typing_indicator(self, chat_id: int, stop_event: asyncio.Event) -> None:
        """周期性上报“正在输入”，直至 stop_event 触发。"""
        try:
            # sendChatAction 的效果大约持续 5 秒，4 秒刷新一次更稳妥
            while not stop_event.is_set():
                await self._send_chat_action(chat_id, "typing")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            logger.debug("typing indicator task error: %s", e)

    @staticmethod
    def _chunk_text(text: str, size: int) -> Iterable[str]:
        for i in range(0, len(text), size):
            yield text[i : i + size]


def _coalesce_lines(lines: List[str]) -> List[str]:
    """Merge small lines to reduce message count."""
    out: List[str] = []
    buf = []
    acc = 0
    for line in lines:
        if not line:
            continue
        if acc + len(line) > 3000 and buf:
            out.append("\n".join(buf))
            buf = []
            acc = 0
        buf.append(line)
        acc += len(line)
    if buf:
        out.append("\n".join(buf))
    return out
