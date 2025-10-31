"""Main auto trading agent implementation with multi-instance support"""

import asyncio
import json
import logging
import os
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional

from agno.agent import Agent

from valuecell.core.agent.responses import streaming
from valuecell.core.types import (
    BaseAgent,
    ComponentType,
    FilteredCardPushNotificationComponentData,
    FilteredLineChartComponentData,
    StreamResponse,
)

from .constants import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CHECK_INTERVAL,
    ENV_PARSER_MODEL_ID,
    ENV_SIGNAL_MODEL_ID,
)
from .exchanges import ExchangeBase, ExchangeType, OkxExchange, PaperTrading
from .formatters import MessageFormatter
from .models import (
    AutoTradingConfig,
    TradingRequest,
)
from .portfolio_decision_manager import (
    AssetAnalysis,
    PortfolioDecisionManager,
)
from .technical_analysis import AISignalGenerator, TechnicalAnalyzer
from .trading_executor import TradingExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum cached notifications per session
MAX_NOTIFICATION_CACHE_SIZE = 5000


class AutoTradingAgent(BaseAgent):
    """
    Automated crypto trading agent with technical analysis and position management.
    Supports multiple trading instances per session with independent configurations.
    """

    def __init__(self):
        super().__init__()

        # Multi-instance state management
        # Structure: {session_id: {instance_id: TradingInstanceData}}
        self.trading_instances: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Notification cache for batch sending
        # Structure: {session_id: deque[FilteredCardPushNotificationComponentData]}
        # Using deque with maxlen for automatic FIFO eviction
        self.notification_cache: Dict[
            str, Deque[FilteredCardPushNotificationComponentData]
        ] = {}

        try:
            # Parser agent for natural language query parsing
            # Uses centralized configuration system with automatic provider detection
            from valuecell.utils.model import get_model

            parser_model = get_model(
                env_key=ENV_PARSER_MODEL_ID,
            )

            self.parser_agent = Agent(
                model=parser_model,
                output_schema=TradingRequest,
                markdown=True,
            )
            logger.info("Auto Trading Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Auto Trading Agent: {e}")
            raise

    def _create_exchange_adapter(
        self, config: AutoTradingConfig
    ) -> Optional[ExchangeBase]:
        """
        Create exchange adapter based on agent configuration.

        Returns:
            Exchange adapter instance or None if configuration is invalid.
        """
        if config.exchange == ExchangeType.PAPER:
            return PaperTrading(initial_balance=config.initial_capital)

        if config.exchange == ExchangeType.OKX:
            api_key = os.getenv("OKX_API_KEY")
            api_secret = os.getenv("OKX_SECRET")
            passphrase = os.getenv("OKX_PASSPHRASE")

            if not api_key or not api_secret or not passphrase:
                logger.error(
                    "OKX exchange selected but API credentials are missing. "
                    "Please set OKX_API_KEY, OKX_SECRET, and OKX_PASSPHRASE."
                )
                return None

            sandbox = os.getenv("OKX_SANDBOX", "false").strip().lower() == "true"
            default_type = os.getenv("OKX_DEFAULT_TYPE", "spot").strip().lower()
            leverage_str = os.getenv("OKX_LEVERAGE", "").strip()
            margin_mode = os.getenv("OKX_MARGIN_MODE", "").strip().lower() or None

            leverage: Optional[int]
            if leverage_str:
                try:
                    leverage = int(leverage_str)
                except ValueError:
                    logger.warning(
                        "Invalid OKX_LEVERAGE value '%s'. Falling back to default.",
                        leverage_str,
                    )
                    leverage = None
            else:
                leverage = None

            try:
                adapter = OkxExchange(
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    sandbox=sandbox,
                    default_type=default_type,
                    leverage=leverage,
                    margin_mode=margin_mode,
                )
                return adapter
            except Exception as exc:
                logger.error("Failed to create OKX adapter: %s", exc)
                return None

        logger.warning("Unsupported exchange %s, falling back to None", config.exchange)
        return None

    async def _process_trading_instance(
        self,
        session_id: str,
        instance_id: str,
        semaphore: asyncio.Semaphore,
        unified_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Process a single trading instance with semaphore control for concurrency limiting.

        Args:
            session_id: Session identifier
            instance_id: Trading instance identifier
            semaphore: Asyncio semaphore to limit concurrent processing
            unified_timestamp: Optional unified timestamp for snapshot alignment across instances
        """
        async with semaphore:
            try:
                # Check if instance still exists and is active
                if instance_id not in self.trading_instances.get(session_id, {}):
                    return

                instance = self.trading_instances[session_id][instance_id]
                if not instance["active"]:
                    return

                # Get instance components
                executor = instance["executor"]
                config = instance["config"]
                ai_signal_generator = instance["ai_signal_generator"]

                # Update check info
                instance["check_count"] += 1
                instance["last_check"] = datetime.now()
                check_count = instance["check_count"]

                logger.info(
                    f"Trading check #{check_count} for instance {instance_id} (model: {config.agent_model})"
                )

                logger.info(
                    f"\n{'=' * 50}\n"
                    f"🔄 **Check #{check_count}** - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Instance: `{instance_id}`\n"
                    f"Model: `{config.agent_model}`\n"
                    f"{'=' * 50}\n\n"
                )

                # Phase 1: Collect analysis for all symbols
                logger.info("📊 **Phase 1: Analyzing all assets...**\n\n")

                # Initialize portfolio manager with LLM client for AI-powered decisions
                llm_client = None
                if ai_signal_generator and ai_signal_generator.llm_client:
                    llm_client = ai_signal_generator.llm_client

                portfolio_manager = PortfolioDecisionManager(config, llm_client)

                for symbol in config.crypto_symbols:
                    # Calculate indicators
                    indicators = TechnicalAnalyzer.calculate_indicators(symbol)

                    if indicators is None:
                        logger.warning(f"Skipping {symbol} - insufficient data")
                        continue

                    # Generate technical signal
                    technical_action, technical_trade_type = (
                        TechnicalAnalyzer.generate_signal(indicators)
                    )

                    # Generate AI signal if enabled
                    ai_action, ai_trade_type, ai_reasoning, ai_confidence = (
                        None,
                        None,
                        None,
                        None,
                    )

                    if ai_signal_generator:
                        ai_signal = await ai_signal_generator.get_signal(indicators)
                        if ai_signal:
                            (
                                ai_action,
                                ai_trade_type,
                                ai_reasoning,
                                ai_confidence,
                            ) = ai_signal
                            logger.info(
                                f"AI signal for {symbol}: {ai_action.value} {ai_trade_type.value} "
                                f"(confidence: {ai_confidence}%)"
                            )

                    # Create asset analysis
                    asset_analysis = AssetAnalysis(
                        symbol=symbol,
                        indicators=indicators,
                        technical_action=technical_action,
                        technical_trade_type=technical_trade_type,
                        ai_action=ai_action,
                        ai_trade_type=ai_trade_type,
                        ai_reasoning=ai_reasoning,
                        ai_confidence=ai_confidence,
                    )

                    # Add to portfolio manager
                    portfolio_manager.add_asset_analysis(asset_analysis)

                    # Display individual asset analysis
                    logger.info(
                        MessageFormatter.format_market_analysis_notification(
                            symbol,
                            indicators,
                            asset_analysis.recommended_action,
                            asset_analysis.recommended_trade_type,
                            executor.positions,
                            ai_reasoning,
                        )
                    )

                # Phase 2: Make portfolio-level decision
                logger.info(
                    "\n" + "=" * 50 + "\n"
                    "🎯 **Phase 2: Portfolio Decision Making...**\n" + "=" * 50 + "\n\n"
                )

                # Get portfolio summary
                portfolio_summary = portfolio_manager.get_portfolio_summary()
                logger.info(portfolio_summary + "\n")

                # Make coordinated decision (async call for AI analysis)
                portfolio_decision = await portfolio_manager.make_portfolio_decision(
                    current_positions=executor.positions,
                    available_cash=executor.get_current_capital(),
                    total_portfolio_value=executor.get_portfolio_value(),
                )

                # Display decision reasoning - cache it
                portfolio_decision_msg = FilteredCardPushNotificationComponentData(
                    title=f"{config.agent_model} Analysis",
                    data=f"💰 **Portfolio Decision Reasoning**\n{portfolio_decision.reasoning}\n",
                    filters=[config.agent_model],
                    table_title="Market Analysis",
                    create_time=datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
                # Cache the decision notification
                self._cache_notification(session_id, portfolio_decision_msg)

                # Phase 3: Execute approved trades
                if portfolio_decision.trades_to_execute:
                    logger.info(
                        "\n" + "=" * 50 + "\n"
                        f"⚡ **Phase 3: Executing {len(portfolio_decision.trades_to_execute)} trade(s)...**\n"
                        + "=" * 50
                        + "\n\n"
                    )

                    for (
                        symbol,
                        action,
                        trade_type,
                    ) in portfolio_decision.trades_to_execute:
                        # Get indicators for this symbol
                        asset_analysis = portfolio_manager.asset_analyses.get(symbol)
                        if not asset_analysis:
                            continue

                        # Execute trade
                        trade_details = await executor.execute_trade(
                            symbol, action, trade_type, asset_analysis.indicators
                        )

                        if trade_details:
                            # Cache trade notification
                            trade_message_text = (
                                MessageFormatter.format_trade_notification(
                                    trade_details, config.agent_model
                                )
                            )
                            trade_message = FilteredCardPushNotificationComponentData(
                                title=f"{config.agent_model} Trade",
                                data=f"💰 **Trade Executed:**\n{trade_message_text}\n",
                                filters=[config.agent_model],
                                table_title="Trade Detail",
                                create_time=datetime.now(timezone.utc).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            )
                            # Cache the trade notification
                            self._cache_notification(session_id, trade_message)
                        else:
                            trade_message = FilteredCardPushNotificationComponentData(
                                title=f"{config.agent_model} Trade",
                                data=f"💰 **Trade Failed:** Could not execute {action.value} "
                                f"{trade_type.value} on {symbol}\n",
                                filters=[config.agent_model],
                                table_title="Trade Detail",
                                create_time=datetime.now(timezone.utc).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            )
                            # Cache the failed trade notification
                            self._cache_notification(session_id, trade_message)

                # Take snapshots with unified timestamp if provided
                timestamp = unified_timestamp if unified_timestamp else datetime.now()
                executor.snapshot_positions(timestamp)
                executor.snapshot_portfolio(timestamp)

                # Send portfolio update
                portfolio_value = executor.get_portfolio_value()
                total_pnl = portfolio_value - config.initial_capital

                portfolio_msg = (
                    f"💰 **Portfolio Update**\n"
                    f"Model: {config.agent_model}\n"
                    f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Total Value: ${portfolio_value:,.2f}\n"
                    f"P&L: ${total_pnl:,.2f}\n"
                    f"Open Positions: {len(executor.positions)}\n"
                    f"Available Capital: ${executor.current_capital:,.2f}\n"
                )

                if executor.positions:
                    portfolio_msg += "\n**Open Positions:**\n"
                    for symbol, pos in executor.positions.items():
                        try:
                            import yfinance as yf

                            ticker = yf.Ticker(symbol)
                            current_price = ticker.history(period="1d", interval="1m")[
                                "Close"
                            ].iloc[-1]
                            if pos.trade_type.value == "long":
                                current_pnl = (current_price - pos.entry_price) * abs(
                                    pos.quantity
                                )
                            else:
                                current_pnl = (pos.entry_price - current_price) * abs(
                                    pos.quantity
                                )
                            pnl_emoji = "🟢" if current_pnl >= 0 else "🔴"
                            portfolio_msg += f"- {symbol}: {pos.trade_type.value.upper()} @ ${pos.entry_price:,.2f} {pnl_emoji} P&L: ${current_pnl:,.2f}\n"
                        except Exception as e:
                            logger.warning(f"Failed to calculate P&L for {symbol}: {e}")
                            portfolio_msg += f"- {symbol}: {pos.trade_type.value.upper()} @ ${pos.entry_price:,.2f}\n"

                logger.info(portfolio_msg + "\n")

                # Cache portfolio status notification
                component_data = self._get_instance_status_component_data(
                    session_id, instance_id
                )
                if component_data:
                    self._cache_notification(session_id, component_data)

            except Exception as e:
                logger.error(f"Error processing trading instance {instance_id}: {e}")
                # Don't raise - let other instances continue

    def _generate_instance_id(self, task_id: str, model_id: str) -> str:
        """
        Generate unique instance ID for a specific model

        Args:
            task_id: Task ID from the request
            model_id: Model identifier (e.g., 'deepseek/deepseek-v3.1-terminus')

        Returns:
            Unique instance ID combining timestamp, task, and model
        """
        import hashlib

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )  # Include microseconds for uniqueness
        # Create a short hash from model_id for readability
        model_hash = hashlib.md5(model_id.encode()).hexdigest()[:6]
        # Extract model name (last part after /)
        model_name = model_id.split("/")[-1].replace("-", "_").replace(".", "_")[:15]

        return f"trade_{timestamp}_{model_name}_{model_hash}"

    def _init_notification_cache(self, session_id: str) -> None:
        """Initialize notification cache for a session if not exists"""
        if session_id not in self.notification_cache:
            self.notification_cache[session_id] = deque(
                maxlen=MAX_NOTIFICATION_CACHE_SIZE
            )
            logger.info(f"Initialized notification cache for session {session_id}")

    def _cache_notification(
        self, session_id: str, notification: FilteredCardPushNotificationComponentData
    ) -> None:
        """
        Cache a notification for later batch sending.
        Automatically evicts oldest notifications when cache exceeds MAX_NOTIFICATION_CACHE_SIZE.

        Args:
            session_id: Session ID
            notification: Notification to cache
        """
        self._init_notification_cache(session_id)
        self.notification_cache[session_id].append(notification)
        logger.debug(
            f"Cached notification for session {session_id}. "
            f"Cache size: {len(self.notification_cache[session_id])}"
        )

    def _get_cached_notifications(
        self, session_id: str
    ) -> List[FilteredCardPushNotificationComponentData]:
        """
        Get all cached notifications for a session.

        Args:
            session_id: Session ID

        Returns:
            List of cached notifications (oldest to newest)
        """
        if session_id not in self.notification_cache:
            return []
        return list(self.notification_cache[session_id])

    def _clear_notification_cache(self, session_id: str) -> None:
        """
        Clear notification cache for a session.

        Args:
            session_id: Session ID
        """
        if session_id in self.notification_cache:
            self.notification_cache[session_id].clear()
            logger.info(f"Cleared notification cache for session {session_id}")

    async def _parse_trading_request(self, query: str) -> TradingRequest:
        """
        Parse natural language query to extract trading parameters

        Args:
            query: User's natural language query

        Returns:
            TradingRequest object with parsed parameters
        """
        try:
            parse_prompt = f"""
            Parse the following user query and extract auto trading configuration parameters:
            
            User query: "{query}"
            
            Please identify:
            1. crypto_symbols: List of cryptocurrency symbols to trade (e.g., BTC-USD, ETH-USD, SOL-USD)
               - If user mentions "Bitcoin", extract as "BTC-USD"
               - If user mentions "Ethereum", extract as "ETH-USD"
               - If user mentions "Solana", extract as "SOL-USD"
               - Always use format: SYMBOL-USD
            2. initial_capital: Initial trading capital in USD (default: 100000 if not specified)
            3. use_ai_signals: Whether to use AI-enhanced signals (default: true)
            4. agent_model: Model ID for trading decisions (default: DEFAULT_AGENT_MODEL)
            
            Examples:
            - "Trade Bitcoin and Ethereum with $50000" -> {{"crypto_symbols": ["BTC-USD", "ETH-USD"], "initial_capital": 50000, "use_ai_signals": true}}
            - "Start auto trading BTC-USD" -> {{"crypto_symbols": ["BTC-USD"], "initial_capital": 100000, "use_ai_signals": true}}
            - "Trade BTC with AI signals" -> {{"crypto_symbols": ["BTC-USD"], "initial_capital": 100000, "use_ai_signals": true}}
            - "Trade BTC with AI signals using DeepSeek model" -> {{"crypto_symbols": ["BTC-USD"], "initial_capital": 100000, "use_ai_signals": true, "agent_models": ["deepseek/deepseek-v3.1-terminus"]}}
            - "Trade Bitcoin, SOL, Eth and DOGE with 100000 capital, using x-ai/grok-4, deepseek/deepseek-v3.1-terminus model" -> {{"crypto_symbols": ["BTC-USD", "SOL-USD", "ETH-USD", "DOGE-USD"], "initial_capital": 100000, "use_ai_signals": true, "agent_models": ["x-ai/grok-4", "deepseek/deepseek-v3.1-terminus"]}}
            """

            response = await self.parser_agent.arun(parse_prompt)
            raw = getattr(response, "content", response)

            # Normalize to dict from JSON-like text
            data = None
            if isinstance(raw, dict):
                data = raw
            elif isinstance(raw, str):
                text = raw.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```[a-zA-Z0-9_\-]*\n|```$", "", text).strip()
                # Extract JSON object boundaries if extra text exists
                if "{" in text and "}" in text:
                    text = text[text.find("{") : text.rfind("}") + 1]
                # Try strict JSON
                try:
                    data = json.loads(text)
                except Exception:
                    # Heuristic: replace single quotes and trailing commas
                    try:
                        text2 = re.sub(r",\s*([}\]])", r"\1", text.replace("'", '"'))
                        data = json.loads(text2)
                    except Exception:
                        data = None

            if not isinstance(data, dict):
                raise ValueError("Parser did not return a valid JSON object")

            # Accept both agent_model (str) and agent_models (list)
            if "agent_models" not in data and "agent_model" in data:
                am = data.pop("agent_model")
                data["agent_models"] = [am] if isinstance(am, str) else list(am or [])

            # Normalize crypto_symbols: allow comma-separated string
            cs = data.get("crypto_symbols")
            if isinstance(cs, str):
                data["crypto_symbols"] = [s.strip() for s in cs.split(",") if s.strip()]

            # Build and validate TradingRequest
            trading_request = TradingRequest(**data)
            logger.info(f"Parsed trading request (structured): {trading_request}")
            return trading_request

        except Exception as e:
            logger.error(f"Failed to parse trading request: {e}")
            raise ValueError(
                f"Could not parse trading configuration from query: {query}"
            )

    @staticmethod
    def _has_trading_intent(query: str) -> bool:
        """Heuristic check to determine whether text expresses trading intent."""

        lowered = query.lower()
        keywords = [
            "trade",
            "trading",
            "buy",
            "sell",
            "下单",
            "交易",
            "开仓",
            "平仓",
            "买入",
            "卖出",
            "实盘",
            "真仓",
            "模拟",
            "paper",
            "okx",
            "止损",
            "加仓",
        ]
        if any(k in lowered for k in keywords):
            return True

        # Symbol patterns such as BTC-USD or ETH-USDT
        if re.search(r"\b[A-Z]{2,10}-(USD|USDT)\b", query, re.IGNORECASE):
            return True

        # Coin tickers without suffix (BTC, ETH, SOL, etc.)
        if re.search(
            r"\b(BTC|ETH|SOL|DOGE|OKB|XRP|LTC|BNB|ADA|MATIC|TON|SUI|AVAX)\b",
            query,
            re.IGNORECASE,
        ):
            return True

        return False

    def _initialize_ai_signal_generator(
        self, config: AutoTradingConfig
    ) -> Optional[AISignalGenerator]:
        """Initialize AI signal generator if configured.

        Uses the centralized configuration system with proper provider selection.
        Supports any provider configured in the config system.

        Args:
            config: AutoTradingConfig with use_ai_signals, agent_model, and agent_provider settings

        Returns:
            AISignalGenerator instance or None if AI signals are disabled or creation fails
        """
        if not config.use_ai_signals:
            return None

        try:
            # Use centralized configuration system for model creation
            # Supports automatic provider detection and fallback
            from valuecell.adapters.models.factory import create_model

            # Check for environment variable override
            model_id_override = os.getenv(ENV_SIGNAL_MODEL_ID)
            model_id = model_id_override or config.agent_model

            # Create model with provider auto-detection or explicit provider
            llm_client = create_model(
                model_id=model_id,
                provider=config.agent_provider,  # None = auto-detect
                use_fallback=True,  # Enable fallback to other providers
            )

            logger.info(
                f"Initialized AI signal generator: model_id={model_id}, "
                f"provider={config.agent_provider or 'auto-detect'}"
            )
            return AISignalGenerator(llm_client)

        except Exception as e:
            logger.error(f"Failed to initialize AI signal generator: {e}")
            logger.info(
                "Hint: Make sure provider API keys are configured in .env file. "
                "Check configs/providers/ for required environment variables. "
                "AI signals will be disabled for this trading instance."
            )
            return None

    def _get_instance_status_component_data(
        self, session_id: str, instance_id: str
    ) -> Optional[FilteredCardPushNotificationComponentData]:
        """
        Generate portfolio status report in rich text format

        Returns:
            FilteredCardPushNotificationComponentData object or None if instance not found
        """
        if session_id not in self.trading_instances:
            return None

        if instance_id not in self.trading_instances[session_id]:
            return None

        instance = self.trading_instances[session_id][instance_id]
        executor: TradingExecutor = instance["executor"]
        config: AutoTradingConfig = instance["config"]

        # Get comprehensive portfolio summary
        portfolio_summary = executor.get_portfolio_summary()

        # Calculate overall statistics
        total_pnl = portfolio_summary["portfolio"]["total_pnl"]
        pnl_pct = portfolio_summary["portfolio"]["pnl_percentage"]
        portfolio_value = portfolio_summary["portfolio"]["total_value"]
        available_cash = portfolio_summary["cash"]["available"]

        # Build rich text output
        output = []

        # Header
        output.append(f"📊 **Trading Portfolio Status** - {instance_id}")
        output.append("\n**Instance Configuration**")
        output.append(f"- Model: `{config.agent_model}`")
        output.append(f"- Symbols: {', '.join(config.crypto_symbols)}")
        output.append(
            f"- Status: {'🟢 Active' if instance['active'] else '🔴 Stopped'}"
        )

        # Portfolio Summary Section
        output.append("\n💰 **Portfolio Summary**")
        output.append("\n**Overall Performance**")
        output.append(f"- Initial Capital: `${config.initial_capital:,.2f}`")
        output.append(f"- Current Value: `${portfolio_value:,.2f}`")

        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        pnl_sign = "+" if total_pnl >= 0 else ""
        output.append(
            f"- Total P&L: {pnl_emoji} **{pnl_sign}${total_pnl:,.2f}** ({pnl_sign}{pnl_pct:.2f}%)"
        )

        output.append("\n**Cash Position**")
        output.append(f"- Available Cash: `${available_cash:,.2f}`")

        # Current Positions Section
        output.append(f"\n📈 **Current Positions ({len(executor.positions)})**")

        if executor.positions:
            output.append(
                "\n| Symbol | Type | Quantity | Avg Price | Current Price | Position Value | Unrealized P&L |"
            )
            output.append(
                "|--------|------|----------|-----------|---------------|----------------|----------------|"
            )

            for symbol, pos in executor.positions.items():
                try:
                    import yfinance as yf

                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period="1d", interval="1m")[
                        "Close"
                    ].iloc[-1]

                    # Calculate unrealized P&L
                    if pos.trade_type.value == "long":
                        unrealized_pnl = (current_price - pos.entry_price) * abs(
                            pos.quantity
                        )
                        position_value = abs(pos.quantity) * current_price
                    else:
                        unrealized_pnl = (pos.entry_price - current_price) * abs(
                            pos.quantity
                        )
                        position_value = pos.notional + unrealized_pnl

                    # Format row
                    pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                    pnl_sign = "+" if unrealized_pnl >= 0 else ""

                    output.append(
                        f"| **{symbol}** | {pos.trade_type.value.upper()} | "
                        f"{abs(pos.quantity):.4f} | ${pos.entry_price:,.2f} | "
                        f"${current_price:,.2f} | ${position_value:,.2f} | "
                        f"{pnl_emoji} {pnl_sign}${unrealized_pnl:,.2f} |"
                    )

                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
                    # Fallback display with entry price only
                    output.append(
                        f"| **{symbol}** | {pos.trade_type.value.upper()} | "
                        f"{abs(pos.quantity):.4f} | ${pos.entry_price:,.2f} | "
                        f"N/A | ${pos.notional:,.2f} | N/A |"
                    )
        else:
            output.append("\n*No open positions*")

        component_data = FilteredCardPushNotificationComponentData(
            title=f"{config.agent_model} Portfolio Status",
            data="\n".join(output),
            filters=[config.agent_model],
            table_title="Portfolio Detail",
            create_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        return component_data

    def _get_session_portfolio_chart_data(self, session_id: str) -> str:
        """
        Generate FilteredLineChartComponentData for all instances in a session
        Uses forward-fill strategy to handle missing timestamps

        Data format:
        [
            ['Time', 'model1', 'model2', 'model3'],
            ['2025-10-21 10:00:00', 100000, 50000, 30000],
            ['2025-10-21 10:01:00', 100234, 50123, 30045],
            ...
        ]

        Returns:
            JSON string of FilteredLineChartComponentData
        """
        if session_id not in self.trading_instances:
            return ""

        # Collect portfolio value history from all instances
        # Store as {model_id: {'initial_capital': float, 'history': [(timestamp, value)]}}
        model_data = {}

        for instance_id, instance in self.trading_instances[session_id].items():
            executor: TradingExecutor = instance["executor"]
            config: AutoTradingConfig = instance["config"]
            model_id = config.agent_model

            if model_id not in model_data:
                model_data[model_id] = {
                    "initial_capital": config.initial_capital,
                    "history": [],
                }

            portfolio_history = executor.get_portfolio_history()

            for snapshot in portfolio_history:
                model_data[model_id]["history"].append(
                    (snapshot.timestamp, snapshot.total_value)
                )

        if not model_data:
            return ""

        # Sort each model's history by timestamp
        for model_id in model_data:
            model_data[model_id]["history"].sort(key=lambda x: x[0])

        # Collect all unique timestamps across all models
        all_timestamps = set()
        for model_id, data in model_data.items():
            for timestamp, _ in data["history"]:
                all_timestamps.add(timestamp)

        if not all_timestamps:
            return ""

        sorted_timestamps = sorted(all_timestamps)
        model_ids = list(model_data.keys())

        # Build data array with forward-fill strategy
        # First row: ['Time', 'model1', 'model2', ...]
        data_array = [["Time"] + model_ids]

        # Track last known value for each model (for forward-fill)
        last_known_values = {
            model_id: data["initial_capital"] for model_id, data in model_data.items()
        }

        # Data rows: ['timestamp', value1, value2, ...]
        for timestamp in sorted_timestamps:
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            row = [timestamp_str]

            for model_id in model_ids:
                # Find value at this timestamp for this model
                value_at_timestamp = None
                for ts, val in model_data[model_id]["history"]:
                    if ts == timestamp:
                        value_at_timestamp = val
                        break

                # Update logic: use new value if found, otherwise forward-fill
                if value_at_timestamp is not None:
                    last_known_values[model_id] = value_at_timestamp
                    row.append(value_at_timestamp)
                else:
                    # Use last known value (forward-fill)
                    row.append(last_known_values[model_id])

            data_array.append(row)

        component_data = FilteredLineChartComponentData(
            title=f"Portfolio Value History - Session {session_id[:8]}",
            data=json.dumps(data_array),
            create_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )

        return component_data.model_dump_json()

    async def _handle_stop_command(
        self, session_id: str, query: str
    ) -> AsyncGenerator[StreamResponse, None]:
        """Handle stop command for trading instances"""
        query_lower = query.lower().strip()

        # Check if specific instance_id is provided
        instance_id = None
        if "instance_id:" in query_lower or "instance:" in query_lower:
            # Extract instance_id
            parts = query.split(":")
            if len(parts) >= 2:
                instance_id = parts[1].strip()

        if session_id not in self.trading_instances:
            yield streaming.message_chunk(
                "⚠️ No active trading instances found in this session.\n"
            )
            return

        if instance_id:
            # Stop specific instance
            if instance_id in self.trading_instances[session_id]:
                instance = self.trading_instances[session_id][instance_id]
                instance["active"] = False
                executor: TradingExecutor = instance["executor"]
                exchange_adapter: Optional[ExchangeBase] = instance.get(
                    "exchange_adapter"
                )
                if exchange_adapter:
                    try:
                        await exchange_adapter.disconnect()
                    except Exception as exc:
                        logger.warning(
                            "Failed to disconnect exchange adapter for %s: %s",
                            instance_id,
                            exc,
                        )
                portfolio_value = executor.get_portfolio_value()

                yield streaming.message_chunk(
                    f"🛑 **Trading Instance Stopped**\n\n"
                    f"Instance ID: `{instance_id}`\n"
                    f"Final Portfolio Value: ${portfolio_value:,.2f}\n"
                    f"Open Positions: {len(executor.positions)}\n\n"
                )
            else:
                yield streaming.message_chunk(
                    f"⚠️ Instance ID '{instance_id}' not found.\n"
                )
        else:
            # Stop all instances in this session
            count = 0
            for inst_id in self.trading_instances[session_id]:
                instance = self.trading_instances[session_id][inst_id]
                instance["active"] = False
                exchange_adapter: Optional[ExchangeBase] = instance.get(
                    "exchange_adapter"
                )
                if exchange_adapter:
                    try:
                        await exchange_adapter.disconnect()
                    except Exception as exc:
                        logger.warning(
                            "Failed to disconnect exchange adapter for %s: %s",
                            inst_id,
                            exc,
                        )
                count += 1

            yield streaming.message_chunk(
                f"🛑 **All Trading Instances Stopped**\n\n"
                f"Stopped {count} instance(s) in session: {session_id[:8]}\n\n"
            )

    async def _handle_status_command(
        self, session_id: str
    ) -> AsyncGenerator[StreamResponse, None]:
        """Handle status query command"""
        if (
            session_id not in self.trading_instances
            or not self.trading_instances[session_id]
        ):
            yield streaming.message_chunk(
                "⚠️ No trading instances found in this session.\n"
            )
            return

        status_message = f"📊 **Session Status** - {session_id[:8]}\n\n"
        status_message += (
            f"**Total Instances:** {len(self.trading_instances[session_id])}\n\n"
        )

        for instance_id, instance in self.trading_instances[session_id].items():
            executor: TradingExecutor = instance["executor"]
            config: AutoTradingConfig = instance["config"]

            status = "🟢 Active" if instance["active"] else "🔴 Stopped"
            portfolio_value = executor.get_portfolio_value()
            total_pnl = portfolio_value - config.initial_capital

            status_message += (
                f"**Instance:** `{instance_id}`  {status}\n"
                f"- Model: {config.agent_model}\n"
                f"- Exchange: {config.exchange.value.upper()}\n"
                f"- Symbols: {', '.join(config.crypto_symbols)}\n"
                f"- Portfolio Value: ${portfolio_value:,.2f}\n"
                f"- P&L: ${total_pnl:,.2f}\n"
                f"- Open Positions: {len(executor.positions)}\n"
                f"- Total Trades: {len(executor.get_trade_history())}\n"
                f"- Checks: {instance['check_count']}\n\n"
            )

        logger.info(f"Status message: {status_message}")

    async def _handle_switch_exchange_command(
        self, session_id: str, target: str
    ) -> AsyncGenerator[StreamResponse, None]:
        """Switch exchange adapter for all active instances in a session.

        Args:
            session_id: Current session identifier
            target: Target exchange string (e.g., 'paper', 'okx')
        """
        target = (target or "").strip().lower()
        valid = {e.value: e for e in ExchangeType}
        if target not in valid:
            yield streaming.failed(
                f"不支持的交易所/模式: {target}. 可选: {', '.join(valid.keys())}"
            )
            return

        target_ex = valid[target]

        if session_id not in self.trading_instances or not self.trading_instances[session_id]:
            yield streaming.message_chunk(
                f"当前会话没有运行中的交易实例。切换默认模式为 {target_ex.value.upper()} 只会影响新建实例。"
            )
            return

        count = 0
        success = 0
        for inst_id, instance in self.trading_instances[session_id].items():
            count += 1
            try:
                config: AutoTradingConfig = instance["config"]
                executor: TradingExecutor = instance["executor"]

                # Replace adapter
                config.exchange = target_ex
                new_adapter = self._create_exchange_adapter(config)
                if new_adapter:
                    try:
                        await new_adapter.connect()
                    except Exception as exc:
                        logger.error("Failed to connect new adapter for %s: %s", inst_id, exc)
                        new_adapter = None

                # Disconnect old adapter
                old_adapter = instance.get("exchange_adapter")
                if old_adapter:
                    try:
                        await old_adapter.disconnect()
                    except Exception:
                        pass

                # Apply
                instance["exchange_adapter"] = new_adapter
                executor.exchange = new_adapter
                success += 1

            except Exception as exc:
                logger.error("切换实例 %s 交易所失败: %s", inst_id, exc)

        yield streaming.message_chunk(
            f"🔄 已切换 {success}/{count} 个实例到: {target_ex.value.upper()}。新下单将实时生效。"
        )
        return

    async def stream(
        self,
        query: str,
        session_id: str,
        task_id: str,
        dependencies: Optional[Dict] = None,
    ) -> AsyncGenerator[StreamResponse, None]:
        """
        Process trading requests and manage multiple trading instances per session.

        Args:
            query: User's natural language query
            session_id: Session ID
            task_id: Task ID
            dependencies: Optional dependencies

        Yields:
            StreamResponse: Trading setup, execution updates, and data visualizations
        """
        # Track created instances for cleanup
        created_instances = []

        try:
            logger.info(
                f"Processing auto trading request - session: {session_id}, task: {task_id}"
            )

            query_lower = query.lower().strip()

            # Handle stop commands
            if any(
                cmd in query_lower.split()
                for cmd in ["stop", "pause", "halt", "停止", "暂停"]
            ):
                async for response in self._handle_stop_command(session_id, query):
                    yield response
                return

            # Handle switch exchange commands
            if (
                query_lower.startswith("switch_exchange")
                or query_lower.startswith("exchange ")
                or query_lower.startswith("切换交易所")
                or query_lower.startswith("切换 ")
                or query_lower in {"paper", "okx", "真仓", "实盘", "模拟", "模拟盘"}
            ):
                # Extract target token
                cleaned = (
                    query_lower.replace("切换交易所", "")
                    .replace("switch_exchange", "")
                    .replace("exchange", "")
                    .replace("切换", "")
                    .strip()
                )
                parts = cleaned.split()
                target = parts[0] if parts else query_lower
                # Map Chinese synonyms
                if target in {"真仓", "实盘", "live"}:
                    target = "okx"
                if target in {"模拟", "模拟盘", "paper"}:
                    target = "paper"
                async for response in self._handle_switch_exchange_command(
                    session_id, target
                ):
                    yield response
                return

            # Handle status query commands
            if any(
                cmd in query_lower.split()
                for cmd in ["status", "summary", "状态", "摘要"]
            ):
                async for response in self._handle_status_command(session_id):
                    yield response
                return

            if not self._has_trading_intent(query):
                logger.info("Detected non-trading intent message; skipping trading flow.")
                yield streaming.message_chunk(
                    "🤖 我是自动交易助手。如需发起交易，请提供交易品种、资金规模及交易模式，例如：\n"
                    "“使用 100000 美元在 OKX 交易 BTC-USD，开启 AI 信号”。\n"
                    "若想进行普通咨询，请改用 ResearchAgent 或网页端的其他模块。"
                )
                return

            # Parse natural language query to extract trading configuration
            yield streaming.message_chunk("🔍 **解析交易请求...**\n\n")

            try:
                trading_request = await self._parse_trading_request(query)
                logger.info(f"Parsed request: {trading_request}")
            except Exception as e:
                logger.error(f"Failed to parse trading request: {e}")
                yield streaming.message_chunk(
                    (
                        "⚠️ 我未能识别出完整的交易配置。请说明要交易的币种（例如 BTC-USD）、"
                        "投入资金以及是否开启 AI 信号。"
                    )
                )
                return

            # Initialize session structure if needed
            if session_id not in self.trading_instances:
                self.trading_instances[session_id] = {}

            # Initialize notification cache for this session
            self._init_notification_cache(session_id)

            # Get list of models to create instances for
            agent_models = trading_request.agent_models or [DEFAULT_AGENT_MODEL]

            # Create one trading instance per model
            yield streaming.message_chunk(
                f"🚀 **Creating {len(agent_models)} trading instance(s)...**\n\n"
            )

            for model_id in agent_models:
                # Generate unique instance ID for this model
                instance_id = self._generate_instance_id(task_id, model_id)

                # Determine exchange preference (request override > environment)
                exchange_pref = (
                    trading_request.exchange
                    if trading_request.exchange is not None
                    else os.getenv("EXCHANGE", ExchangeType.PAPER.value)
                )
                try:
                    exchange_type = ExchangeType(str(exchange_pref).lower())
                except ValueError:
                    logger.warning(
                        "Unsupported exchange '%s'. Falling back to paper trading.",
                        exchange_pref,
                    )
                    exchange_type = ExchangeType.PAPER

                # Create configuration for this specific model
                config = AutoTradingConfig(
                    initial_capital=trading_request.initial_capital or 100000,
                    crypto_symbols=trading_request.crypto_symbols,
                    use_ai_signals=trading_request.use_ai_signals or False,
                    agent_model=model_id,
                    exchange=exchange_type,
                )

                # Initialize exchange adapter if configured
                exchange_adapter = self._create_exchange_adapter(config)
                if exchange_adapter:
                    try:
                        await exchange_adapter.connect()
                    except Exception as exc:
                        logger.error(
                            "Failed to connect to %s exchange: %s",
                            config.exchange.value,
                            exc,
                        )
                        exchange_adapter = None

                # Initialize executor
                executor = TradingExecutor(config, exchange=exchange_adapter)

                # Initialize AI signal generator if enabled
                ai_signal_generator = self._initialize_ai_signal_generator(config)

                # Store instance
                self.trading_instances[session_id][instance_id] = {
                    "instance_id": instance_id,
                    "config": config,
                    "executor": executor,
                    "ai_signal_generator": ai_signal_generator,
                    "exchange_adapter": exchange_adapter,
                    "active": True,
                    "created_at": datetime.now(),
                    "check_count": 0,
                    "last_check": None,
                }

                created_instances.append(instance_id)

                # Display configuration for this instance
                ai_status = "✅ Enabled" if config.use_ai_signals else "❌ Disabled"
                exchange_name = config.exchange.value.upper()
                config_message = (
                    f"✅ **Trading Instance Created**\n\n"
                    f"**Instance ID:** `{instance_id}`\n"
                    f"**Model:** `{model_id}`\n\n"
                    f"**Configuration:**\n"
                    f"- Trading Symbols: {', '.join(config.crypto_symbols)}\n"
                    f"- Initial Capital: ${config.initial_capital:,.2f}\n"
                    f"- Exchange: {exchange_name}\n"
                    f"- Check Interval: {config.check_interval}s (1 minute)\n"
                    f"- Risk Per Trade: {config.risk_per_trade * 100:.1f}%\n"
                    f"- Max Positions: {config.max_positions}\n"
                    f"- AI Signals: {ai_status}\n\n"
                )

                yield streaming.message_chunk(config_message)

            # Summary message
            yield streaming.message_chunk(
                f"**Session ID:** `{session_id[:8]}`\n"
                f"**Total Active Instances in Session:** {len(self.trading_instances[session_id])}\n\n"
                f"🚀 **Starting continuous trading for all instances...**\n"
                f"All instances will run continuously until stopped.\n\n"
            )

            # Initialize all instances with portfolio snapshots
            # Use unified timestamp for initial snapshots to align chart data
            unified_initial_timestamp = datetime.now()
            for instance_id in created_instances:
                instance = self.trading_instances[session_id][instance_id]
                executor = instance["executor"]
                config = instance["config"]

                # Send initial portfolio snapshot - cache it
                portfolio_value = executor.get_portfolio_value()
                executor.snapshot_portfolio(unified_initial_timestamp)

                initial_portfolio_msg = FilteredCardPushNotificationComponentData(
                    title=f"{config.agent_model} Portfolio",
                    data=f"💰 **Initial Portfolio**\nTotal Value: ${portfolio_value:,.2f}\nAvailable Capital: ${executor.current_capital:,.2f}\n",
                    filters=[config.agent_model],
                    table_title="Portfolio Detail",
                    create_time=datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
                # Cache the initial notification
                self._cache_notification(session_id, initial_portfolio_msg)

            # Set check interval
            check_interval = DEFAULT_CHECK_INTERVAL

            # Create semaphore to limit concurrent instance processing (max 10)
            semaphore = asyncio.Semaphore(10)

            # Main trading loop - monitor all instances in parallel
            yield streaming.message_chunk(
                "📈 **Starting monitoring loop for all instances...**\n\n"
            )

            # Check if any instance is still active
            while any(
                self.trading_instances[session_id][inst_id]["active"]
                for inst_id in created_instances
                if inst_id in self.trading_instances[session_id]
            ):
                try:
                    # Create unified timestamp for this iteration to align snapshots
                    unified_timestamp = datetime.now()

                    # Process all active instances concurrently using task pool
                    tasks = []
                    for instance_id in created_instances:
                        # Skip if instance was removed or is inactive
                        if instance_id not in self.trading_instances[session_id]:
                            continue

                        instance = self.trading_instances[session_id][instance_id]
                        if not instance["active"]:
                            continue

                        # Create task for this instance with semaphore control and unified timestamp
                        task = asyncio.create_task(
                            self._process_trading_instance(
                                session_id, instance_id, semaphore, unified_timestamp
                            )
                        )
                        tasks.append(task)

                    # Wait for all instance tasks to complete (process concurrently)
                    if tasks:
                        # Gather all tasks and handle any exceptions
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Log any exceptions that occurred
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                logger.error(
                                    f"Task {i} failed with exception: {result}"
                                )

                    # After processing all instances, send batched notifications
                    cached_notifications = self._get_cached_notifications(session_id)
                    if cached_notifications:
                        logger.info(
                            f"Sending {len(cached_notifications)} cached notifications for session {session_id}"
                        )
                        # Convert all cached notifications to a list of dicts for batch sending
                        batch_data = [
                            notif.model_dump() for notif in cached_notifications
                        ]
                        # Send as a single batch component - frontend will receive all historical data
                        yield streaming.component_generator(
                            json.dumps(batch_data),
                            ComponentType.FILTERED_CARD_PUSH_NOTIFICATION,
                            component_id=f"trading_status_{session_id}",
                        )

                    # Send chart data (not cached, sent separately)
                    chart_data = self._get_session_portfolio_chart_data(session_id)
                    if chart_data:
                        yield streaming.component_generator(
                            content=chart_data,
                            component_type=ComponentType.FILTERED_LINE_CHART,
                            component_id=f"portfolio_chart_{session_id}",
                        )

                    # Wait for next check interval - only sleep once after processing all instances
                    logger.info(f"Waiting {check_interval}s until next check...")
                    await asyncio.sleep(check_interval)

                except Exception as e:
                    logger.error(f"Error during trading cycle: {e}")
                    yield streaming.message_chunk(
                        f"⚠️ **Error during trading cycle**: {str(e)}\n"
                        f"Continuing with next check...\n\n"
                    )
                    await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Critical error in stream method: {e}")
            yield streaming.failed(f"Critical error: {str(e)}")
        finally:
            # Mark all created instances as inactive but keep data for history
            if session_id in self.trading_instances:
                for instance_id in created_instances:
                    if instance_id in self.trading_instances[session_id]:
                        self.trading_instances[session_id][instance_id]["active"] = (
                            False
                        )
                        logger.info(f"Stopped instance: {instance_id}")
