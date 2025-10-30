"""OKX exchange adapter using ccxt's asynchronous client"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt_async

from .base_exchange import ExchangeBase, ExchangeType, Order, OrderStatus

logger = logging.getLogger(__name__)


class OkxExchange(ExchangeBase):
    """Adapter that wraps CCXT's OKX client and exposes ExchangeBase interface."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        sandbox: bool = False,
        default_type: str = "spot",
        leverage: Optional[int] = None,
        margin_mode: Optional[str] = None,  # cash | cross | isolated
    ):
        """
        Initialize OKX exchange adapter.

        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX API passphrase
            sandbox: Use sandbox trading environment (default False)
            default_type: Market type (spot, swap, option, future)
            leverage: Optional leverage for derivative trading
        """
        super().__init__(ExchangeType.OKX)

        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        self.default_type = default_type
        self.leverage = leverage
        # tdMode for OKX: spot usually uses cash; derivatives use cross/isolated
        self.margin_mode = (margin_mode or ("cash" if default_type == "spot" else "cross")).lower()

        self._client = ccxt_async.okx(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": passphrase,
                "enableRateLimit": True,
                "options": {
                    "defaultType": default_type,
                },
            }
        )

        if sandbox:
            self._client.set_sandbox_mode(True)

        logger.info(
            "OKX exchange adapter initialised (mode=%s, type=%s)",
            "SANDBOX" if sandbox else "LIVE",
            default_type,
        )

    # ============ Helpers ============

    @staticmethod
    def _map_status(status: Optional[str]) -> OrderStatus:
        """Map CCXT order status to internal OrderStatus enum."""
        normalized = (status or "").lower()
        mapping = {
            "open": OrderStatus.PENDING,
            "new": OrderStatus.PENDING,
            "pending": OrderStatus.PENDING,
            "partially-filled": OrderStatus.PARTIALLY_FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "partial": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return mapping.get(normalized, OrderStatus.PENDING)

    def _build_order_from_ccxt(self, symbol: str, data: Dict[str, Any]) -> Order:
        """Convert CCXT order payload into local Order instance."""
        order_id = str(data.get("id") or data.get("orderId") or "okx_order")
        price = data.get("price")
        amount = data.get("amount") or data.get("filled") or 0.0
        ccxt_symbol = data.get("symbol") or symbol
        side = data.get("side", "buy")
        order_type = data.get("type", "limit")

        order = Order(
            order_id=order_id,
            symbol=ccxt_symbol,
            side=side,
            quantity=float(amount) if amount is not None else 0.0,
            price=float(price) if price not in (None, "") else 0.0,
            order_type=order_type,
        )
        order.status = self._map_status(data.get("status"))
        order.filled_quantity = float(data.get("filled", 0.0) or 0.0)
        order.filled_price = float(data.get("average", 0.0) or data.get("avgPrice") or 0.0)
        return order

    async def _ensure_markets(self) -> None:
        """Ensure OKX markets metadata is loaded."""
        if not getattr(self, "_markets_loaded", False):
            await self._client.load_markets()
            self._markets_loaded = True

    # ============ Connection Management ============

    async def connect(self) -> bool:
        """Authenticate and load markets."""
        try:
            await self._ensure_markets()
            if self.leverage and hasattr(self._client, "set_leverage"):
                try:
                    await self._client.set_leverage(self.leverage)
                except Exception as leverage_error:
                    logger.warning("Failed to set leverage %s: %s", self.leverage, leverage_error)
            self.is_connected = True
            return True
        except Exception as exc:
            await self.handle_connection_error(exc)
            return False

    async def disconnect(self) -> bool:
        """Close CCXT client session."""
        try:
            await self._client.close()
        finally:
            self.is_connected = False
        return True

    async def validate_connection(self) -> bool:
        """Ping OKX by fetching server time."""
        if not self.is_connected:
            return False
        try:
            await self._client.fetch_time()
            return True
        except Exception as exc:
            logger.error("OKX connection validation failed: %s", exc)
            self.is_connected = False
            return False

    # ============ Account Information ============

    async def get_balance(self) -> Dict[str, float]:
        """Fetch account balances."""
        try:
            await self._ensure_markets()
            balance = await self._client.fetch_balance()
            result: Dict[str, float] = {}
            totals = balance.get("total") or {}
            free_balances = balance.get("free") or {}

            for asset, amount in free_balances.items():
                total_amount = totals.get(asset, amount)
                if total_amount is None:
                    continue
                total_float = float(total_amount)
                if total_float != 0.0:
                    result[asset] = total_float

            return result
        except Exception as exc:
            logger.error("Failed to fetch OKX balance: %s", exc)
            return {}

    async def get_asset_balance(self, asset: str) -> float:
        """Fetch balance for specific asset."""
        balances = await self.get_balance()
        return float(balances.get(asset.upper(), 0.0))

    # ============ Market Data ============

    async def get_current_price(self, symbol: str) -> float:
        """Fetch latest traded price for symbol."""
        try:
            exchange_symbol = self.normalize_symbol(symbol)
            await self._ensure_markets()
            ticker = await self._client.fetch_ticker(exchange_symbol)
            last_price = ticker.get("last") or ticker.get("close") or 0.0
            return float(last_price)
        except Exception as exc:
            logger.error("Failed to fetch OKX ticker for %s: %s", symbol, exc)
            return 0.0

    async def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch 24h stats for symbol."""
        try:
            exchange_symbol = self.normalize_symbol(symbol)
            await self._ensure_markets()
            ticker = await self._client.fetch_ticker(exchange_symbol)
            return {
                "symbol": exchange_symbol,
                "current_price": float(ticker.get("last") or 0.0),
                "24h_high": float(ticker.get("high") or 0.0),
                "24h_low": float(ticker.get("low") or 0.0),
                "24h_volume": float(ticker.get("baseVolume") or 0.0),
                "24h_change": float(ticker.get("percentage") or 0.0),
                "raw": ticker,
            }
        except Exception as exc:
            logger.error("Failed to fetch OKX 24h ticker for %s: %s", symbol, exc)
            return {}

    # ============ Order Management ============

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        **kwargs: Any,
    ) -> Order:
        """Place order via OKX REST API."""
        exchange_symbol = self.normalize_symbol(symbol)
        params = dict(kwargs.get("params", {}))
        # Map common kwargs to OKX/ccxt params
        client_order_id = kwargs.get("client_order_id") or kwargs.get("clOrdId")
        time_in_force = kwargs.get("time_in_force") or kwargs.get("timeInForce")
        reduce_only = kwargs.get("reduce_only")
        # OKX tdMode: cash/cross/isolated
        params.setdefault("tdMode", self.margin_mode)
        if client_order_id:
            params["clOrdId"] = str(client_order_id)
        if time_in_force:
            params["timeInForce"] = str(time_in_force)
        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)
        try:
            await self._ensure_markets()
            order = await self._client.create_order(
                exchange_symbol,
                order_type,
                side.lower(),
                float(quantity),
                price if price is None else float(price),
                params,
            )
            order_obj = self._build_order_from_ccxt(exchange_symbol, order)
            self.orders[order_obj.order_id] = order_obj
            logger.info(
                "OKX order placed: %s %s %s %s @ %s",
                exchange_symbol,
                side,
                quantity,
                order_type,
                price if price is not None else "MARKET",
            )
            return order_obj
        except Exception as exc:
            logger.error("Failed to place OKX order for %s: %s", exchange_symbol, exc)
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel previously submitted order."""
        exchange_symbol = self.normalize_symbol(symbol)
        try:
            await self._ensure_markets()
            await self._client.cancel_order(order_id, exchange_symbol)
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
            logger.info("OKX order cancelled: %s", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel OKX order %s: %s", order_id, exc)
            return False

    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        """Retrieve order status."""
        exchange_symbol = self.normalize_symbol(symbol)
        try:
            await self._ensure_markets()
            order = await self._client.fetch_order(order_id, exchange_symbol)
            status = self._map_status(order.get("status"))
            if order_id in self.orders:
                self.orders[order_id].status = status
            return status
        except Exception as exc:
            logger.error("Failed to fetch OKX order status %s: %s", order_id, exc)
            return OrderStatus.REJECTED

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return list of open orders."""
        try:
            await self._ensure_markets()
            exchange_symbol = self.normalize_symbol(symbol) if symbol else None
            orders = await self._client.fetch_open_orders(symbol=exchange_symbol)
            return [self._build_order_from_ccxt(o.get("symbol", ""), o) for o in orders]
        except Exception as exc:
            logger.error("Failed to fetch OKX open orders: %s", exc)
            return []

    async def get_order_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> List[Order]:
        """Return order history entries."""
        try:
            await self._ensure_markets()
            exchange_symbol = self.normalize_symbol(symbol) if symbol else None
            orders = await self._client.fetch_orders(
                symbol=exchange_symbol, limit=limit
            )
            return [self._build_order_from_ccxt(o.get("symbol", ""), o) for o in orders]
        except Exception as exc:
            logger.error("Failed to fetch OKX order history: %s", exc)
            return []

    # ============ Position Management ============

    async def get_open_positions(
        self, symbol: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch open positions (derivatives only)."""
        positions: Dict[str, Dict[str, Any]] = {}

        if not hasattr(self._client, "fetch_positions"):
            return positions

        try:
            await self._ensure_markets()
            raw_positions = await self._client.fetch_positions()
            for position in raw_positions:
                pos_symbol = position.get("symbol")
                if symbol and pos_symbol != self.normalize_symbol(symbol):
                    continue

                size = float(position.get("contracts") or position.get("size") or 0.0)
                if size == 0:
                    continue

                positions[pos_symbol] = {
                    "symbol": pos_symbol,
                    "contracts": size,
                    "entry_price": float(position.get("entryPrice") or 0.0),
                    "notional": float(position.get("notional") or 0.0),
                    "unrealized_pnl": float(
                        position.get("unrealizedPnl")
                        or position.get("unrealizedProfit")
                        or 0.0
                    ),
                    "side": position.get("side"),
                    "raw": position,
                }
            return positions
        except Exception as exc:
            logger.error("Failed to fetch OKX positions: %s", exc)
            return {}

    async def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return details for a single position."""
        positions = await self.get_open_positions(symbol)
        return positions.get(self.normalize_symbol(symbol))

    # ============ Trade Execution ============

    async def execute_buy(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Order]:
        """Convenience wrapper for buy orders."""
        return await self.place_order(
            symbol,
            side="buy",
            quantity=quantity,
            price=price,
            order_type=kwargs.pop("order_type", "limit"),
            **kwargs,
        )

    async def execute_sell(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Order]:
        """Convenience wrapper for sell orders."""
        return await self.place_order(
            symbol,
            side="sell",
            quantity=quantity,
            price=price,
            order_type=kwargs.pop("order_type", "limit"),
            **kwargs,
        )

    # ============ Utilities ============

    def normalize_symbol(self, symbol: Optional[str]) -> str:
        """Normalize incoming symbols to OKX format (e.g. BTC/USDT).

        Notes:
            - Agent analysis uses yfinance symbols like BTC-USD; we map USD -> USDT for OKX spot.
            - Accepts formats: BTC-USD, BTC_USD, BTC/USDT, BTCUSDT.
        """
        if not symbol:
            raise ValueError("Symbol is required for OKX operations")

        cleaned = symbol.strip().upper()
        if "/" in cleaned:
            # Add settle currency for derivatives if missing
            if self.default_type != "spot" and ":" not in cleaned:
                return f"{cleaned}:USDT"
            return cleaned
        if "-" in cleaned:
            base, quote = cleaned.split("-", 1)
            # Map USD -> USDT by default for OKX spot
            if quote == "USD":
                quote = "USDT"
            sym = f"{base}/{quote}"
            if self.default_type != "spot":
                sym = f"{sym}:USDT"
            return sym
        if "_" in cleaned:
            base, quote = cleaned.split("_", 1)
            sym = f"{base}/{quote}"
            if self.default_type != "spot":
                sym = f"{sym}:USDT"
            return sym
        if cleaned.endswith("USDT"):
            base = cleaned[:-4]
            sym = f"{base}/USDT"
            if self.default_type != "spot":
                sym = f"{sym}:USDT"
            return sym
        if cleaned.endswith("USD"):
            base = cleaned[:-3]
            # Map USD -> USDT for OKX spot
            sym = f"{base}/USDT"
            if self.default_type != "spot":
                sym = f"{sym}:USDT"
            return sym
        # Fallback: treat as BASEQUOTE
        if cleaned.endswith("USDT"):
            base = cleaned[:-4]
            sym = f"{base}/USDT"
            if self.default_type != "spot":
                sym = f"{sym}:USDT"
            return sym
        return cleaned

    async def get_fee_tier(self) -> Dict[str, float]:
        """Fetch maker/taker fees if available."""
        try:
            await self._ensure_markets()
            fees = await self._client.fetch_trading_fees()
            maker = taker = None

            symbol_fee = fees.get("info", {}).get("data", [{}])[0]
            maker = float(symbol_fee.get("maker", 0.0))
            taker = float(symbol_fee.get("taker", 0.0))

            return {"maker": maker, "taker": taker}
        except Exception as exc:
            logger.warning("Failed to fetch OKX fee tier: %s", exc)
            return {}

    async def get_trading_limits(self, symbol: str) -> Dict[str, float]:
        """Return trading limits/precision."""
        try:
            await self._ensure_markets()
            exchange_symbol = self.normalize_symbol(symbol)
            market = self._client.market(exchange_symbol)
            limits = market.get("limits", {})
            amount_limits = limits.get("amount") or {}
            price_limits = limits.get("price") or {}
            precision = market.get("precision") or {}

            return {
                "min_amount": float(amount_limits.get("min") or 0.0),
                "max_amount": float(amount_limits.get("max") or 0.0),
                "min_price": float(price_limits.get("min") or 0.0),
                "max_price": float(price_limits.get("max") or 0.0),
                "amount_precision": float(precision.get("amount") or 0.0),
                "price_precision": float(precision.get("price") or 0.0),
            }
        except Exception as exc:
            logger.error("Failed to fetch OKX trading limits for %s: %s", symbol, exc)
            return {}
