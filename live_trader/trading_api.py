#!/usr/bin/env python3
"""Binance margin trading utility with helpers to open/close long and short positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException

from src.utils.config import Config


@dataclass
class OrderResult:
    """Normalized response for order requests."""

    order_id: Optional[int]
    status: str
    side: str
    type: str
    symbol: str
    quantity: float
    price: Optional[float]
    raw: Dict[str, Any]


class BinanceMarginTradingAPI:
    """Helper wrapper around Binance margin endpoints (cross by default)."""

    SPOT_TESTNET_URL = "https://testnet.binance.vision"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        is_isolated: bool = False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol.upper()
        self.testnet = testnet if testnet is not None else Config.BINANCE_TESTNET
        self.is_isolated = is_isolated

        api_key = api_key or Config.BINANCE_API_KEY
        api_secret = api_secret or Config.BINANCE_SECRET_KEY
        if not api_key or not api_secret:
            raise ValueError("Missing Binance API credentials.")

        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=self.testnet)
        if self.testnet:
            self.logger.info("Using Binance spot testnet endpoint.")
            self.client.API_URL = self.SPOT_TESTNET_URL

        self.base_asset, self.quote_asset = self._parse_symbol(self.symbol)
        self._init_symbol_filters()

    # ------------------------------------------------------------------ #
    # Parsing & account helpers
    # ------------------------------------------------------------------ #
    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        if symbol.endswith("USDT"):
            return symbol[:-4], "USDT"
        raise ValueError(f"Unsupported symbol format: {symbol}")

    def _get_margin_account_info(self) -> Dict[str, Any]:
        try:
            if self.is_isolated:
                return self.client.get_isolated_margin_account()
            return self.client.get_margin_account()
        except BinanceAPIException as exc:
            raise RuntimeError(f"Unable to fetch margin account info: {exc}")

    def get_balance(self, asset: str) -> float:
        info = self._get_margin_account_info()
        if self.is_isolated:
            for entry in info.get("assets", []):
                if entry.get("symbol") != self.symbol:
                    continue
                if asset == self.base_asset:
                    return float(entry["baseAsset"]["free"])
                if asset == self.quote_asset:
                    return float(entry["quoteAsset"]["free"])
        else:
            for entry in info.get("userAssets", []):
                if entry.get("asset") == asset:
                    return float(entry.get("free", 0))
        return 0.0

    def _get_borrowed_amount(self, asset: str) -> float:
        info = self._get_margin_account_info()
        if self.is_isolated:
            for entry in info.get("assets", []):
                if entry.get("symbol") != self.symbol:
                    continue
                if asset == self.base_asset:
                    return float(entry["baseAsset"]["borrowed"])
                if asset == self.quote_asset:
                    return float(entry["quoteAsset"]["borrowed"])
        else:
            for entry in info.get("userAssets", []):
                if entry.get("asset") == asset:
                    return float(entry.get("borrowed", 0))
        return 0.0

    def margin_loan(self, asset: str, amount: float) -> Dict[str, Any]:
        params = {"asset": asset, "amount": str(amount)}
        if self.is_isolated:
            params["isIsolated"] = "TRUE"
            params["symbol"] = self.symbol
        try:
            return self.client.create_margin_loan(**params)
        except BinanceAPIException as exc:
            raise RuntimeError(f"Margin loan failed: {exc}")

    def margin_repay(self, asset: str, amount: float) -> Dict[str, Any]:
        params = {"asset": asset, "amount": str(amount)}
        if self.is_isolated:
            params["isIsolated"] = "TRUE"
            params["symbol"] = self.symbol
        try:
            return self.client.create_margin_repay(**params)
        except BinanceAPIException as exc:
            raise RuntimeError(f"Margin repay failed: {exc}")

    def quantize_quantity(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        step = self.qty_step or 0.0001
        precision = int(round(-np.log10(step))) if step < 1 else 0
        quantized = max(self.min_qty or 0.0, round(np.floor(qty / step) * step, precision))
        return float(quantized)

    # ------------------------------------------------------------------ #
    # Margin order helpers
    # ------------------------------------------------------------------ #
    def open_long(
        self,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        **kwargs,
    ) -> OrderResult:
        quantity = self.quantize_quantity(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        return self._create_margin_order(
            "BUY",
            quantity,
            order_type,
            price,
            side_effect_type="MARGIN_BUY",
            **kwargs,
        )

    def close_long(
        self,
        quantity: Optional[float] = None,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        **kwargs,
    ) -> Optional[OrderResult]:
        holding = self.get_balance(self.base_asset)
        if holding <= 0:
            self.logger.info("No %s holdings to close.", self.base_asset)
            return None
        qty = self.quantize_quantity(min(holding, quantity or holding))
        if qty <= 0:
            self.logger.info("Quantity after quantization is zero for %s.", self.base_asset)
            return None
        order = self._create_margin_order(
            "SELL",
            qty,
            order_type,
            price,
            side_effect_type="AUTO_REPAY",
            **kwargs,
        )
        borrowed_quote = self._get_borrowed_amount(self.quote_asset)
        if borrowed_quote > 0:
            repay_amount = min(borrowed_quote, order.price * qty if order.price else borrowed_quote)
            if repay_amount > 0:
                self.margin_repay(self.quote_asset, repay_amount)
        return order

    def open_short(
        self,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        **kwargs,
    ) -> OrderResult:
        quantity = self.quantize_quantity(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        self.margin_loan(self.base_asset, quantity)
        return self._create_margin_order(
            "SELL",
            quantity,
            order_type,
            price,
            **kwargs,
        )

    def close_short(
        self,
        quantity: Optional[float] = None,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        **kwargs,
    ) -> Optional[OrderResult]:
        borrowed_base = self._get_borrowed_amount(self.base_asset)
        if borrowed_base <= 0:
            self.logger.info("No borrowed %s to repay.", self.base_asset)
            return None
        qty = self.quantize_quantity(min(borrowed_base, quantity or borrowed_base))
        if qty <= 0:
            self.logger.info("Quantity after quantization is zero for %s.", self.base_asset)
            return None
        order = self._create_margin_order(
            "BUY",
            qty,
            order_type,
            price,
            side_effect_type="AUTO_REPAY",
            **kwargs,
        )
        self.margin_repay(self.base_asset, qty)
        return order

    # ------------------------------------------------------------------ #
    def _create_margin_order(
        self,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        side_effect_type: Optional[str] = None,
        **kwargs,
    ) -> OrderResult:
        params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": side,
            "type": order_type.upper(),
            "quantity": str(quantity),
        }

        if params["type"] == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price.")
            params["price"] = float(price)
            params.setdefault("timeInForce", "GTC")

        params.update(kwargs)

        if self.is_isolated:
            params["isIsolated"] = "TRUE"
        if side_effect_type:
            params["sideEffectType"] = side_effect_type

        try:
            raw = self.client.create_margin_order(**params)
            self.logger.info("Margin order placed: %s %s %s qty=%s", params["type"], side, self.symbol, params["quantity"])
        except BinanceAPIException as exc:
            raise RuntimeError(f"Order placement failed: {exc}")

        fills = raw.get("fills", [])
        price = float(fills[0]["price"]) if fills else float(raw.get("price", 0) or 0)

        return OrderResult(
            order_id=raw.get("orderId"),
            status=raw.get("status", "UNKNOWN"),
            side=raw.get("side", side),
            type=raw.get("type", params["type"]),
            symbol=raw.get("symbol", self.symbol),
            quantity=float(raw.get("origQty", params["quantity"])),
            price=price if price > 0 else None,
            raw=raw,
        )

    def _init_symbol_filters(self) -> None:
        try:
            info = self.client.get_symbol_info(self.symbol)
        except BinanceAPIException as exc:
            raise RuntimeError(f"Failed to fetch symbol info: {exc}")

        if not info:
            raise RuntimeError(f"Symbol {self.symbol} not found.")

        lot = next((f for f in info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), {})
        notional = next((f for f in info.get("filters", []) if f.get("filterType") == "MIN_NOTIONAL"), {})

        self.min_qty = float(lot.get("minQty", 0.0))
        self.qty_step = float(lot.get("stepSize", 0.0001))
        self.min_notional = float(notional.get("minNotional", 0.0))
        self.logger.info(
            "Spot filters – min_qty=%s qty_step=%s min_notional=%s",
            self.min_qty,
            self.qty_step,
            self.min_notional,
        )


BinanceSpotTradingAPI = BinanceMarginTradingAPI  # Backwards compatibility

__all__ = ["BinanceMarginTradingAPI", "BinanceSpotTradingAPI", "OrderResult"]


if __name__ == "__main__":
    """
    Basic manual test harness. Populate the variables below with your Binance margin
    API credentials (use the spot testnet keys if you are testing in simulation).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    TESTNET = True  # Leave True when using testnet keys
    API_KEY = "xqH2OXMlaKbcqaXejaUCa005EOqlSegrhrCZwVMLcRrnNZ3yNGzkCkQgajMZ9r4o"
    API_SECRET = "TUAs42WDdp8QgBGRG6z50MQKSe7HzxaxVDHS0ssbJY0EzZTgbCRwLXucVoh1gv32"

    if not API_KEY or not API_SECRET:
        raise SystemExit("Please fill API_KEY and API_SECRET inside trading_api.py before running the test harness.")

    api = BinanceMarginTradingAPI(
        symbol="BTCUSDT",
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=TESTNET,
        is_isolated=False,
    )

    quote_balance = api.get_balance(api.quote_asset)
    print(f"{api.quote_asset} free balance: {quote_balance}")
    base_balance = api.get_balance(api.base_asset)
    print(f"{api.base_asset} free balance: {base_balance}")

    qty = api.quantize_quantity(0.001)
    print(f"Opening long position for {qty} {api.base_asset}...")
    long_order = api.open_long(quantity=qty)
    print("Long order:", long_order)

    print("Closing long position...")
    api.close_long()

    qty = api.quantize_quantity(0.001)
    print(f"Opening short position for {qty} {api.base_asset}...")
    short_order = api.open_short(quantity=qty)
    print("Short order:", short_order)

    print("Closing short position...")
    api.close_short()
