"""
Event, Market Data, Order, and Portfolio State data structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Bar:
    """OHLCV candlestick bar for any given timeframe or sampling interval."""
    timestamp: float
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timeframe_seconds: float = 60.0
    trades_count: int = 0
    vwap: Optional[float] = None

    @property
    def mid_price(self) -> float:
        return 0.5 * (self.high + self.low)


@dataclass
class MarketQuote:
    """Top of book quote / tick update with bid-ask spread."""
    timestamp: float
    symbol: str
    bid: float
    ask: float
    bid_size: float = 100.0
    ask_size: float = 100.0
    last_price: Optional[float] = None

    @property
    def mid_price(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def half_spread_pct(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return 0.0
        return (self.ask - self.bid) / (2.0 * mid)


@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fee: float = 0.0
    client_tag: str = ""

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass
class Fill:
    """Execution fill report generated when an order executes."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: float
    slippage: float = 0.0
    is_maker: bool = False


@dataclass
class Position:
    """Individual instrument position tracker."""
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    multiplier: float = 1.0
    realized_pnl: float = 0.0
    accumulated_fees: float = 0.0
    borrow_costs_paid: float = 0.0
    funding_pnl: float = 0.0
    last_update_timestamp: float = 0.0

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.current_price * self.multiplier

    @property
    def signed_notional(self) -> float:
        return self.quantity * self.current_price * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        price_diff = self.current_price - self.entry_price
        return self.quantity * price_diff * self.multiplier

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.accumulated_fees - self.borrow_costs_paid + self.funding_pnl


@dataclass
class PortfolioState:
    """Complete portfolio equity, cash, margin, and positions snapshot."""
    timestamp: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_cash: float = 100000.0

    @property
    def unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def equity(self) -> float:
        """Total portfolio equity = Initial Cash + sum(Position Total PnL across all assets)."""
        total = self.initial_cash + sum(pos.total_pnl for pos in self.positions.values())
        return max(0.0, total)

    @property
    def leverage(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        return self.total_notional / eq

    def get_position_weight(self, symbol: str) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        return pos.signed_notional / eq

