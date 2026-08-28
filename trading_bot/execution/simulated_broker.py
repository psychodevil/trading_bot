"""
Simulated Broker and Order Execution Engine.
Simulates realistic exchange order matching, bid-ask spread crossing, slippage / market impact,
continuous margin tracking, and periodic funding / borrow fee settlements.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional
import uuid

from trading_bot.core.instruments import Instrument, CryptoPerp, ForexPair, Stock, OptionContract
from trading_bot.core.events import (
    Order, OrderType, OrderSide, OrderStatus, Fill, Position, PortfolioState, MarketQuote, Bar
)
from trading_bot.optimizer.cost_model import TransactionCostModel


class SimulatedBroker:
    """
    Realistic exchange simulator supporting multi-asset order routing and portfolio margin.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        cost_model: Optional[TransactionCostModel] = None
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.cost_model = cost_model or TransactionCostModel()
        self.positions: Dict[str, Position] = {}
        self.instruments: Dict[str, Instrument] = {}
        self.latest_quotes: Dict[str, MarketQuote] = {}
        self.active_orders: List[Order] = []
        self.fill_history: List[Fill] = []
        self.current_timestamp: float = 0.0
        self.last_holding_settlement_timestamp: float = 0.0

    def register_instrument(self, instrument: Instrument):
        """Register an instrument's contract specs."""
        self.instruments[instrument.symbol] = instrument
        if instrument.symbol not in self.positions:
            self.positions[instrument.symbol] = Position(
                symbol=instrument.symbol,
                multiplier=instrument.multiplier
            )

    def get_portfolio_state(self) -> PortfolioState:
        """Get current snapshot of portfolio equity, cash, and positions."""
        return PortfolioState(
            timestamp=self.current_timestamp,
            cash=self.cash,
            positions={k: Position(**pos.__dict__) for k, pos in self.positions.items()},
            initial_cash=self.initial_cash
        )

    def on_quote(self, quote: MarketQuote, funding_rate_per_interval: float = 0.0):
        """Process incoming market quote and update positions and pending orders."""
        self.latest_quotes[quote.symbol] = quote
        self.current_timestamp = quote.timestamp

        # Update position mark price
        if quote.symbol in self.positions:
            pos = self.positions[quote.symbol]
            pos.current_price = quote.mid_price
            pos.last_update_timestamp = quote.timestamp

        # Accrue holding carry costs
        if self.last_holding_settlement_timestamp > 0:
            elapsed = max(0.0, quote.timestamp - self.last_holding_settlement_timestamp)
            self._settle_holding_costs(elapsed, funding_rate_per_interval)
        self.last_holding_settlement_timestamp = quote.timestamp

        # Match open limit/stop orders
        self._match_open_orders(quote)

    def on_bar(self, bar: Bar, funding_rate_per_interval: float = 0.0):
        """Process incoming OHLCV bar by synthesizing a quote and updating state."""
        half_spread = bar.close * 0.0002
        quote = MarketQuote(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            bid=bar.close - half_spread,
            ask=bar.close + half_spread,
            bid_size=bar.volume * 0.5 if bar.volume > 0 else 1000.0,
            ask_size=bar.volume * 0.5 if bar.volume > 0 else 1000.0,
            last_price=bar.close
        )
        self.on_quote(quote, funding_rate_per_interval)

    def submit_order(self, order: Order) -> Optional[Fill]:
        """Submit an order to the broker. Market orders execute immediately."""
        order.timestamp = self.current_timestamp
        inst = self.instruments.get(order.symbol)
        if not inst:
            order.status = OrderStatus.REJECTED
            return None

        # Round quantity according to instrument lot size
        order.quantity = inst.round_quantity(order.quantity)
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            return None

        quote = self.latest_quotes.get(order.symbol)
        if not quote:
            # Queue order until quote arrives
            self.active_orders.append(order)
            return None

        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, quote, inst)
        else:
            self.active_orders.append(order)
            return None

    def execute_target_weight(
        self,
        symbol: str,
        target_weight: float,
        quote: Optional[MarketQuote] = None
    ) -> Optional[Fill]:
        """
        Rebalances portfolio to match target weight on symbol.
        """
        inst = self.instruments.get(symbol)
        if not inst:
            raise ValueError(f"Unknown instrument symbol: {symbol}")

        mkt_quote = quote or self.latest_quotes.get(symbol)
        if not mkt_quote:
            return None

        eq = self.get_portfolio_state().equity
        if eq <= 0:
            return None

        target_signed_notional = target_weight * eq
        target_qty = target_signed_notional / (mkt_quote.mid_price * inst.multiplier)
        current_pos = self.positions.get(symbol)
        current_qty = current_pos.quantity if current_pos else 0.0

        delta_qty = target_qty - current_qty
        rounded_delta_qty = inst.round_quantity(abs(delta_qty))

        if rounded_delta_qty < inst.lot_size:
            # Below minimum lot size
            return None

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        order = Order(
            order_id=f"ord_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=rounded_delta_qty,
            timestamp=self.current_timestamp
        )

        return self.submit_order(order)

    def _execute_market_order(self, order: Order, quote: MarketQuote, inst: Instrument) -> Fill:
        """Fills market order with spread crossing and non-linear market impact."""
        is_buy = (order.side == OrderSide.BUY)
        base_price = quote.ask if is_buy else quote.bid

        # Calculate slippage from market impact
        portfolio_eq = max(1.0, self.get_portfolio_state().equity)
        notional_order = order.quantity * base_price * inst.multiplier
        weight_delta = notional_order / portfolio_eq

        impact_slippage_pct = self.cost_model.impact_coefficient * (weight_delta ** (self.cost_model.impact_exponent - 1.0))
        slippage_price_diff = base_price * impact_slippage_pct

        fill_price = inst.round_price(base_price + slippage_price_diff if is_buy else base_price - slippage_price_diff)
        fee = inst.calculate_transaction_fee(order.quantity, fill_price, is_taker=True)

        # Update position and cash
        pos = self.positions.setdefault(order.symbol, Position(symbol=order.symbol, multiplier=inst.multiplier))
        old_qty = pos.quantity
        fill_qty = order.quantity if is_buy else -order.quantity
        new_qty = old_qty + fill_qty

        # Realized PnL calculation if reducing / flipping position
        if (old_qty > 0 and not is_buy) or (old_qty < 0 and is_buy):
            closed_qty = min(abs(old_qty), abs(fill_qty))
            pnl_direction = 1.0 if old_qty > 0 else -1.0
            realized = closed_qty * (fill_price - pos.entry_price) * inst.multiplier * pnl_direction
            pos.realized_pnl += realized
            self.cash += realized

        # Update average entry price
        if new_qty == 0:
            pos.entry_price = 0.0
        elif (old_qty >= 0 and fill_qty > 0) or (old_qty <= 0 and fill_qty < 0):
            total_cost = (abs(old_qty) * pos.entry_price + abs(fill_qty) * fill_price)
            pos.entry_price = total_cost / abs(new_qty)
        elif abs(fill_qty) > abs(old_qty):
            # Flipped position
            pos.entry_price = fill_price

        pos.quantity = new_qty
        pos.current_price = quote.mid_price
        pos.accumulated_fees += fee
        self.cash -= fee

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.total_fee = fee

        fill = Fill(
            fill_id=f"fill_{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=self.current_timestamp,
            slippage=abs(fill_price - quote.mid_price),
            is_maker=False
        )
        self.fill_history.append(fill)
        return fill

    def _match_open_orders(self, quote: MarketQuote):
        """Matches resting limit and stop orders."""
        remaining_orders = []
        inst = self.instruments.get(quote.symbol)
        if not inst:
            return

        for order in self.active_orders:
            if order.symbol != quote.symbol:
                remaining_orders.append(order)
                continue

            filled = False
            if order.order_type == OrderType.LIMIT and order.price is not None:
                if order.side == OrderSide.BUY and quote.ask <= order.price:
                    self._execute_market_order(order, quote, inst)
                    filled = True
                elif order.side == OrderSide.SELL and quote.bid >= order.price:
                    self._execute_market_order(order, quote, inst)
                    filled = True

            if not filled:
                remaining_orders.append(order)

        self.active_orders = remaining_orders

    def _settle_holding_costs(self, elapsed_seconds: float, funding_rate_per_interval: float):
        """Settles continuous carry costs across all open positions."""
        if elapsed_seconds <= 0:
            return

        for symbol, pos in self.positions.items():
            if pos.quantity == 0:
                continue

            inst = self.instruments.get(symbol)
            if not inst:
                continue

            # 1. Short stock borrow fee
            if isinstance(inst, Stock) and pos.quantity < 0:
                cost = inst.calculate_borrow_cost(pos.quantity, pos.current_price, elapsed_seconds)
                pos.borrow_costs_paid += cost
                self.cash -= cost

            # 2. Crypto perpetual funding rate
            elif isinstance(inst, CryptoPerp):
                # Scale interval funding rate to elapsed fraction
                interval_frac = elapsed_seconds / inst.funding_interval_seconds
                funding_cashflow = inst.calculate_funding_payment(
                    pos.quantity, pos.current_price, funding_rate_per_interval * interval_frac
                )
                pos.funding_pnl += funding_cashflow
                self.cash += funding_cashflow

            # 3. Forex overnight swap
            elif isinstance(inst, ForexPair):
                swap_cashflow = inst.calculate_swap_cost(pos.quantity, pos.current_price, elapsed_seconds)
                self.cash += swap_cashflow

