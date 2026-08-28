"""
Comprehensive Transaction and Holding Cost Models.
Captures linear exchange fees, bid-ask spreads, non-linear market impact / slippage,
short borrow rates, crypto perpetual funding rates, and option theta decay.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

from trading_bot.core.instruments import Instrument, CryptoPerp, ForexPair, OptionContract


@dataclass
class TransactionCostModel:
    """
    Transaction and turnover cost specification:
    Cost(delta_w) = (linear_fee + half_spread) * |delta_w| + impact_coeff * |delta_w|^impact_exponent
    """
    linear_fee_rate: float = 0.0005     # 5 bps taker fee
    bid_ask_half_spread: float = 0.0002 # 2 bps half-spread
    impact_coefficient: float = 0.001   # Slippage / price impact factor
    impact_exponent: float = 1.5        # Non-linear market impact power (1.5 - 2.0)
    fixed_order_cost: float = 0.0       # Fixed dollar ticket charge

    def calculate_turnover_cost(self, delta_w: float, portfolio_equity: float = 100000.0) -> float:
        """
        Calculates fraction of portfolio lost to transaction costs when shifting weight by delta_w.
        """
        abs_dw = abs(delta_w)
        if abs_dw < 1e-7:
            return 0.0

        linear_cost = (self.linear_fee_rate + self.bid_ask_half_spread) * abs_dw
        impact_cost = self.impact_coefficient * (abs_dw ** self.impact_exponent)
        fixed_cost_pct = (self.fixed_order_cost / portfolio_equity) if portfolio_equity > 0 else 0.0

        return linear_cost + impact_cost + fixed_cost_pct

    def marginal_turnover_cost(self, delta_w: float) -> float:
        """
        First derivative of transaction cost with respect to delta_w (for inaction band gradient checks).
        """
        abs_dw = abs(delta_w)
        if abs_dw < 1e-7:
            return self.linear_fee_rate + self.bid_ask_half_spread
        sign = 1.0 if delta_w > 0 else -1.0
        return sign * (self.linear_fee_rate + self.bid_ask_half_spread +
                       self.impact_exponent * self.impact_coefficient * (abs_dw ** (self.impact_exponent - 1.0)))


@dataclass
class HoldingCostModel:
    """
    Continuous carry cost over holding horizon dt:
    - Short borrow fee (stocks)
    - Perpetual funding rate (crypto)
    - Interest swap differential (forex)
    - Theta decay (options)
    """
    borrow_rate_annual: float = 0.03
    funding_rate_per_day: float = 0.0
    forex_swap_annual: float = 0.0
    option_theta_daily_pct: float = 0.0

    @classmethod
    def from_instrument(
        cls,
        instrument: Instrument,
        funding_rate_8h: float = 0.0001,
        underlying_spot: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> HoldingCostModel:
        """Construct calibrated holding cost model from instrument specs."""
        borrow_rate = instrument.borrow_rate_annual
        funding_rate_day = 0.0
        forex_swap = 0.0
        theta_pct = 0.0

        if isinstance(instrument, CryptoPerp):
            # 8h funding to daily rate
            funding_rate_day = funding_rate_8h * 3.0
        elif isinstance(instrument, ForexPair):
            forex_swap = instrument.swap_long_annual
        elif isinstance(instrument, OptionContract) and underlying_spot and current_time:
            greeks = instrument.greeks(underlying_spot, current_time)
            opt_price = instrument.price(underlying_spot, current_time)
            if opt_price > 0:
                theta_pct = (greeks["theta"] / 365.0) / opt_price

        return cls(
            borrow_rate_annual=borrow_rate,
            funding_rate_per_day=funding_rate_day,
            forex_swap_annual=forex_swap,
            option_theta_daily_pct=theta_pct
        )

    def calculate_holding_cost(self, weight: float, horizon_seconds: float) -> float:
        """
        Calculates fraction of portfolio lost to holding carry costs for weight w over horizon_seconds.
        """
        if abs(weight) < 1e-7:
            return 0.0

        days = horizon_seconds / 86400.0
        years = horizon_seconds / (365.0 * 86400.0)
        cost = 0.0

        # Short borrow cost
        if weight < 0:
            cost += abs(weight) * self.borrow_rate_annual * years

        # Crypto perp funding (longs pay when rate > 0)
        if self.funding_rate_per_day != 0.0:
            cost += weight * self.funding_rate_per_day * days

        # Forex swap carry
        if self.forex_swap_annual != 0.0:
            cost -= weight * self.forex_swap_annual * years

        # Option theta decay
        if self.option_theta_daily_pct != 0.0 and weight > 0:
            cost -= weight * self.option_theta_daily_pct * days

        return cost

