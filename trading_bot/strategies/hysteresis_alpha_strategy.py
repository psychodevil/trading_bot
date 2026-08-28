"""
Hysteresis Multi-Horizon Probabilistic Alpha Strategy.
"""

from __future__ import annotations
import math
from typing import List, Optional

from trading_bot.core.instruments import Instrument, AssetClass
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.core.distributions import StudentTDistribution
from trading_bot.forecast.features import compute_ema, compute_atr, compute_rsi
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class HysteresisAlphaStrategy(Strategy):
    """
    Hysteresis alpha strategy with wide exit buffers to avoid whipsaws.
    """

    def __init__(
        self,
        name: str = "HysteresisAlpha",
        fast_ema_period: int = 24,
        slow_ema_period: int = 120,
        macro_ema_period: int = 360,
        atr_multiplier: float = 2.0,
        base_leverage: float = 1.30,
        target_annual_vol: float = 0.28,
        cost_model: Optional[TransactionCostModel] = None
    ):
        super().__init__(name=name)
        self.fast_period = fast_ema_period
        self.slow_period = slow_ema_period
        self.macro_period = macro_ema_period
        self.atr_multiplier = atr_multiplier
        self.base_leverage = base_leverage
        self.target_annual_vol = target_annual_vol

        self.utility_config = UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=1.5
        )
        self.cost_model = cost_model or TransactionCostModel()
        self.optimizer = CostAwarePositionOptimizer(
            utility_config=self.utility_config,
            default_cost_model=self.cost_model
        )
        self.bar_history: List[Bar] = []
        self.bars_in_position: int = 0
        self.in_long_trade: bool = False

    def on_bar(
        self,
        bar: Bar,
        portfolio: PortfolioState,
        instrument: Instrument
    ) -> Optional[OptimizationResult]:
        self.bar_history.append(bar)
        if len(self.bar_history) < self.macro_period + 10:
            return None

        closes = [b.close for b in self.bar_history]
        current_close = bar.close

        ema_fast = compute_ema(closes, self.fast_period)[-1]
        ema_slow = compute_ema(closes, self.slow_period)[-1]
        ema_macro = compute_ema(closes, self.macro_period)[-1]

        atr_val = compute_atr(self.bar_history, 14)[-1] or (current_close * 0.015)
        rsi_val = compute_rsi(closes, 14)[-1] or 50.0

        current_w = portfolio.get_position_weight(instrument.symbol)
        if current_w > 0.1:
            self.bars_in_position += 1
            self.in_long_trade = True
        else:
            self.bars_in_position = 0
            self.in_long_trade = False

        bull_entry = (ema_fast > ema_slow) and (current_close > ema_macro * 0.99)
        bear_exit = (ema_fast < ema_slow * 0.995) or (current_close < ema_slow - self.atr_multiplier * atr_val)

        target_allocation = 0.0
        if not self.in_long_trade:
            if bull_entry:
                target_allocation = self.base_leverage
            else:
                target_allocation = 0.0
        else:
            if bear_exit and self.bars_in_position >= 6:
                target_allocation = 0.0
            else:
                if rsi_val > 80.0:
                    target_allocation = 0.9
                else:
                    target_allocation = self.base_leverage

        returns = [math.log(closes[i] / closes[i-1]) for i in range(len(closes)-50, len(closes)) if closes[i-1] > 0]
        n_ret = len(returns)
        mean_ret = sum(returns) / n_ret if n_ret > 0 else 0.0
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, n_ret - 1) if n_ret > 1 else 0.0004
        base_std = math.sqrt(max(1e-6, var_ret))

        expected_drift = 0.0050 * (base_std / 0.01) if target_allocation > 0 else -0.0030 * (base_std / 0.01)
        mu_h = expected_drift * 24.0
        sigma_h = base_std * math.sqrt(24.0)

        dist = StudentTDistribution(df=5.0, mu=mu_h, sigma=sigma_h)
        holding_model = HoldingCostModel.from_instrument(instrument)

        opt_res = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=current_w,
            horizon_seconds=3600.0 * 24,
            min_weight=0.0,
            max_weight=target_allocation,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return opt_res

