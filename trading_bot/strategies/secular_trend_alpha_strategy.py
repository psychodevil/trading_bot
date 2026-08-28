"""
Secular Trend & Regime-Filtered Probabilistic Alpha Strategy.
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


class SecularTrendAlphaStrategy(Strategy):
    """
    High-performance multi-asset strategy designed to reliably outperform
    buy-and-hold benchmarks by riding macro bull regimes and avoiding bear drawdowns in cash.
    """

    def __init__(
        self,
        name: str = "SecularTrendAlpha",
        horizon_seconds: float = 3600.0 * 24,
        macro_ema_period: int = 150,
        pullback_ema_period: int = 35,
        base_leverage: float = 1.25,
        risk_aversion: float = 1.4,
        target_annual_vol: float = 0.28,
        cost_model: Optional[TransactionCostModel] = None
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.macro_period = macro_ema_period
        self.pullback_period = pullback_ema_period
        self.base_leverage = base_leverage
        self.target_annual_vol = target_annual_vol

        self.utility_config = UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=risk_aversion
        )
        self.cost_model = cost_model or TransactionCostModel()
        self.optimizer = CostAwarePositionOptimizer(
            utility_config=self.utility_config,
            default_cost_model=self.cost_model
        )
        self.bar_history: List[Bar] = []

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

        ema_macro = compute_ema(closes, self.macro_period)[-1]
        ema_pullback = compute_ema(closes, self.pullback_period)[-1]
        rsi_val = compute_rsi(closes, 14)[-1] or 50.0

        is_macro_bull = (current_close > ema_macro * 0.99) and (ema_pullback > ema_macro * 0.98)
        is_macro_bear = (current_close < ema_macro * 0.98) and (ema_pullback < ema_macro)

        returns = [math.log(closes[i] / closes[i-1]) for i in range(len(closes)-50, len(closes)) if closes[i-1] > 0]
        n_ret = len(returns)
        mean_ret = sum(returns) / n_ret if n_ret > 0 else 0.0
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, n_ret - 1) if n_ret > 1 else 0.0004
        base_std = math.sqrt(max(1e-6, var_ret))

        target_allocation = 0.0
        expected_drift_1step = 0.0

        if is_macro_bull:
            if rsi_val < 45.0:
                expected_drift_1step = 0.0065 * (base_std / 0.01)
                target_allocation = self.base_leverage
            elif rsi_val > 78.0:
                expected_drift_1step = 0.0020 * (base_std / 0.01)
                target_allocation = 0.85
            else:
                expected_drift_1step = 0.0050 * (base_std / 0.01)
                target_allocation = self.base_leverage

        elif is_macro_bear:
            expected_drift_1step = -0.0050 * (base_std / 0.01)
            target_allocation = 0.0
        else:
            if current_close > ema_macro:
                expected_drift_1step = 0.0025 * (base_std / 0.01)
                target_allocation = 0.70
            else:
                expected_drift_1step = -0.0020 * (base_std / 0.01)
                target_allocation = 0.0

        mu_h = expected_drift_1step * 24.0
        dist = StudentTDistribution(df=6.0, mu=mu_h, sigma=base_std * math.sqrt(24.0))

        current_w = portfolio.get_position_weight(instrument.symbol)
        holding_model = HoldingCostModel.from_instrument(instrument)

        opt_res = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=0.0,
            max_weight=target_allocation,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return opt_res

