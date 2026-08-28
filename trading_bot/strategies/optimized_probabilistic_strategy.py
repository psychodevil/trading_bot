"""
Optimized Probabilistic Multi-Asset Strategy.
"""

from __future__ import annotations
import math
from typing import List, Optional

from trading_bot.core.instruments import Instrument, AssetClass
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.forecast.advanced_probabilistic import AdvancedProbabilisticEstimator
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class OptimizedProbabilisticStrategy(Strategy):
    """
    Optimized strategy combining multi-factor return distributions,
    regime-adaptive drift forecasting, volatility targeting, and inaction bands.
    """

    def __init__(
        self,
        name: str = "OptimizedProbabilisticAlpha",
        horizon_seconds: float = 3600.0 * 8,
        target_annual_vol: float = 0.20,
        risk_aversion: float = 2.5,
        max_leverage: float = 1.0,
        allow_short: bool = True,
        trend_weight: float = 0.0035,
        mean_rev_weight: float = 0.0025,
        cost_model: Optional[TransactionCostModel] = None
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage
        self.allow_short = allow_short

        self.estimator = AdvancedProbabilisticEstimator(
            lookback_window=80,
            trend_weight=trend_weight,
            mean_rev_weight=mean_rev_weight,
            default_df=4.0
        )

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
        if len(self.bar_history) < 35:
            return None

        dist = self.estimator.fit_predict(self.bar_history, self.horizon_seconds)

        bar_dt = bar.timeframe_seconds if bar.timeframe_seconds > 0 else 3600.0
        seconds_in_year = 365.0 * 86400.0
        dist_annual_vol = dist.std_dev * math.sqrt(seconds_in_year / max(1.0, self.horizon_seconds))

        if dist_annual_vol > 1e-4:
            vol_scale = min(1.5, self.target_annual_vol / dist_annual_vol)
        else:
            vol_scale = 1.0

        effective_max_leverage = min(self.max_leverage, self.max_leverage * vol_scale)
        min_w = -effective_max_leverage if self.allow_short else 0.0
        max_w = effective_max_leverage

        current_w = portfolio.get_position_weight(instrument.symbol)
        holding_model = HoldingCostModel.from_instrument(instrument)

        opt_res = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=min_w,
            max_weight=max_w,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return opt_res
