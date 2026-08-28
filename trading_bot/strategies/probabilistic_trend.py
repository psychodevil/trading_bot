"""
Probabilistic Trend Following and Momentum Strategy for Stocks and Crypto.
Estimates heavy-tailed Student-t return distribution over forward horizon H,
and optimizes target weight under transaction cost friction and inaction bands.
"""

from __future__ import annotations
from typing import List, Optional

from trading_bot.core.instruments import Instrument
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.forecast.base import DistributionEstimator
from trading_bot.forecast.parametric import StudentTEstimator
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class ProbabilisticTrendStrategy(Strategy):
    """
    Combines rolling momentum and fat-tailed Student-t distribution forecasting
    with cost-aware dynamic position sizing.
    """

    def __init__(
        self,
        name: str = "ProbabilisticTrend",
        horizon_seconds: float = 3600.0, # 1 hour forward forecast
        estimator: Optional[DistributionEstimator] = None,
        utility_config: Optional[UtilityConfig] = None,
        cost_model: Optional[TransactionCostModel] = None,
        max_leverage: float = 1.0,
        allow_short: bool = True
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.estimator = estimator or StudentTEstimator(lookback_window=80, default_df=4.5)
        self.utility_config = utility_config or UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=3.0
        )
        self.cost_model = cost_model or TransactionCostModel()
        self.max_leverage = max_leverage
        self.allow_short = allow_short
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
        if len(self.bar_history) < 20:
            return None

        # 1. Forecast forward return distribution P(R_{t -> t+H})
        dist = self.estimator.fit_predict(self.bar_history, self.horizon_seconds)

        # 2. Get current portfolio weight
        current_w = portfolio.get_position_weight(instrument.symbol)

        # 3. Construct holding cost model from instrument specs
        holding_model = HoldingCostModel.from_instrument(instrument)

        min_w = -self.max_leverage if self.allow_short else 0.0
        max_w = self.max_leverage

        # 4. Optimize position under costs & compute inaction region
        result = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=min_w,
            max_weight=max_w,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return result

