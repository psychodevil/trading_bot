"""
Forex Mean Reversion and Carry Strategy.
Models mean-reverting currency pairs using Student-t distribution,
optimizing overnight interest swap carry against spread crossing and turnover costs.
"""

from __future__ import annotations
from typing import List, Optional

from trading_bot.core.instruments import Instrument, ForexPair
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.core.distributions import StudentTDistribution
from trading_bot.forecast.base import DistributionEstimator
from trading_bot.forecast.parametric import StudentTEstimator
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class ForexMeanReversionStrategy(Strategy):
    """
    Forex mean-reversion trading strategy combining deviation from moving average
    with overnight interest swap capture and tight inaction bands.
    """

    def __init__(
        self,
        name: str = "ForexMeanReversion",
        horizon_seconds: float = 4 * 3600.0, # 4 hours
        reversion_speed_kappa: float = 0.5,
        lookback_window: int = 100,
        utility_config: Optional[UtilityConfig] = None,
        cost_model: Optional[TransactionCostModel] = None,
        max_leverage: float = 5.0 # Typical 5x leverage for forex
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.reversion_speed = reversion_speed_kappa
        self.lookback_window = lookback_window
        self.estimator = StudentTEstimator(lookback_window=lookback_window, default_df=5.0)
        self.utility_config = utility_config or UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=2.0
        )
        self.cost_model = cost_model or TransactionCostModel(linear_fee_rate=0.00005, bid_ask_half_spread=0.0001)
        self.max_leverage = max_leverage
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
        if len(self.bar_history) < self.lookback_window // 2:
            return None

        # Compute rolling equilibrium mean price
        recent_bars = self.bar_history[-self.lookback_window:]
        mean_price = sum(b.close for b in recent_bars) / len(recent_bars)
        price_deviation = (mean_price - bar.close) / bar.close

        # Mean reverting expected drift
        expected_drift = price_deviation * self.reversion_speed

        # Estimate background volatility
        base_dist = self.estimator.fit_predict(self.bar_history, self.horizon_seconds)

        # Combine expected mean reversion drift with fat-tailed Student-t
        forex_dist = StudentTDistribution(
            df=5.0,
            mu=expected_drift,
            sigma=base_dist.std_dev
        )

        current_w = portfolio.get_position_weight(instrument.symbol)
        holding_model = HoldingCostModel.from_instrument(instrument)

        result = self.optimizer.optimize_position(
            distribution=forex_dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=-self.max_leverage,
            max_weight=self.max_leverage,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return result

