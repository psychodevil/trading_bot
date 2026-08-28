"""
Options Volatility Harvesting and Delta-Neutral Strategy.
Trades options when implied volatility diverges from forecasted return distribution volatility,
accounting for Black-Scholes Greeks, Theta time-decay, and rebalancing transaction costs.
"""

from __future__ import annotations
import math
from typing import List, Optional

from trading_bot.core.instruments import Instrument, OptionContract
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.core.distributions import GaussianDistribution
from trading_bot.forecast.base import DistributionEstimator
from trading_bot.forecast.parametric import GARCH11Estimator
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class OptionsVolHarvestStrategy(Strategy):
    """
    Capitalizes on volatility mispricing by trading option contracts
    with continuous Greeks monitoring and Theta decay cost optimization.
    """

    def __init__(
        self,
        name: str = "OptionsVolHarvest",
        horizon_seconds: float = 86400.0, # 1 day horizon
        estimator: Optional[DistributionEstimator] = None,
        utility_config: Optional[UtilityConfig] = None,
        cost_model: Optional[TransactionCostModel] = None,
        max_leverage: float = 0.5
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.estimator = estimator or GARCH11Estimator()
        self.utility_config = utility_config or UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=4.0
        )
        self.cost_model = cost_model or TransactionCostModel(linear_fee_rate=0.0010, bid_ask_half_spread=0.0015)
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
        if len(self.bar_history) < 25:
            return None

        # Forecast underlying asset distribution
        underlying_dist = self.estimator.fit_predict(self.bar_history, self.horizon_seconds)

        # Annualized forecasted volatility
        bar_dt = bar.timeframe_seconds if bar.timeframe_seconds > 0 else 60.0
        seconds_in_year = 365.0 * 86400.0
        forecast_annual_vol = underlying_dist.std_dev * math.sqrt(seconds_in_year / max(1.0, self.horizon_seconds))

        if isinstance(instrument, OptionContract):
            # Calculate option pricing & Greeks
            spot = bar.close
            greeks = instrument.greeks(spot, bar.timestamp, volatility=forecast_annual_vol)
            opt_price = instrument.price(spot, bar.timestamp, volatility=forecast_annual_vol)
            
            # Map underlying return distribution to option return distribution using Delta & Gamma
            # R_opt approx Delta * (S / P_opt) * R_underlying + 0.5 * Gamma * (S^2 / P_opt) * R_underlying^2
            opt_leverage = (greeks["delta"] * spot / max(0.01, opt_price))
            expected_opt_return = opt_leverage * underlying_dist.mean
            opt_return_vol = abs(opt_leverage) * underlying_dist.std_dev

            opt_dist = GaussianDistribution(mu=expected_opt_return, sigma=max(0.01, opt_return_vol))
            holding_model = HoldingCostModel.from_instrument(instrument, underlying_spot=spot, current_time=bar.timestamp)
        else:
            opt_dist = underlying_dist
            holding_model = HoldingCostModel.from_instrument(instrument)

        current_w = portfolio.get_position_weight(instrument.symbol)

        result = self.optimizer.optimize_position(
            distribution=opt_dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=-self.max_leverage,
            max_weight=self.max_leverage,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return result

