"""
Crypto Perpetual Funding Rate and Basis Strategy.
Captures perpetual contract funding yields while hedging or managing directional risk
and factoring in transaction and slippage costs.
"""

from __future__ import annotations
from typing import List, Optional

from trading_bot.core.instruments import Instrument, CryptoPerp
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.core.distributions import StudentTDistribution
from trading_bot.forecast.base import DistributionEstimator
from trading_bot.forecast.parametric import EWMAEstimator
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class CryptoPerpFundingStrategy(Strategy):
    """
    Opportunistically harvests perpetual funding rates by factoring in expected funding cashflows
    directly into the return distribution and holding carry model.
    """

    def __init__(
        self,
        name: str = "CryptoPerpFunding",
        horizon_seconds: float = 8 * 3600.0, # 8 hours standard funding window
        current_funding_rate_8h: float = 0.0003, # 3 bps per 8h (approx 32% APR)
        estimator: Optional[DistributionEstimator] = None,
        utility_config: Optional[UtilityConfig] = None,
        cost_model: Optional[TransactionCostModel] = None,
        max_leverage: float = 2.0
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.funding_rate_8h = current_funding_rate_8h
        self.estimator = estimator or EWMAEstimator(lookback_window=100)
        self.utility_config = utility_config or UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=2.0
        )
        self.cost_model = cost_model or TransactionCostModel(linear_fee_rate=0.0004, bid_ask_half_spread=0.0001)
        self.max_leverage = max_leverage
        self.optimizer = CostAwarePositionOptimizer(
            utility_config=self.utility_config,
            default_cost_model=self.cost_model
        )
        self.bar_history: List[Bar] = []

    def set_funding_rate(self, rate_8h: float):
        """Update current market funding rate."""
        self.funding_rate_8h = rate_8h

    def on_bar(
        self,
        bar: Bar,
        portfolio: PortfolioState,
        instrument: Instrument
    ) -> Optional[OptimizationResult]:
        self.bar_history.append(bar)
        if len(self.bar_history) < 20:
            return None

        # Base market return distribution forecast
        base_dist = self.estimator.fit_predict(self.bar_history, self.horizon_seconds)

        # Expected funding yield over horizon
        # For long position: yield is -funding_rate_8h (cost)
        # For short position: yield is +funding_rate_8h (credit)
        expected_funding_yield = self.funding_rate_8h

        # Adjust distribution mean for net carry
        # If funding is very high positive, bias return downwards to encourage shorting / neutral basis
        adjusted_mu = base_dist.mean - expected_funding_yield
        adjusted_dist = StudentTDistribution(
            df=5.0,
            mu=adjusted_mu,
            sigma=base_dist.std_dev
        )

        current_w = portfolio.get_position_weight(instrument.symbol)
        holding_model = HoldingCostModel.from_instrument(instrument, funding_rate_8h=self.funding_rate_8h)

        result = self.optimizer.optimize_position(
            distribution=adjusted_dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=-self.max_leverage,
            max_weight=self.max_leverage,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return result

