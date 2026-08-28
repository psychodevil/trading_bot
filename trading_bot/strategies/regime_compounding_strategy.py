"""
Regime-Adaptive Probabilistic Compounding Strategy.
"""

from __future__ import annotations
import math
from typing import List, Optional

from trading_bot.core.instruments import Instrument, AssetClass
from trading_bot.core.events import Bar, PortfolioState
from trading_bot.core.distributions import StudentTDistribution
from trading_bot.forecast.features import extract_market_features
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer, OptimizationResult
from trading_bot.strategies.base import Strategy


class RegimeCompoundingStrategy(Strategy):
    """
    Quantitative trend-compounding & cash-preservation strategy.
    """

    def __init__(
        self,
        name: str = "RegimeCompoundingAlpha",
        horizon_seconds: float = 3600.0 * 12,
        lookback_window: int = 80,
        risk_aversion: float = 1.8,
        target_annual_vol: float = 0.25,
        max_leverage: float = 1.2,
        cost_model: Optional[TransactionCostModel] = None
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.lookback_window = lookback_window
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage

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
        if len(self.bar_history) < 55:
            return None

        recent_bars = self.bar_history[-self.lookback_window:]
        feats = extract_market_features(recent_bars)
        if feats is None:
            return None

        closes = [b.close for b in recent_bars]
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0 and closes[i] > 0:
                returns.append(math.log(closes[i] / closes[i-1]))
        
        n = len(returns)
        mean_ret = sum(returns) / n if n > 0 else 0.0
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, n - 1) if n > 1 else 0.0004
        base_std = math.sqrt(max(1e-6, var_ret))

        bar_dt = bar.timeframe_seconds if bar.timeframe_seconds > 0 else 3600.0
        time_ratio = max(0.01, self.horizon_seconds / max(1.0, bar_dt))
        sqrt_time = math.sqrt(time_ratio)
        seconds_in_year = 365.0 * 86400.0
        annual_vol = base_std * math.sqrt(seconds_in_year / max(1.0, bar_dt))

        current_close = bar.close
        is_macro_bull = (feats.ema_fast > feats.ema_medium) and (current_close > feats.ema_slow)
        is_macro_bear = (feats.ema_fast < feats.ema_medium) and (current_close < feats.ema_slow)

        expected_drift_1step = 0.0
        min_allowed_weight = 0.0
        target_allocation = 0.0

        if is_macro_bull:
            if 35.0 <= feats.rsi <= 65.0 or feats.bb_zscore < 0.0:
                expected_drift_1step = 0.0045 * (base_std / 0.01)
                target_allocation = self.max_leverage
            elif feats.rsi > 75.0:
                expected_drift_1step = 0.0010 * (base_std / 0.01)
                target_allocation = 0.4
            else:
                expected_drift_1step = 0.0030 * (base_std / 0.01)
                target_allocation = min(1.0, self.max_leverage)

        elif is_macro_bear:
            expected_drift_1step = -0.0030 * (base_std / 0.01)
            target_allocation = 0.0
            if instrument.asset_class in (AssetClass.CRYPTO_PERP, AssetClass.FOREX):
                min_allowed_weight = -0.5
                target_allocation = -0.5
            else:
                min_allowed_weight = 0.0

        else:
            if feats.rsi < 35.0 or feats.bb_zscore < -1.5:
                expected_drift_1step = 0.0035 * (base_std / 0.01)
                target_allocation = 0.6
            elif feats.rsi > 68.0 or feats.bb_zscore > 1.5:
                expected_drift_1step = -0.0010 * (base_std / 0.01)
                target_allocation = 0.0
            else:
                expected_drift_1step = 0.0005 * (base_std / 0.01)
                target_allocation = 0.2

        mu_h = expected_drift_1step * time_ratio
        df = 4.0 if feats.regime == "HIGH_VOLATILITY" else 6.0
        sigma_h = base_std * sqrt_time * math.sqrt((df - 2.0) / df)

        dist = StudentTDistribution(df=df, mu=mu_h, sigma=sigma_h)

        vol_scale = min(1.5, max(0.4, self.target_annual_vol / max(0.05, annual_vol)))
        max_w = min(self.max_leverage, target_allocation * vol_scale)

        current_w = portfolio.get_position_weight(instrument.symbol)
        holding_model = HoldingCostModel.from_instrument(instrument)

        opt_res = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=current_w,
            horizon_seconds=self.horizon_seconds,
            min_weight=min_allowed_weight,
            max_weight=max_w,
            cost_model=self.cost_model,
            holding_model=holding_model,
            portfolio_equity=portfolio.equity
        )

        return opt_res
