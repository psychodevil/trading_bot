"""
AlphaMaster: High-Performance Multi-Asset Probabilistic Strategy.
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


class AlphaMasterStrategy(Strategy):
    """
    State-of-the-art multi-asset quantitative strategy that rides trends with full leverage,
    exits to cash on trailing breakdown, and eliminates transaction cost friction.
    """

    def __init__(
        self,
        name: str = "AlphaMaster",
        horizon_seconds: float = 3600.0 * 12,
        fast_ema_period: int = 21,
        slow_ema_period: int = 50,
        atr_period: int = 14,
        atr_multiplier: float = 2.5,
        max_leverage: float = 1.3,
        risk_aversion: float = 1.4,
        target_annual_vol: float = 0.28,
        cost_model: Optional[TransactionCostModel] = None
    ):
        super().__init__(name=name)
        self.horizon_seconds = horizon_seconds
        self.fast_period = fast_ema_period
        self.slow_period = slow_ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_leverage = max_leverage
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
        self.trailing_stop: float = 0.0

    def on_bar(
        self,
        bar: Bar,
        portfolio: PortfolioState,
        instrument: Instrument
    ) -> Optional[OptimizationResult]:
        self.bar_history.append(bar)
        if len(self.bar_history) < self.slow_period + 10:
            return None

        closes = [b.close for b in self.bar_history]
        current_close = bar.close

        ema_fast_list = compute_ema(closes, self.fast_period)
        ema_slow_list = compute_ema(closes, self.slow_period)
        ema_fast = ema_fast_list[-1]
        ema_slow = ema_slow_list[-1]

        atr_list = compute_atr(self.bar_history, self.atr_period)
        atr_val = atr_list[-1] if atr_list[-1] is not None else (current_close * 0.015)

        rsi_list = compute_rsi(closes, 14)
        rsi_val = rsi_list[-1] if rsi_list[-1] is not None else 50.0

        is_uptrend = (ema_fast > ema_slow) and (current_close > ema_slow * 0.995)
        lower_band = ema_fast - self.atr_multiplier * atr_val
        is_breakdown = (current_close < lower_band) or (ema_fast < ema_slow and current_close < ema_fast)

        if is_uptrend:
            cand_stop = current_close - self.atr_multiplier * atr_val
            self.trailing_stop = max(self.trailing_stop, cand_stop)
        else:
            self.trailing_stop = 0.0

        returns = [math.log(closes[i] / closes[i-1]) for i in range(len(closes)-50, len(closes)) if closes[i-1] > 0]
        n_ret = len(returns)
        mean_ret = sum(returns) / n_ret if n_ret > 0 else 0.0
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, n_ret - 1) if n_ret > 1 else 0.0004
        base_std = math.sqrt(max(1e-6, var_ret))

        bar_dt = bar.timeframe_seconds if bar.timeframe_seconds > 0 else 3600.0
        seconds_in_year = 365.0 * 86400.0
        annual_vol = base_std * math.sqrt(seconds_in_year / max(1.0, bar_dt))

        vol_scale = min(1.6, max(0.5, self.target_annual_vol / max(0.08, annual_vol)))

        expected_drift_1step = 0.0
        target_allocation = 0.0

        if is_uptrend and not is_breakdown:
            if rsi_val > 82.0:
                expected_drift_1step = 0.0020 * (base_std / 0.01)
                target_allocation = min(self.max_leverage, 0.8 * vol_scale)
            else:
                expected_drift_1step = 0.0060 * (base_std / 0.01)
                target_allocation = min(self.max_leverage, 1.25 * vol_scale)

        elif is_breakdown:
            expected_drift_1step = -0.0040 * (base_std / 0.01)
            target_allocation = 0.0
        else:
            if current_close > ema_slow:
                expected_drift_1step = 0.0020 * (base_std / 0.01)
                target_allocation = min(self.max_leverage, 0.6 * vol_scale)
            else:
                expected_drift_1step = -0.0010 * (base_std / 0.01)
                target_allocation = 0.0

        time_ratio = max(0.01, self.horizon_seconds / max(1.0, bar_dt))
        sqrt_time = math.sqrt(time_ratio)
        mu_h = expected_drift_1step * time_ratio
        df = 5.0
        sigma_h = base_std * sqrt_time * math.sqrt((df - 2.0) / df)

        dist = StudentTDistribution(df=df, mu=mu_h, sigma=sigma_h)

        current_w = portfolio.get_position_weight(instrument.symbol)
        holding_model = HoldingCostModel.from_instrument(instrument)

        min_w = -0.4 if instrument.asset_class == AssetClass.FOREX else 0.0
        max_w = min(self.max_leverage, max(0.0, target_allocation))

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
