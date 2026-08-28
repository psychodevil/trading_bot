"""
Advanced Multi-Factor Probabilistic Distribution Estimator.
"""

from __future__ import annotations
import math
from typing import Sequence, Optional, List

from trading_bot.core.events import Bar
from trading_bot.core.distributions import ProbDistribution, StudentTDistribution
from trading_bot.forecast.base import DistributionEstimator
from trading_bot.forecast.features import extract_market_features, QuantitativeMarketFeatures


class AdvancedProbabilisticEstimator(DistributionEstimator):
    """
    Bayesian multi-factor probability distribution forecaster.
    Adapts drift, volatility, and fat tails based on market regime and multi-factor signals.
    """

    def __init__(
        self,
        lookback_window: int = 80,
        trend_weight: float = 0.0035,
        mean_rev_weight: float = 0.0025,
        vol_floor: float = 0.005,
        default_df: float = 4.5
    ):
        self.lookback_window = lookback_window
        self.trend_weight = trend_weight
        self.mean_rev_weight = mean_rev_weight
        self.vol_floor = vol_floor
        self.default_df = default_df

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        recent_bars = list(bars[-self.lookback_window:])
        if len(recent_bars) < 35:
            return StudentTDistribution(df=self.default_df, mu=0.0, sigma=0.01)

        feats = extract_market_features(recent_bars)
        if feats is None:
            return StudentTDistribution(df=self.default_df, mu=0.0, sigma=0.01)

        returns = self.extract_log_returns(recent_bars)
        n = len(returns)
        if n < 10:
            return StudentTDistribution(df=self.default_df, mu=0.0, sigma=0.01)

        mean_ret = sum(returns) / n
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, n - 1)
        base_std = math.sqrt(max(self.vol_floor ** 2, var_ret))

        bar_dt = recent_bars[-1].timeframe_seconds if recent_bars[-1].timeframe_seconds > 0 else 3600.0
        time_ratio = max(0.01, horizon_seconds / max(1.0, bar_dt))
        sqrt_time = math.sqrt(time_ratio)

        norm_macd = max(-2.0, min(2.0, feats.macd_hist / (base_std * recent_bars[-1].close + 1e-6)))
        trend_component = 0.6 * feats.trend_alignment + 0.4 * (norm_macd / 2.0)

        rsi_centered = (50.0 - feats.rsi) / 25.0
        reversion_component = 0.5 * (-feats.bb_zscore / 2.0) + 0.5 * rsi_centered

        if feats.regime in ("TRENDING_BULL", "TRENDING_BEAR"):
            w_trend = 0.85
            w_rev = 0.15
        elif feats.regime == "HIGH_VOLATILITY":
            w_trend = 0.40
            w_rev = 0.60
        else:
            w_trend = 0.20
            w_rev = 0.80

        if feats.rsi > 72.0 and trend_component > 0:
            trend_component *= 0.2
        elif feats.rsi < 28.0 and trend_component < 0:
            trend_component *= 0.2

        combined_signal = w_trend * trend_component + w_rev * reversion_component
        clamped_signal = max(-1.5, min(1.5, combined_signal))

        expected_drift_1step = clamped_signal * self.trend_weight * (base_std / 0.01)
        mu_h = expected_drift_1step * time_ratio

        if feats.regime == "HIGH_VOLATILITY":
            df = 3.5
            sigma_adj = base_std * 1.2
        elif feats.adx > 30.0:
            df = 6.0
            sigma_adj = base_std * 0.95
        else:
            df = self.default_df
            sigma_adj = base_std

        scale_h = sigma_adj * sqrt_time * math.sqrt((df - 2.0) / df)

        return StudentTDistribution(df=df, mu=mu_h, sigma=scale_h)
