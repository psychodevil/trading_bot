"""
Regime-Switching Gaussian Mixture Distribution Estimator.
Detects market regimes (e.g., Bullish low-volatility, Choppy/Mean-reverting, High-volatility Bearish).
"""

from __future__ import annotations
import math
from typing import Sequence, List

from trading_bot.core.events import Bar
from trading_bot.core.distributions import ProbDistribution, GaussianMixtureDistribution
from trading_bot.forecast.base import DistributionEstimator


class RegimeSwitchingEstimator(DistributionEstimator):
    """
    Fits a 2-state or 3-state Gaussian Mixture Model based on volatility and momentum regimes.
    """

    def __init__(self, lookback_window: int = 150, n_regimes: int = 2):
        self.lookback_window = lookback_window
        self.n_regimes = n_regimes

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        returns = self.extract_log_returns(bars[-self.lookback_window:])
        if len(returns) < 30:
            return GaussianMixtureDistribution(
                weights=[0.7, 0.3],
                means=[0.0005, -0.001],
                sigmas=[0.01, 0.025]
            )

        n = len(returns)
        # Compute rolling short-term vs long-term volatility
        short_window = max(5, n // 5)
        recent_returns = returns[-short_window:]
        recent_mean = sum(recent_returns) / short_window
        recent_var = sum((r - recent_mean) ** 2 for r in recent_returns) / max(1, short_window - 1)
        recent_std = math.sqrt(max(1e-10, recent_var))

        overall_mean = sum(returns) / n
        overall_var = sum((r - overall_mean) ** 2 for r in returns) / max(1, n - 1)
        overall_std = math.sqrt(max(1e-10, overall_var))

        bar_dt = bars[-1].timeframe_seconds if bars else 60.0
        time_ratio = max(0.01, horizon_seconds / max(1.0, bar_dt))
        sqrt_time = math.sqrt(time_ratio)

        # Build 3 regimes:
        # Regime 1: Bullish / Calm (Positive drift, low-to-medium vol)
        # Regime 2: Neutral / Choppy (Near-zero drift, baseline vol)
        # Regime 3: High-Vol Stress / Crash (Negative drift, elevated vol)

        vol_ratio = recent_std / (overall_std + 1e-10)

        if self.n_regimes == 2:
            # 2 regimes: Normal vs High Vol
            if vol_ratio > 1.25 or recent_mean < -0.005:
                # Elevated stress probability
                w1, w2 = 0.35, 0.65
            else:
                w1, w2 = 0.80, 0.20

            m1 = max(0.0, recent_mean) * time_ratio
            s1 = overall_std * 0.8 * sqrt_time
            m2 = min(0.0, recent_mean - overall_std * 0.5) * time_ratio
            s2 = max(recent_std, overall_std * 1.6) * sqrt_time

            return GaussianMixtureDistribution(
                weights=[w1, w2],
                means=[m1, m2],
                sigmas=[s1, s2]
            )
        else:
            # 3 regimes
            w_bull = 0.50 if recent_mean > 0 else 0.25
            w_bear = 0.40 if (vol_ratio > 1.2 or recent_mean < 0) else 0.15
            w_chop = max(0.10, 1.0 - w_bull - w_bear)

            m_bull = abs(recent_mean) * 1.2 * time_ratio
            s_bull = overall_std * 0.75 * sqrt_time

            m_chop = 0.0
            s_chop = overall_std * sqrt_time

            m_bear = -abs(recent_mean) * 1.5 * time_ratio
            s_bear = overall_std * 2.0 * sqrt_time

            return GaussianMixtureDistribution(
                weights=[w_bull, w_chop, w_bear],
                means=[m_bull, m_chop, m_bear],
                sigmas=[s_bull, s_chop, s_bear]
            )

