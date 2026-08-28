"""
Parametric return distribution estimators: Gaussian, Student-t with heavy tails,
Exponentially Weighted Moving Average (EWMA), and GARCH(1,1).
"""

from __future__ import annotations
import math
from typing import Sequence, Optional

from trading_bot.core.events import Bar
from trading_bot.core.distributions import (
    ProbDistribution, GaussianDistribution, StudentTDistribution
)
from trading_bot.forecast.base import DistributionEstimator


class GaussianEstimator(DistributionEstimator):
    """
    Standard Normal distribution estimator scaled across arbitrary forward horizons.
    """

    def __init__(self, lookback_window: int = 50, min_periods: int = 10):
        self.lookback_window = lookback_window
        self.min_periods = min_periods

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        returns = self.extract_log_returns(bars[-self.lookback_window:])
        if len(returns) < self.min_periods:
            return GaussianDistribution(mu=0.0, sigma=0.01)

        n = len(returns)
        mean_1step = sum(returns) / n
        var_1step = sum((r - mean_1step) ** 2 for r in returns) / max(1, n - 1)
        std_1step = math.sqrt(max(1e-10, var_1step))

        # Time horizon scaling factor
        bar_dt = bars[-1].timeframe_seconds if bars else 60.0
        time_ratio = max(0.01, horizon_seconds / max(1.0, bar_dt))

        mu_h = mean_1step * time_ratio
        sigma_h = std_1step * math.sqrt(time_ratio)

        return GaussianDistribution(mu=mu_h, sigma=sigma_h)


class StudentTEstimator(DistributionEstimator):
    """
    Heavy-tailed Student-t estimator fitting degrees of freedom (nu) via kurtosis matching.
    Captures extreme tail events prevalent in crypto, equities, and forex.
    """

    def __init__(self, lookback_window: int = 100, min_periods: int = 20, default_df: float = 5.0):
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.default_df = default_df

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        returns = self.extract_log_returns(bars[-self.lookback_window:])
        if len(returns) < self.min_periods:
            return StudentTDistribution(df=self.default_df, mu=0.0, sigma=0.01)

        n = len(returns)
        mean_1step = sum(returns) / n
        var_1step = sum((r - mean_1step) ** 2 for r in returns) / max(1, n - 1)
        std_1step = math.sqrt(max(1e-10, var_1step))

        # 4th standardized moment (sample excess kurtosis)
        m4 = sum((r - mean_1step) ** 4 for r in returns) / n
        excess_kurt = (m4 / (std_1step ** 4)) - 3.0 if std_1step > 1e-8 else 0.0

        if excess_kurt > 0.2:
            # Kurtosis = 6 / (nu - 4) => nu = 4 + 6 / kurtosis
            estimated_df = 4.0 + (6.0 / excess_kurt)
            df = max(2.5, min(30.0, estimated_df))
        else:
            df = 25.0 # Approaching Gaussian

        # Scale factor for Student-t variance: Var = sigma^2 * df / (df - 2)
        scale_1step = std_1step * math.sqrt((df - 2.0) / df) if df > 2.0 else std_1step

        bar_dt = bars[-1].timeframe_seconds if bars else 60.0
        time_ratio = max(0.01, horizon_seconds / max(1.0, bar_dt))

        mu_h = mean_1step * time_ratio
        scale_h = scale_1step * math.sqrt(time_ratio)

        return StudentTDistribution(df=df, mu=mu_h, sigma=scale_h)


class EWMAEstimator(DistributionEstimator):
    """
    Exponentially Weighted Moving Average (EWMA) volatility and drift estimator.
    Gives higher weight to recent market shocks (RiskMetrics style).
    """

    def __init__(self, decay_lambda: float = 0.94, lookback_window: int = 100, min_periods: int = 15, df: float = 6.0):
        self.decay_lambda = decay_lambda
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.df = df

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        returns = self.extract_log_returns(bars[-self.lookback_window:])
        if len(returns) < self.min_periods:
            return StudentTDistribution(df=self.df, mu=0.0, sigma=0.01)

        # Compute EWMA mean and variance
        weights = []
        n = len(returns)
        for i in range(n):
            w = (1.0 - self.decay_lambda) * (self.decay_lambda ** (n - 1 - i))
            weights.append(w)
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        ewma_mean = sum(w * r for w, r in zip(norm_weights, returns))
        ewma_var = sum(w * ((r - ewma_mean) ** 2) for w, r in zip(norm_weights, returns))
        ewma_std = math.sqrt(max(1e-10, ewma_var))

        scale_1step = ewma_std * math.sqrt((self.df - 2.0) / self.df)

        bar_dt = bars[-1].timeframe_seconds if bars else 60.0
        time_ratio = max(0.01, horizon_seconds / max(1.0, bar_dt))

        mu_h = ewma_mean * time_ratio
        scale_h = scale_1step * math.sqrt(time_ratio)

        return StudentTDistribution(df=self.df, mu=mu_h, sigma=scale_h)


class GARCH11Estimator(DistributionEstimator):
    """
    GARCH(1,1) dynamic conditional volatility and mean forecaster:
    sigma_{t+1}^2 = omega + alpha * eps_t^2 + beta * sigma_t^2
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.08,
        beta: float = 0.90,
        lookback_window: int = 150,
        df: float = 5.0
    ):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.lookback_window = lookback_window
        self.df = df

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        returns = self.extract_log_returns(bars[-self.lookback_window:])
        if len(returns) < 20:
            return StudentTDistribution(df=self.df, mu=0.0, sigma=0.01)

        # Filter forward variance through history
        mean_ret = sum(returns) / len(returns)
        uncond_var = self.omega / max(1e-6, 1.0 - self.alpha - self.beta)
        curr_var = uncond_var

        for r in returns:
            eps = r - mean_ret
            curr_var = self.omega + self.alpha * (eps * eps) + self.beta * curr_var

        # Multi-step term structure forecast over K steps
        bar_dt = bars[-1].timeframe_seconds if bars else 60.0
        k_steps = max(1, int(round(horizon_seconds / max(1.0, bar_dt))))

        persistence = self.alpha + self.beta
        total_forecast_var = 0.0
        forecast_v = curr_var

        for _ in range(k_steps):
            forecast_v = self.omega + persistence * forecast_v
            total_forecast_var += forecast_v

        forecast_std = math.sqrt(max(1e-10, total_forecast_var))
        scale_h = forecast_std * math.sqrt((self.df - 2.0) / self.df)
        mu_h = mean_ret * k_steps

        return StudentTDistribution(df=self.df, mu=mu_h, sigma=scale_h)

