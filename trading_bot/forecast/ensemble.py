"""
Ensemble Probabilistic Estimator combining multiple forecasting models.
"""

from __future__ import annotations
from typing import Sequence, List, Tuple

from trading_bot.core.events import Bar
from trading_bot.core.distributions import ProbDistribution, EmpiricalSampleDistribution
from trading_bot.forecast.base import DistributionEstimator


class EnsembleEstimator(DistributionEstimator):
    """
    Blends multiple distribution estimators by drawing Monte Carlo mixtures.
    """

    def __init__(self, estimators: Sequence[Tuple[DistributionEstimator, float]], n_monte_carlo: int = 2000):
        self.estimators = list(estimators)
        total_w = sum(w for _, w in self.estimators)
        if total_w <= 0:
            raise ValueError("Total ensemble weights must be positive")
        self.norm_estimators = [(est, w / total_w) for est, w in self.estimators]
        self.n_monte_carlo = n_monte_carlo

    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        all_samples: List[float] = []
        for est, weight in self.norm_estimators:
            dist = est.fit_predict(bars, horizon_seconds)
            n_sub = max(10, int(round(self.n_monte_carlo * weight)))
            samples = dist.sample(n_sub)
            all_samples.extend(samples)

        return EmpiricalSampleDistribution(all_samples)

