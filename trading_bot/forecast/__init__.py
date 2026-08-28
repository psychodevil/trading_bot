"""
Probabilistic forecasting estimators for financial returns.
"""

from trading_bot.forecast.base import DistributionEstimator
from trading_bot.forecast.parametric import (
    GaussianEstimator, StudentTEstimator, EWMAEstimator, GARCH11Estimator
)
from trading_bot.forecast.regime import RegimeSwitchingEstimator
from trading_bot.forecast.ensemble import EnsembleEstimator

__all__ = [
    "DistributionEstimator",
    "GaussianEstimator",
    "StudentTEstimator",
    "EWMAEstimator",
    "GARCH11Estimator",
    "RegimeSwitchingEstimator",
    "EnsembleEstimator"
]

