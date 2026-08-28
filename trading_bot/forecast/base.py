"""
Base interface for probabilistic return distribution estimators.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import List, Sequence

from trading_bot.core.events import Bar
from trading_bot.core.distributions import ProbDistribution


class DistributionEstimator(ABC):
    """
    Abstract interface for estimators that compute the forward return distribution
    P(R_{t -> t+H} | F_t) given historical bars or returns.
    """

    @abstractmethod
    def fit_predict(self, bars: Sequence[Bar], horizon_seconds: float) -> ProbDistribution:
        """
        Estimate forward return distribution over horizon_seconds from a sequence of OHLCV bars.
        """
        pass

    def extract_log_returns(self, bars: Sequence[Bar]) -> List[float]:
        """Extract continuous log returns r_t = ln(C_t / C_{t-1}) from bars."""
        if len(bars) < 2:
            return []
        returns = []
        for i in range(1, len(bars)):
            prev_c = bars[i - 1].close
            curr_c = bars[i].close
            if prev_c > 0 and curr_c > 0:
                returns.append(math.log(curr_c / prev_c))
        return returns

