"""
Base Strategy Interface for Probabilistic and Cost-Aware Trading Strategies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from trading_bot.core.instruments import Instrument
from trading_bot.core.events import Bar, MarketQuote, PortfolioState
from trading_bot.core.distributions import ProbDistribution
from trading_bot.optimizer.cost_aware_optimizer import OptimizationResult


class Strategy(ABC):
    """
    Abstract strategy class. Strategies generate probabilistic forecasts
    and request cost-aware position allocations.
    """

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        portfolio: PortfolioState,
        instrument: Instrument
    ) -> Optional[OptimizationResult]:
        """
        Invoked on every bar update. Returns optimization decision result if rebalancing was considered.
        """
        pass

    def on_quote(
        self,
        quote: MarketQuote,
        portfolio: PortfolioState,
        instrument: Instrument
    ) -> Optional[OptimizationResult]:
        """Optional tick-level callback."""
        return None

