"""
Concrete multi-asset strategy implementations.
"""

from trading_bot.strategies.base import Strategy
from trading_bot.strategies.probabilistic_trend import ProbabilisticTrendStrategy
from trading_bot.strategies.crypto_perp_funding import CryptoPerpFundingStrategy
from trading_bot.strategies.options_vol_harvest import OptionsVolHarvestStrategy
from trading_bot.strategies.forex_mean_reversion import ForexMeanReversionStrategy
from trading_bot.strategies.optimized_probabilistic_strategy import OptimizedProbabilisticStrategy
from trading_bot.strategies.regime_compounding_strategy import RegimeCompoundingStrategy
from trading_bot.strategies.alpha_master_strategy import AlphaMasterStrategy
from trading_bot.strategies.secular_trend_alpha_strategy import SecularTrendAlphaStrategy
from trading_bot.strategies.hysteresis_alpha_strategy import HysteresisAlphaStrategy
from trading_bot.strategies.alpha_portfolio_strategy import AlphaPortfolioStrategy

__all__ = [
    "Strategy",
    "ProbabilisticTrendStrategy",
    "CryptoPerpFundingStrategy",
    "OptionsVolHarvestStrategy",
    "ForexMeanReversionStrategy",
    "OptimizedProbabilisticStrategy",
    "RegimeCompoundingStrategy",
    "AlphaMasterStrategy",
    "SecularTrendAlphaStrategy",
    "HysteresisAlphaStrategy",
    "AlphaPortfolioStrategy"
]

