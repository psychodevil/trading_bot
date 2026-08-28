"""
VectorBT Integration Adapter.
Provides fast vectorized signal matrix conversion and portfolio execution with VectorBT.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import math

try:
    import vectorbt as vbt
    import pandas as pd
    import numpy as np
except ImportError:
    vbt = None
    pd = None
    np = None

from trading_bot.core.distributions import StudentTDistribution
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_model import TransactionCostModel


def run_vectorbt_backtest(
    price_df: Any, # pandas DataFrame of asset closing prices
    initial_cash: float = 100000.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0002
) -> Optional[Any]:
    """
    Executes a vectorized backtest using VectorBT given a price matrix.
    """
    if vbt is None or pd is None or np is None:
        print("[!] vectorbt/pandas not installed in this environment. Run inside Docker container to execute.")
        return None

    # Calculate indicators vectorially with VectorBT
    ema20 = vbt.MA.run(price_df, window=20, ewm=True).ma
    ema50 = vbt.MA.run(price_df, window=50, ewm=True).ma
    ema100 = vbt.MA.run(price_df, window=100, ewm=True).ma
    rsi = vbt.RSI.run(price_df, window=14).rsi

    # Generate entries / exits
    entries = (price_df > ema100 * 0.98) & (ema20 > ema50) & (rsi < 55.0)
    exits = (price_df < ema100 * 0.97) & (ema20 < ema100)

    # Build VectorBT portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=price_df,
        entries=entries,
        exits=exits,
        fees=fee_rate,
        slippage=slippage_rate,
        init_cash=initial_cash,
        freq='1h'
    )

    print("\n" + "=" * 80)
    print("VECTORBT PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 80)
    print(portfolio.stats())
    print("=" * 80 + "\n")

    return portfolio

