"""
Event-Driven Multi-Asset, Multi-Timeframe Backtesting Engine.
Executes strategies, tracks portfolio states, applies transaction/holding costs,
and computes quantitative performance reports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from trading_bot.core.instruments import Instrument
from trading_bot.core.events import Bar, MarketQuote, Fill, Position, PortfolioState
from trading_bot.execution.simulated_broker import SimulatedBroker
from trading_bot.strategies.base import Strategy
from trading_bot.backtest.metrics import BacktestPerformanceMetrics, compute_performance_metrics


@dataclass
class BacktestResult:
    """Complete results package containing time series logs and performance metrics."""
    strategy_name: str
    instrument: Instrument
    timeframe_description: str
    equity_curve: List[Tuple[float, float]]
    price_history: List[Tuple[float, float]]
    weights_history: List[Tuple[float, float]]
    inaction_bands_history: List[Tuple[float, float, float]] # (ts, lower, upper)
    frictionless_targets_history: List[Tuple[float, float]]
    forecasts_history: List[Tuple[float, float, float]]      # (ts, mean, std)
    metrics: BacktestPerformanceMetrics
    fills: List[Fill]
    total_decisions: int
    rebalances_executed: int


class BacktestEngine:
    """
    Backtesting engine simulating historical execution step-by-step.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        risk_free_rate: float = 0.03
    ):
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.broker = SimulatedBroker(initial_cash=initial_cash)

    def run(
        self,
        strategy: Strategy,
        instrument: Instrument,
        bars: List[Bar],
        timeframe_desc: str = "1m",
        funding_rate_per_interval: float = 0.0001
    ) -> BacktestResult:
        """
        Runs backtest simulation on a sequence of historical or synthetic bars.
        """
        if not bars:
            raise ValueError("No bars provided for backtest")

        # Reset broker state
        self.broker = SimulatedBroker(initial_cash=self.initial_cash)
        self.broker.register_instrument(instrument)

        equity_curve: List[Tuple[float, float]] = []
        price_history: List[Tuple[float, float]] = []
        weights_history: List[Tuple[float, float]] = []
        inaction_bands: List[Tuple[float, float, float]] = []
        frictionless_targets: List[Tuple[float, float]] = []
        forecasts: List[Tuple[float, float, float]] = []

        total_decisions = 0
        rebalances_executed = 0

        for bar in bars:
            # 1. Update broker with incoming bar
            self.broker.on_bar(bar, funding_rate_per_interval=funding_rate_per_interval)
            current_portfolio = self.broker.get_portfolio_state()

            # 2. Strategy evaluation
            opt_result = strategy.on_bar(bar, current_portfolio, instrument)

            if opt_result is not None:
                total_decisions += 1
                frictionless_targets.append((bar.timestamp, opt_result.target_weight_frictionless))
                inaction_bands.append((bar.timestamp, opt_result.inaction_lower_bound, opt_result.inaction_upper_bound))
                forecasts.append((bar.timestamp, opt_result.expected_return, opt_result.volatility))

                if opt_result.rebalance_required:
                    # Execute target weight rebalance
                    fill = self.broker.execute_target_weight(instrument.symbol, opt_result.recommended_weight)
                    if fill is not None:
                        rebalances_executed += 1

            # 3. Record snapshot
            post_port = self.broker.get_portfolio_state()
            equity_curve.append((bar.timestamp, post_port.equity))
            price_history.append((bar.timestamp, bar.close))
            weights_history.append((bar.timestamp, post_port.get_position_weight(instrument.symbol)))

        # 4. Calculate metrics
        metrics = compute_performance_metrics(
            equity_curve=equity_curve,
            fills=self.broker.fill_history,
            total_decisions=total_decisions,
            rebalances_executed=rebalances_executed,
            risk_free_rate=self.risk_free_rate
        )

        return BacktestResult(
            strategy_name=strategy.name,
            instrument=instrument,
            timeframe_description=timeframe_desc,
            equity_curve=equity_curve,
            price_history=price_history,
            weights_history=weights_history,
            inaction_bands_history=inaction_bands,
            frictionless_targets_history=frictionless_targets,
            forecasts_history=forecasts,
            metrics=metrics,
            fills=self.broker.fill_history,
            total_decisions=total_decisions,
            rebalances_executed=rebalances_executed
        )

