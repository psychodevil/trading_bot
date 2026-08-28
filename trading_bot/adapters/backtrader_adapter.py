"""
Backtrader Integration Adapter.
Enables running the Bayesian Probability Distribution Engine and Cost-Aware Optimizer
inside the industry-standard Backtrader (bt.Cerebro) ecosystem.
"""

from __future__ import annotations
import math
from typing import Optional, Dict, Any

try:
    import backtrader as bt
except ImportError:
    # Graceful fallback mock if backtrader is not installed in local environment
    bt = None

from trading_bot.core.distributions import StudentTDistribution
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer


if bt is not None:
    class BacktraderProbabilisticStrategy(bt.Strategy):
        """
        Backtrader strategy wrapper that connects Backtrader feeds with our
        cost-aware probabilistic rebalancing engine.
        """
        params = (
            ('risk_aversion', 1.2),
            ('horizon_hours', 24.0),
            ('max_leverage', 1.25),
            ('linear_fee_rate', 0.0005),
            ('bid_ask_half_spread', 0.0002),
        )

        def __init__(self):
            self.optimizer = CostAwarePositionOptimizer(
                utility_config=UtilityConfig(utility_type=UtilityType.MEAN_VARIANCE, risk_aversion=self.params.risk_aversion),
                default_cost_model=TransactionCostModel(
                    linear_fee_rate=self.params.linear_fee_rate,
                    bid_ask_half_spread=self.params.bid_ask_half_spread
                )
            )
            # Standard Backtrader Indicators
            self.ema20 = {d: bt.indicators.EMA(d.close, period=20) for d in self.datas}
            self.ema50 = {d: bt.indicators.EMA(d.close, period=50) for d in self.datas}
            self.ema100 = {d: bt.indicators.EMA(d.close, period=100) for d in self.datas}
            self.rsi = {d: bt.indicators.RSI(d.close, period=14) for d in self.datas}
            self.atr = {d: bt.indicators.ATR(d, period=14) for d in self.datas}

        def next(self):
            total_value = self.broker.getvalue()
            if total_value <= 0:
                return

            for data in self.datas:
                sym = data._name
                pos = self.getposition(data)
                curr_px = data.close[0]
                curr_w = (pos.size * curr_px) / total_value

                ema20_val = self.ema20[data][0]
                ema50_val = self.ema50[data][0]
                ema100_val = self.ema100[data][0]
                rsi_val = self.rsi[data][0]
                atr_val = self.atr[data][0]

                # Probabilistic Regime Estimation
                lower_keltner = ema100_val - 3.5 * atr_val
                is_bull = (curr_px > lower_keltner) or (ema20_val > ema50_val)
                is_bear = (curr_px < lower_keltner) and (ema20_val < ema100_val)

                if is_bull and not is_bear:
                    target_w = 0.20 if rsi_val < 48.0 else 0.15
                    drift = 0.0080
                elif is_bear:
                    target_w = 0.0 # Cash Defense
                    drift = -0.0060
                else:
                    target_w = 0.10
                    drift = 0.0020

                # Formulate Probability Distribution
                dist = StudentTDistribution(df=5.5, mu=drift * self.params.horizon_hours, sigma=0.02 * math.sqrt(self.params.horizon_hours))
                
                # Inaction Band Check
                opt_res = self.optimizer.optimize_position(
                    distribution=dist,
                    current_weight=curr_w,
                    horizon_seconds=self.params.horizon_hours * 3600.0,
                    min_weight=0.0,
                    max_weight=target_w,
                    portfolio_equity=total_value
                )

                if opt_res.rebalance_required:
                    target_size = (opt_res.recommended_weight * total_value) / curr_px
                    self.order_target_size(data=data, target=target_size)

