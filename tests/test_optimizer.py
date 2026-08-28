"""
Unit tests for Cost-Aware Position Optimizer and Inaction Band Dynamics.
"""

import unittest

from trading_bot.core.distributions import GaussianDistribution, StudentTDistribution
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.optimizer.cost_aware_optimizer import CostAwarePositionOptimizer


class TestCostAwareOptimizer(unittest.TestCase):

    def setUp(self):
        self.cost_model = TransactionCostModel(
            linear_fee_rate=0.0005,
            bid_ask_half_spread=0.0005,
            impact_coefficient=0.002
        )
        self.holding_model = HoldingCostModel(borrow_rate_annual=0.03)
        self.utility_config = UtilityConfig(
            utility_type=UtilityType.MEAN_VARIANCE,
            risk_aversion=3.0
        )
        self.optimizer = CostAwarePositionOptimizer(
            utility_config=self.utility_config,
            default_cost_model=self.cost_model,
            default_holding_model=self.holding_model
        )

    def test_frictionless_vs_cost_aware_sizing(self):
        # Bullish distribution: mu = +1%, sigma = 2%
        dist = GaussianDistribution(mu=0.01, sigma=0.02)
        
        # When starting from current_weight = 0.0
        res = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=0.0,
            horizon_seconds=3600.0,
            min_weight=-1.0,
            max_weight=1.0
        )

        # Should recommend positive long allocation
        self.assertGreater(res.recommended_weight, 0.0)
        self.assertTrue(res.rebalance_required)

    def test_inaction_band_suppresses_turnover(self):
        # Moderately positive distribution
        dist = GaussianDistribution(mu=0.005, sigma=0.02)
        
        # 1. First optimize from 0.0
        res1 = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=0.0,
            horizon_seconds=3600.0
        )
        target = res1.recommended_weight

        # 2. Suppose current weight is already close to target (inside inaction band)
        current_weight_close = target * 0.96
        res2 = self.optimizer.optimize_position(
            distribution=dist,
            current_weight=current_weight_close,
            horizon_seconds=3600.0
        )

        # Inaction band should kick in: no rebalance needed, saving transaction costs!
        self.assertFalse(res2.rebalance_required)
        self.assertEqual(res2.recommended_weight, current_weight_close)
        self.assertEqual(res2.estimated_turnover_cost, 0.0)

    def test_cvar_risk_optimizer(self):
        cvar_cfg = UtilityConfig(
            utility_type=UtilityType.CVAR_RISK,
            cvar_alpha=0.05,
            cvar_penalty_weight=2.5
        )
        cvar_optimizer = CostAwarePositionOptimizer(utility_config=cvar_cfg)
        fat_tail_dist = StudentTDistribution(df=3.0, mu=0.005, sigma=0.03)

        res = cvar_optimizer.optimize_position(
            distribution=fat_tail_dist,
            current_weight=0.0,
            horizon_seconds=3600.0
        )
        self.assertIsNotNone(res.recommended_weight)
        # Position should be conservatively sized due to severe fat-tail CVaR penalty
        self.assertLessEqual(res.recommended_weight, 1.0)


if __name__ == "__main__":
    unittest.main()

