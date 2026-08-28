"""
Cost-Aware Dynamic Position Optimizer.
Optimizes portfolio allocation by maximizing Expected Utility minus transaction,
market impact, and holding carry costs, with analytical and numerical inaction band computation.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple

from trading_bot.core.distributions import ProbDistribution
from trading_bot.core.math_utils import minimize_scalar_brent
from trading_bot.optimizer.cost_model import TransactionCostModel, HoldingCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType, compute_expected_utility


@dataclass
class OptimizationResult:
    """Detailed output from the cost-aware position optimizer."""
    recommended_weight: float
    target_weight_frictionless: float
    target_weight_cost_aware: float
    current_weight: float
    inaction_lower_bound: float
    inaction_upper_bound: float
    rebalance_required: bool
    expected_net_utility: float
    estimated_turnover_cost: float
    estimated_holding_cost: float
    expected_return: float
    volatility: float


class CostAwarePositionOptimizer:
    """
    Optimizes position weight w given return distribution P(R) and full cost structures.
    """

    def __init__(
        self,
        utility_config: Optional[UtilityConfig] = None,
        default_cost_model: Optional[TransactionCostModel] = None,
        default_holding_model: Optional[HoldingCostModel] = None
    ):
        self.utility_config = utility_config or UtilityConfig()
        self.cost_model = default_cost_model or TransactionCostModel()
        self.holding_model = default_holding_model or HoldingCostModel()

    def optimize_position(
        self,
        distribution: ProbDistribution,
        current_weight: float,
        horizon_seconds: float = 3600.0,
        min_weight: float = -1.0,
        max_weight: float = 1.0,
        cost_model: Optional[TransactionCostModel] = None,
        holding_model: Optional[HoldingCostModel] = None,
        portfolio_equity: float = 100000.0
    ) -> OptimizationResult:
        """
        Solves: max_{w in [min_w, max_w]} E[U(w*R)] - TurnoverCost(w - w_curr) - HoldingCost(w, H)
        and checks inaction region.
        """
        tx_costs = cost_model or self.cost_model
        hold_costs = holding_model or self.holding_model
        cfg = self.utility_config

        # 1. Compute frictionless optimal target (zero transaction costs)
        def neg_frictionless_utility(w: float) -> float:
            u = compute_expected_utility(distribution, w, cfg)
            h = hold_costs.calculate_holding_cost(w, horizon_seconds)
            return -(u - h)

        bracket = (min_weight, max_weight)
        w_frictionless, neg_f_val, _ = minimize_scalar_brent(neg_frictionless_utility, bracket)
        w_frictionless = max(min_weight, min(max_weight, w_frictionless))

        # 2. Objective function with transaction costs
        def neg_net_utility(w: float) -> float:
            u = compute_expected_utility(distribution, w, cfg)
            c_tx = tx_costs.calculate_turnover_cost(w - current_weight, portfolio_equity)
            c_h = hold_costs.calculate_holding_cost(w, horizon_seconds)
            return -(u - c_tx - c_h)

        w_cost_aware, neg_obj_val, _ = minimize_scalar_brent(neg_net_utility, bracket)
        w_cost_aware = max(min_weight, min(max_weight, w_cost_aware))
        best_net_utility = -neg_obj_val

        # 3. Compute baseline utility of holding current weight (zero transaction rebalancing)
        current_net_utility = (
            compute_expected_utility(distribution, current_weight, cfg) -
            hold_costs.calculate_holding_cost(current_weight, horizon_seconds)
        )

        # 4. Compute Inaction Boundaries [w_lower, w_upper]
        # Inaction boundary occurs where marginal utility gain equals marginal turnover cost
        inaction_lower, inaction_upper = self._calculate_inaction_bounds(
            distribution, w_frictionless, tx_costs, hold_costs, horizon_seconds, min_weight, max_weight
        )

        # 5. Check if rebalancing is justified
        utility_gain = best_net_utility - current_net_utility
        is_inside_inaction = (inaction_lower <= current_weight <= inaction_upper)
        
        # If current weight is within inaction zone or net utility gain <= 0, don't rebalance
        if is_inside_inaction or utility_gain <= 1e-8:
            recommended_w = current_weight
            rebalance = False
            turnover_cost = 0.0
            net_u = current_net_utility
        else:
            recommended_w = w_cost_aware
            rebalance = abs(recommended_w - current_weight) > 1e-4
            turnover_cost = tx_costs.calculate_turnover_cost(recommended_w - current_weight, portfolio_equity)
            net_u = best_net_utility

        holding_cost = hold_costs.calculate_holding_cost(recommended_w, horizon_seconds)

        return OptimizationResult(
            recommended_weight=recommended_w,
            target_weight_frictionless=w_frictionless,
            target_weight_cost_aware=w_cost_aware,
            current_weight=current_weight,
            inaction_lower_bound=inaction_lower,
            inaction_upper_bound=inaction_upper,
            rebalance_required=rebalance,
            expected_net_utility=net_u,
            estimated_turnover_cost=turnover_cost,
            estimated_holding_cost=holding_cost,
            expected_return=distribution.mean,
            volatility=distribution.std_dev
        )

    def _calculate_inaction_bounds(
        self,
        dist: ProbDistribution,
        w_frictionless: float,
        tx_costs: TransactionCostModel,
        hold_costs: HoldingCostModel,
        horizon_seconds: float,
        min_weight: float,
        max_weight: float
    ) -> Tuple[float, float]:
        """
        Calculates the inaction/no-trade band [w_low, w_high] around frictionless optimal target.
        """
        gamma = self.utility_config.risk_aversion
        var = max(1e-6, dist.variance)
        linear_friction = tx_costs.linear_fee_rate + tx_costs.bid_ask_half_spread
        half_width = linear_friction / (gamma * var + 1e-6)
        half_width = min(0.12, max(0.02, half_width))

        lower_bound = max(min_weight, w_frictionless - half_width)
        upper_bound = min(max_weight, w_frictionless + half_width)
        return lower_bound, upper_bound
