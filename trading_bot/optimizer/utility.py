"""
Utility and Risk Objective functions for probabilistic portfolio optimization.
Supports Mean-Variance, Fractional Kelly, CRRA (Power Utility), CARA (Exponential Utility),
and CVaR / Expected Shortfall penalty formulations.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable

from trading_bot.core.distributions import ProbDistribution


class UtilityType(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    KELLY = "kelly"
    POWER_CRRA = "power_crra"
    EXPONENTIAL_CARA = "exponential_cara"
    CVAR_RISK = "cvar_risk"


@dataclass
class UtilityConfig:
    """Configuration parameters for portfolio objective functions."""
    utility_type: UtilityType = UtilityType.MEAN_VARIANCE
    risk_aversion: float = 2.5       # Gamma in Mean-Variance and CRRA/CARA
    kelly_fraction: float = 0.5      # Half-Kelly safety multiplier
    cvar_alpha: float = 0.05         # 95% confidence level
    cvar_penalty_weight: float = 2.0 # Weight on tail loss CVaR penalty


def compute_expected_utility(
    dist: ProbDistribution,
    weight: float,
    config: UtilityConfig
) -> float:
    """
    Computes expected utility or risk-adjusted objective for position weight w given return distribution P(R).
    """
    mu = dist.mean
    var = dist.variance

    if config.utility_type == UtilityType.MEAN_VARIANCE:
        # Expected Return - 0.5 * gamma * w^2 * Var(R)
        return weight * mu - 0.5 * config.risk_aversion * (weight ** 2) * var

    elif config.utility_type == UtilityType.KELLY:
        # Fractional Kelly: w * mu - (1 / (2 * f_kelly)) * w^2 * Var
        var_safe = max(1e-8, var)
        penalty_factor = 1.0 / (2.0 * max(0.05, config.kelly_fraction))
        return weight * mu - penalty_factor * (weight ** 2) * var_safe

    elif config.utility_type == UtilityType.EXPONENTIAL_CARA:
        # CARA Utility: U(W) = -exp(-gamma * w * R) / gamma
        # For general distributions, integrate U(w*r) over density
        gamma = max(0.1, config.risk_aversion)
        def cara_fn(wr: float) -> float:
            # Clamped for numerical stability
            clamped_arg = min(20.0, max(-20.0, -gamma * wr))
            return -math.exp(clamped_arg) / gamma

        return dist.expected_utility(cara_fn, weight)

    elif config.utility_type == UtilityType.POWER_CRRA:
        # CRRA Utility: U(1 + w*R) = ((1 + w*R)^(1-gamma) - 1) / (1-gamma)
        gamma = config.risk_aversion
        def crra_fn(wr: float) -> float:
            wealth = 1.0 + wr
            if wealth <= 1e-4:
                return -100.0 # Severe bankruptcy penalty
            if abs(gamma - 1.0) < 1e-4:
                return math.log(wealth)
            return ((wealth ** (1.0 - gamma)) - 1.0) / (1.0 - gamma)

        return dist.expected_utility(crra_fn, weight)

    elif config.utility_type == UtilityType.CVAR_RISK:
        # Objective: Return - lambda * TailLoss(w)
        # For position w, loss distribution is -w * R
        cvar_loss = dist.cvar(alpha=config.cvar_alpha)
        # Tail penalty scales with position size
        position_cvar = abs(weight) * cvar_loss
        return weight * mu - config.cvar_penalty_weight * position_cvar

    else:
        return weight * mu - 0.5 * config.risk_aversion * (weight ** 2) * var

