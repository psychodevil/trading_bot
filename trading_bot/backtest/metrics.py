"""
Quantitative Performance and Risk Analytics.
Computes CAGR, Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR, Turnover, Fee Drag, and Inaction Efficiency.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple

from trading_bot.core.events import Fill


@dataclass
class BacktestPerformanceMetrics:
    """Comprehensive performance metrics summary."""
    initial_equity: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    total_turnover_dollars: float
    total_fees_paid_dollars: float
    fee_drag_pct: float
    inaction_efficiency_pct: float
    var_95_daily_pct: float
    cvar_95_daily_pct: float


def compute_performance_metrics(
    equity_curve: List[Tuple[float, float]],
    fills: List[Fill],
    total_decisions: int = 1,
    rebalances_executed: int = 0,
    risk_free_rate: float = 0.03
) -> BacktestPerformanceMetrics:
    """
    Computes all standard and cost-drag quant metrics from (timestamp, equity) time series and fill records.
    """
    if not equity_curve:
        raise ValueError("Equity curve is empty")

    init_eq = equity_curve[0][1]
    final_eq = equity_curve[-1][1]
    total_ret = (final_eq - init_eq) / init_eq if init_eq > 0 else 0.0

    duration_seconds = max(1.0, equity_curve[-1][0] - equity_curve[0][0])
    duration_years = duration_seconds / (365.0 * 86400.0)
    duration_days = duration_seconds / 86400.0

    # Annualized CAGR
    cagr = ((final_eq / init_eq) ** (1.0 / duration_years) - 1.0) if (init_eq > 0 and final_eq > 0 and duration_years > 0.01) else total_ret

    # Step returns
    step_returns: List[float] = []
    for i in range(1, len(equity_curve)):
        prev_eq = equity_curve[i - 1][1]
        curr_eq = equity_curve[i][1]
        if prev_eq > 0:
            step_returns.append((curr_eq - prev_eq) / prev_eq)

    if not step_returns:
        step_returns = [0.0]

    # Frequency annualization
    n_steps = len(step_returns)
    steps_per_year = n_steps / max(1e-4, duration_years)

    mean_ret = sum(step_returns) / n_steps
    var_ret = sum((r - mean_ret) ** 2 for r in step_returns) / max(1, n_steps - 1)
    std_ret = math.sqrt(max(1e-12, var_ret))

    ann_vol = std_ret * math.sqrt(steps_per_year)

    # Downside volatility for Sortino
    downside_var = sum(min(0.0, r) ** 2 for r in step_returns) / max(1, n_steps - 1)
    downside_std = math.sqrt(max(1e-12, downside_var))
    ann_downside_vol = downside_std * math.sqrt(steps_per_year)

    # Sharpe & Sortino
    ann_return = mean_ret * steps_per_year
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-6 else 0.0
    sortino = (ann_return - risk_free_rate) / ann_downside_vol if ann_downside_vol > 1e-6 else 0.0

    # Drawdowns
    peak = init_eq
    max_dd = 0.0
    max_dd_duration = 0.0
    current_dd_start = equity_curve[0][0]

    for ts, eq in equity_curve:
        if eq > peak:
            peak = eq
            current_dd_start = ts
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = (ts - current_dd_start) / 86400.0

    calmar = (cagr / max_dd) if max_dd > 1e-6 else (cagr if cagr > 0 else 0.0)

    # Trade statistics & win rate
    total_fees = sum(f.fee for f in fills)
    total_turnover = sum(f.quantity * f.price for f in fills)
    fee_drag = (total_fees / init_eq) * 100.0 if init_eq > 0 else 0.0

    # Trade win rate from fills (grouped by consecutive roundtrips)
    winning_trades = 0
    losing_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(1, len(fills)):
        # Approx trade PnL
        f_prev = fills[i - 1]
        f_curr = fills[i]
        if f_prev.side != f_curr.side:
            pnl = (f_curr.price - f_prev.price) * f_curr.quantity if f_prev.side.value == "buy" else (f_prev.price - f_curr.price) * f_curr.quantity
            if pnl > 0:
                winning_trades += 1
                gross_profit += pnl
            else:
                losing_trades += 1
                gross_loss += abs(pnl)

    total_completed = winning_trades + losing_trades
    win_rate = (winning_trades / total_completed * 100.0) if total_completed > 0 else 50.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 1.0)

    # Inaction efficiency ratio (percentage of evaluation steps where unnecessary rebalance was skipped)
    avoided_rebalances = max(0, total_decisions - rebalances_executed)
    inaction_eff = (avoided_rebalances / total_decisions * 100.0) if total_decisions > 0 else 0.0

    # Daily VaR and CVaR
    sorted_returns = sorted(step_returns)
    alpha_idx = int(0.05 * len(sorted_returns))
    var_95 = -sorted_returns[alpha_idx] if alpha_idx < len(sorted_returns) else 0.0
    tail_losses = sorted_returns[:max(1, alpha_idx)]
    cvar_95 = -sum(tail_losses) / len(tail_losses) if tail_losses else var_95

    return BacktestPerformanceMetrics(
        initial_equity=init_eq,
        final_equity=final_eq,
        total_return_pct=total_ret * 100.0,
        cagr_pct=cagr * 100.0,
        annualized_volatility_pct=ann_vol * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd * 100.0,
        max_drawdown_duration_days=max_dd_duration,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        total_trades=len(fills),
        total_turnover_dollars=total_turnover,
        total_fees_paid_dollars=total_fees,
        fee_drag_pct=fee_drag,
        inaction_efficiency_pct=inaction_eff,
        var_95_daily_pct=var_95 * 100.0,
        cvar_95_daily_pct=cvar_95 * 100.0
    )

