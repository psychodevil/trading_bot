#!/usr/bin/env python3
"""
CLI Runner for Multi-Asset, Multi-Timeframe Probabilistic Trading Bot Experiments.
"""

import argparse
import os
import sys
import time
from typing import List

# Ensure trading_bot is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.core.instruments import (
    Stock, CryptoSpot, CryptoPerp, ForexPair, FuturesContract, OptionContract
)
from trading_bot.core.distributions import StudentTDistribution
from trading_bot.data.synthetic import (
    generate_geometric_brownian_motion,
    generate_merton_jump_diffusion,
    generate_heston_stochastic_vol,
    generate_ornstein_uhlenbeck,
    convert_ticks_to_quotes,
    convert_quotes_to_bars
)
from trading_bot.data.resampler import (
    SamplingScheme, MultiTimeframeResampler
)
from trading_bot.optimizer.cost_model import TransactionCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.strategies.probabilistic_trend import ProbabilisticTrendStrategy
from trading_bot.strategies.crypto_perp_funding import CryptoPerpFundingStrategy
from trading_bot.strategies.options_vol_harvest import OptionsVolHarvestStrategy
from trading_bot.strategies.forex_mean_reversion import ForexMeanReversionStrategy
from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.visualization.report_generator import ReportGenerator


def run_single_experiment(
    vehicle_type: str,
    timeframe_type: str,
    n_steps: int = 1500,
    seed: int = 42
) -> BacktestResult:
    """Executes a single backtest for a specific vehicle and timeframe sampling mode."""
    print(f"[*] Running experiment: Vehicle={vehicle_type.upper()}, Sampling={timeframe_type} ...")

    # 1. Generate Synthetic Tick/Price Path based on vehicle characteristics
    if vehicle_type == "stock":
        inst = Stock(symbol="AAPL", tick_size=0.01, lot_size=1.0, taker_fee_rate=0.0005)
        raw_ticks = generate_geometric_brownian_motion(
            s0=180.0, mu_annual=0.12, sigma_annual=0.22, dt_seconds=10.0, n_steps=n_steps * 6, seed=seed
        )
        quotes = convert_ticks_to_quotes(raw_ticks, symbol="AAPL", spread_bps=3.0)
        strategy = ProbabilisticTrendStrategy(name="Stock_Trend_StudentT", horizon_seconds=3600.0)

    elif vehicle_type == "crypto_spot":
        inst = CryptoSpot(symbol="BTC/USDT", tick_size=0.1, lot_size=0.001, taker_fee_rate=0.0006)
        raw_ticks = generate_merton_jump_diffusion(
            s0=65000.0, mu_annual=0.25, sigma_annual=0.55, jump_lambda_annual=15.0,
            jump_mean=-0.03, jump_std=0.06, dt_seconds=10.0, n_steps=n_steps * 6, seed=seed
        )
        quotes = convert_ticks_to_quotes(raw_ticks, symbol="BTC/USDT", spread_bps=5.0)
        strategy = ProbabilisticTrendStrategy(name="CryptoSpot_HeavyTail", horizon_seconds=3600.0)

    elif vehicle_type == "crypto_perp":
        inst = CryptoPerp(symbol="ETH/USDT-PERP", tick_size=0.05, lot_size=0.01, margin_requirement=0.05)
        raw_ticks = generate_merton_jump_diffusion(
            s0=3500.0, mu_annual=0.20, sigma_annual=0.60, jump_lambda_annual=20.0,
            dt_seconds=10.0, n_steps=n_steps * 6, seed=seed
        )
        quotes = convert_ticks_to_quotes(raw_ticks, symbol="ETH/USDT-PERP", spread_bps=6.0)
        strategy = CryptoPerpFundingStrategy(
            name="CryptoPerp_FundingArbitrage",
            current_funding_rate_8h=0.0004, # 4 bps per 8h
            max_leverage=2.0
        )

    elif vehicle_type == "forex":
        inst = ForexPair(symbol="EUR/USD", pip_size=0.0001, lot_size=1000.0, margin_requirement=0.02)
        raw_ticks = generate_ornstein_uhlenbeck(
            s0=1.0850, mean_target_theta=1.0850, mean_reversion_kappa=4.0, volatility_sigma=0.08,
            dt_seconds=10.0, n_steps=n_steps * 6, seed=seed
        )
        quotes = convert_ticks_to_quotes(raw_ticks, symbol="EUR/USD", spread_bps=1.0)
        strategy = ForexMeanReversionStrategy(name="Forex_OU_MeanReversion", horizon_seconds=7200.0)

    elif vehicle_type == "option":
        inst = OptionContract(
            symbol="SPY_240920_C500",
            underlying_symbol="SPY",
            strike=500.0,
            expiry_timestamp=1700000000.0 + (30 * 86400.0), # 30 days expiry
            is_call=True,
            multiplier=100.0
        )
        heston_path = generate_heston_stochastic_vol(
            s0=500.0, v0=0.04, mu_annual=0.10, kappa=2.5, theta=0.04, xi=0.4,
            dt_seconds=10.0, n_steps=n_steps * 6, seed=seed
        )
        raw_ticks = [(ts, s) for ts, s, v in heston_path]
        quotes = convert_ticks_to_quotes(raw_ticks, symbol="SPY_240920_C500", spread_bps=10.0)
        strategy = OptionsVolHarvestStrategy(name="Options_Vol_Harvest", horizon_seconds=86400.0)

    else:
        inst = FuturesContract(symbol="ES_FUT", underlying_symbol="ES", multiplier=50.0, margin_requirement=0.10)
        raw_ticks = generate_geometric_brownian_motion(
            s0=5000.0, mu_annual=0.10, sigma_annual=0.18, dt_seconds=10.0, n_steps=n_steps * 6, seed=seed
        )
        quotes = convert_ticks_to_quotes(raw_ticks, symbol="ES_FUT", spread_bps=2.0)
        strategy = ProbabilisticTrendStrategy(name="Futures_Trend", horizon_seconds=3600.0)

    # 2. Resample quotes according to the chosen timeframe / sampling scheme
    if timeframe_type == "1m":
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.FIXED_TIME, fixed_seconds=60.0)
        desc = "1-Minute Bars"
    elif timeframe_type == "5m":
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.FIXED_TIME, fixed_seconds=300.0)
        desc = "5-Minute Bars"
    elif timeframe_type == "1h":
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.FIXED_TIME, fixed_seconds=3600.0)
        desc = "1-Hour Bars"
    elif timeframe_type == "poisson_random":
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.RANDOM_POISSON, poisson_mean_seconds=120.0, seed=seed)
        desc = "Stochastic Poisson Bars (Avg 120s)"
    elif timeframe_type == "uniform_random":
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.RANDOM_UNIFORM, random_range_seconds=(30.0, 300.0), seed=seed)
        desc = "Uniform Random Intervals [30s, 300s]"
    elif timeframe_type == "dollar_bars":
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.DOLLAR_BAR, dollar_threshold=50000.0)
        desc = "Dollar Turnover Bars ($50k)"
    else:
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.FIXED_TIME, fixed_seconds=60.0)
        desc = f"Fixed {timeframe_type}"

    bars = resampler.resample_quotes(quotes)
    print(f"    Resampled {len(quotes)} quotes into {len(bars)} {desc}.")

    # 3. Run backtest simulation
    engine = BacktestEngine(initial_cash=100000.0)
    result = engine.run(
        strategy=strategy,
        instrument=inst,
        bars=bars,
        timeframe_desc=desc,
        funding_rate_per_interval=0.0004 if vehicle_type == "crypto_perp" else 0.0
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Probabilistic Trading Bot Multi-Asset Experiment Runner")
    parser.add_argument(
        "--vehicle",
        choices=["all", "stock", "crypto_spot", "crypto_perp", "forex", "futures", "option"],
        default="all",
        help="Financial vehicle type to test"
    )
    parser.add_argument(
        "--timeframe",
        choices=["all", "1m", "5m", "1h", "poisson_random", "uniform_random", "dollar_bars"],
        default="all",
        help="Timeframe sampling scheme"
    )
    parser.add_argument("--steps", type=int, default=1200, help="Number of simulation steps")
    parser.add_argument("--output-report", type=str, default="reports/experiment_summary.html", help="Path to write HTML report")
    args = parser.parse_args()

    vehicles = ["stock", "crypto_spot", "crypto_perp", "forex", "option"] if args.vehicle == "all" else [args.vehicle]
    timeframes = ["1m", "1h", "poisson_random", "dollar_bars"] if args.timeframe == "all" else [args.timeframe]

    results: List[BacktestResult] = []
    print("\n" + "=" * 80)
    print("PROBABILISTIC TRADING BOT EXPERIMENT SUITE")
    print("=" * 80)

    for v in vehicles:
        for tf in timeframes:
            res = run_single_experiment(v, tf, n_steps=args.steps)
            results.append(res)

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy / Vehicle':<30} | {'Timeframe':<24} | {'Return':<9} | {'Sharpe':<7} | {'MaxDD':<8} | {'Inaction Eff':<12}")
    print("-" * 100)

    for r in results:
        m = r.metrics
        print(f"{r.strategy_name:<30} | {r.timeframe_description:<24} | {m.total_return_pct:>+7.2f}% | {m.sharpe_ratio:>7.2f} | {m.max_drawdown_pct:>7.2f}% | {m.inaction_efficiency_pct:>10.1f}%")

    # Generate interactive HTML dashboard report
    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    ReportGenerator.generate_html_report(
        results=results,
        title="Probabilistic Trading Bot: Multi-Asset & Multi-Timeframe Benchmark",
        output_path=args.output_report
    )
    print("\n" + "=" * 80)
    print(f"[+] Interactive HTML Dashboard successfully generated: {os.path.abspath(args.output_report)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

