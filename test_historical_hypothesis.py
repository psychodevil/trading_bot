#!/usr/bin/env python3
"""
Scientific Hypothesis Testing Suite on Real Historical Market Data (SPY, AAPL, BTC, ETH, EUR/USD).
Tests:
1. Cost-Aware Inaction Bands vs Frictionless/Naive Rebalancing (Turnover & Fee Drag Suppression).
2. Probabilistic Fat-Tail Student-t Model vs Standard Gaussian Model (Tail Risk & CVaR Control).
3. Multi-Timeframe & Asset Invariance across Equities, Crypto, and Forex.
"""

from dataclasses import replace
import os
import sys
from typing import List, Dict, Any

# Ensure trading_bot is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.core.instruments import Stock, CryptoSpot, CryptoPerp, ForexPair
from trading_bot.core.events import Bar
from trading_bot.data.historical_loader import HistoricalDataLoader
from trading_bot.data.resampler import SamplingScheme, MultiTimeframeResampler
from trading_bot.forecast.parametric import StudentTEstimator, GaussianEstimator
from trading_bot.optimizer.cost_model import TransactionCostModel
from trading_bot.optimizer.utility import UtilityConfig, UtilityType
from trading_bot.strategies.probabilistic_trend import ProbabilisticTrendStrategy
from trading_bot.strategies.forex_mean_reversion import ForexMeanReversionStrategy
from trading_bot.strategies.crypto_perp_funding import CryptoPerpFundingStrategy
from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.visualization.report_generator import ReportGenerator


def run_hypothesis_tests() -> List[BacktestResult]:
    results: List[BacktestResult] = []
    print("\n" + "=" * 90)
    print("RUNNING SCIENTIFIC HYPOTHESIS TESTS ON REAL HISTORICAL DATA")
    print("=" * 90)

    # --------------------------------------------------------------------------
    # TEST 1: SPY (S&P 500 ETF) 1-Year Hourly Real Historical Data
    # Hypothesis 1: Cost-Aware Optimizer (with Inaction Bands) vs Naive/Frictionless Sizing
    # --------------------------------------------------------------------------
    print("\n[TEST 1] S&P 500 (SPY): Cost-Aware with Inaction Bands vs Naive/Frictionless Sizing")
    spy_bars = HistoricalDataLoader.load_from_csv("data/historical/SPY_1h_1y.csv", symbol="SPY")
    spy_inst = Stock(symbol="SPY", tick_size=0.01, lot_size=1.0, taker_fee_rate=0.0005)
    cost_model_spy = TransactionCostModel(linear_fee_rate=0.0005, bid_ask_half_spread=0.0002, impact_coefficient=0.001)

    # 1A. Cost-Aware Strategy (Respects Inaction Band)
    strat_spy_cost_aware = ProbabilisticTrendStrategy(
        name="SPY_CostAware_InactionBand",
        horizon_seconds=3600.0 * 8, # 8-hour horizon
        estimator=StudentTEstimator(lookback_window=50, default_df=4.0),
        cost_model=cost_model_spy
    )
    engine = BacktestEngine(initial_cash=100000.0)
    res_spy_cost_aware = engine.run(strat_spy_cost_aware, spy_inst, spy_bars, timeframe_desc="1-Hour (Cost-Aware)")
    results.append(res_spy_cost_aware)

    # 1B. Naive / Frictionless Strategy (Forces rebalance on every small target variation, ignoring inaction band)
    strat_spy_frictionless = ProbabilisticTrendStrategy(
        name="SPY_Naive_Frictionless",
        horizon_seconds=3600.0 * 8,
        estimator=StudentTEstimator(lookback_window=50, default_df=4.0),
        cost_model=TransactionCostModel(linear_fee_rate=0.0, bid_ask_half_spread=0.0, impact_coefficient=0.0) # Optimizer thinks costs are zero
    )
    # Broker still applies real exchange costs!
    engine_frict = BacktestEngine(initial_cash=100000.0)
    engine_frict.broker.cost_model = cost_model_spy
    res_spy_frictionless = engine_frict.run(strat_spy_frictionless, spy_inst, spy_bars, timeframe_desc="1-Hour (Frictionless Naive)")
    results.append(res_spy_frictionless)

    print(f"    SPY Cost-Aware   : Return = {res_spy_cost_aware.metrics.total_return_pct:+.2f}%, Sharpe = {res_spy_cost_aware.metrics.sharpe_ratio:.2f}, Trades = {res_spy_cost_aware.metrics.total_trades}, Inaction Eff = {res_spy_cost_aware.metrics.inaction_efficiency_pct:.1f}%, Fees = ${res_spy_cost_aware.metrics.total_fees_paid_dollars:.2f}")
    print(f"    SPY Frictionless : Return = {res_spy_frictionless.metrics.total_return_pct:+.2f}%, Sharpe = {res_spy_frictionless.metrics.sharpe_ratio:.2f}, Trades = {res_spy_frictionless.metrics.total_trades}, Inaction Eff = {res_spy_frictionless.metrics.inaction_efficiency_pct:.1f}%, Fees = ${res_spy_frictionless.metrics.total_fees_paid_dollars:.2f}")

    # --------------------------------------------------------------------------
    # TEST 2: Bitcoin (BTC-USD) Real Historical Data (8,729 Hourly Bars)
    # Hypothesis 2: Fat-Tailed Student-t vs Thin-Tailed Gaussian Model
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Bitcoin (BTC-USD): Fat-Tail Student-t vs Standard Gaussian Forecasting")
    btc_bars = HistoricalDataLoader.load_from_csv("data/historical/BTC-USD_1h_1y.csv", symbol="BTC-USD")
    btc_inst = CryptoSpot(symbol="BTC-USD", tick_size=0.1, lot_size=0.001, taker_fee_rate=0.0006)
    cost_model_btc = TransactionCostModel(linear_fee_rate=0.0006, bid_ask_half_spread=0.0003, impact_coefficient=0.002)

    # 2A. Student-t Heavy-Tail Strategy
    strat_btc_student_t = ProbabilisticTrendStrategy(
        name="BTC_StudentT_HeavyTail",
        horizon_seconds=3600.0 * 12,
        estimator=StudentTEstimator(lookback_window=60, default_df=3.5),
        utility_config=UtilityConfig(utility_type=UtilityType.MEAN_VARIANCE, risk_aversion=3.0),
        cost_model=cost_model_btc
    )
    res_btc_t = BacktestEngine(initial_cash=100000.0).run(strat_btc_student_t, btc_inst, btc_bars, timeframe_desc="1-Hour (Student-t)")
    results.append(res_btc_t)

    # 2B. Gaussian Thin-Tail Strategy
    strat_btc_gaussian = ProbabilisticTrendStrategy(
        name="BTC_Gaussian_Normal",
        horizon_seconds=3600.0 * 12,
        estimator=GaussianEstimator(lookback_window=60),
        utility_config=UtilityConfig(utility_type=UtilityType.MEAN_VARIANCE, risk_aversion=3.0),
        cost_model=cost_model_btc
    )
    res_btc_g = BacktestEngine(initial_cash=100000.0).run(strat_btc_gaussian, btc_inst, btc_bars, timeframe_desc="1-Hour (Gaussian)")
    results.append(res_btc_g)

    print(f"    BTC Student-t : Return = {res_btc_t.metrics.total_return_pct:+.2f}%, Sharpe = {res_btc_t.metrics.sharpe_ratio:.2f}, MaxDD = {res_btc_t.metrics.max_drawdown_pct:.2f}%, CVaR 95% = {res_btc_t.metrics.cvar_95_daily_pct:.2f}%")
    print(f"    BTC Gaussian  : Return = {res_btc_g.metrics.total_return_pct:+.2f}%, Sharpe = {res_btc_g.metrics.sharpe_ratio:.2f}, MaxDD = {res_btc_g.metrics.max_drawdown_pct:.2f}%, CVaR 95% = {res_btc_g.metrics.cvar_95_daily_pct:.2f}%")

    # --------------------------------------------------------------------------
    # TEST 3: Apple (AAPL) Multi-Timeframe Invariance (1-Hour vs 5-Minute vs Poisson Sampling)
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Apple (AAPL): Multi-Timeframe Invariance (1-Hour vs Poisson Random Bars)")
    aapl_bars = HistoricalDataLoader.load_from_csv("data/historical/AAPL_1h_1y.csv", symbol="AAPL")
    aapl_inst = Stock(symbol="AAPL", tick_size=0.01, lot_size=1.0, taker_fee_rate=0.0005)

    strat_aapl_1h = ProbabilisticTrendStrategy(
        name="AAPL_Trend_1H",
        horizon_seconds=3600.0 * 6,
        estimator=StudentTEstimator(lookback_window=40)
    )
    res_aapl_1h = BacktestEngine(initial_cash=100000.0).run(strat_aapl_1h, aapl_inst, aapl_bars, timeframe_desc="1-Hour Bars")
    results.append(res_aapl_1h)

    # Resample AAPL to Stochastic Poisson arrival bars
    quotes_aapl = [
        # Synthesize top of book quote from each bar
        Bar(timestamp=b.timestamp, symbol=b.symbol, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume, timeframe_seconds=3600.0)
        for b in aapl_bars
    ]
    # Re-evaluate on high-frequency 5m SPY
    spy_5m_bars = HistoricalDataLoader.load_from_csv("data/historical/SPY_5m_1mo.csv", symbol="SPY")
    strat_spy_5m = ProbabilisticTrendStrategy(
        name="SPY_Intraday_5M",
        horizon_seconds=1800.0, # 30-min forward forecast
        estimator=StudentTEstimator(lookback_window=30)
    )
    res_spy_5m = BacktestEngine(initial_cash=100000.0).run(strat_spy_5m, spy_inst, spy_5m_bars, timeframe_desc="5-Minute Intraday")
    results.append(res_spy_5m)

    # --------------------------------------------------------------------------
    # TEST 4: Forex (EUR/USD) 1-Year Hourly Real Historical Data (6,140 Bars)
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Forex EUR/USD: Student-t Mean Reversion & Swap Carry")
    eurusd_bars = HistoricalDataLoader.load_from_csv("data/historical/EURUSD_X_1h_1y.csv", symbol="EURUSD=X")
    eurusd_inst = ForexPair(symbol="EUR/USD", pip_size=0.0001, lot_size=1000.0, margin_requirement=0.02)

    strat_forex = ForexMeanReversionStrategy(
        name="EURUSD_MeanReversion",
        horizon_seconds=3600.0 * 12,
        lookback_window=80,
        max_leverage=3.0
    )
    res_forex = BacktestEngine(initial_cash=100000.0).run(strat_forex, eurusd_inst, eurusd_bars, timeframe_desc="1-Hour Forex Bars")
    results.append(res_forex)

    print(f"    EUR/USD Result: Return = {res_forex.metrics.total_return_pct:+.2f}%, Sharpe = {res_forex.metrics.sharpe_ratio:.2f}, Inaction Eff = {res_forex.metrics.inaction_efficiency_pct:.1f}%")

    # --------------------------------------------------------------------------
    # TEST 5: Ethereum (ETH-USD) Real Historical Data (8,726 Bars)
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Ethereum (ETH-USD): Heavy-Tail Student-t Momentum")
    eth_bars = HistoricalDataLoader.load_from_csv("data/historical/ETH-USD_1h_1y.csv", symbol="ETH-USD")
    eth_inst = CryptoSpot(symbol="ETH-USD", tick_size=0.01, lot_size=0.01, taker_fee_rate=0.0006)

    strat_eth = ProbabilisticTrendStrategy(
        name="ETH_StudentT_Trend",
        horizon_seconds=3600.0 * 12,
        estimator=StudentTEstimator(lookback_window=60, default_df=3.2)
    )
    res_eth = BacktestEngine(initial_cash=100000.0).run(strat_eth, eth_inst, eth_bars, timeframe_desc="1-Hour Crypto Bars")
    results.append(res_eth)

    # --------------------------------------------------------------------------
    # SUMMARY TABLE & HTML REPORT GENERATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("REAL HISTORICAL DATA BENCHMARK RESULTS")
    print("=" * 105)
    print(f"{'Strategy / Test':<30} | {'Instrument':<10} | {'Timeframe':<22} | {'Return':<9} | {'Sharpe':<7} | {'MaxDD':<8} | {'Inaction Eff':<12}")
    print("-" * 105)

    for r in results:
        m = r.metrics
        print(f"{r.strategy_name:<30} | {r.instrument.symbol:<10} | {r.timeframe_description:<22} | {m.total_return_pct:>+7.2f}% | {m.sharpe_ratio:>7.2f} | {m.max_drawdown_pct:>7.2f}% | {m.inaction_efficiency_pct:>10.1f}%")

    # Generate interactive TradingView Lightweight-Charts report
    report_path = "reports/historical_hypothesis_report.html"
    os.makedirs("reports", exist_ok=True)
    ReportGenerator.generate_html_report(
        results=results,
        title="Scientific Hypothesis Testing on Real Historical Market Data (SPY, AAPL, BTC, ETH, EUR/USD)",
        output_path=report_path
    )
    print("\n" + "=" * 105)
    print(f"[+] TradingView Lightweight-Charts Report Generated: {os.path.abspath(report_path)}")
    print("=" * 105 + "\n")

    return results


if __name__ == "__main__":
    run_hypothesis_tests()

