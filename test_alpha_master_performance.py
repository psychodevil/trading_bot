#!/usr/bin/env python3
"""
Comprehensive Evaluation of AlphaMaster Probabilistic Strategy vs Buy & Hold across Real Historical Data.
"""

from dataclasses import dataclass
import os
import sys
from typing import List, Dict, Tuple, Optional

# Ensure trading_bot is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.core.instruments import Stock, CryptoSpot, ForexPair, CommodityAsset
from trading_bot.data.historical_loader import HistoricalDataLoader
from trading_bot.optimizer.cost_model import TransactionCostModel
from trading_bot.strategies.alpha_master_strategy import AlphaMasterStrategy
from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.visualization.report_generator import ReportGenerator


@dataclass
class AlphaMasterComparison:
    symbol: str
    asset_name: str
    sector: str
    bot_result: BacktestResult
    buy_and_hold_return_pct: float
    alpha_pct: float


def test_asset(
    symbol: str,
    asset_name: str,
    sector: str,
    asset_class_type: str = "stock",
    fast_period: int = 21,
    slow_period: int = 50,
    atr_mult: float = 2.5,
    max_leverage: float = 1.3,
    target_vol: float = 0.28,
    fee_rate: float = 0.0005
) -> Optional[AlphaMasterComparison]:
    safe_sym = symbol.replace('^', '').replace('=', '_')
    csv_path = f"data/historical/{safe_sym}_1h_1y.csv"

    if not os.path.exists(csv_path):
        return None

    bars = HistoricalDataLoader.load_from_csv(csv_path, symbol=symbol)
    if len(bars) < 60:
        return None

    start_price = bars[0].close
    end_price = bars[-1].close
    bnh_return_pct = ((end_price - start_price) / start_price) * 100.0

    if asset_class_type == "crypto":
        inst = CryptoSpot(symbol=symbol, tick_size=0.01, lot_size=0.001, taker_fee_rate=fee_rate)
    elif asset_class_type == "commodity":
        inst = CommodityAsset(symbol=symbol, tick_size=0.01, lot_size=1.0, taker_fee_rate=fee_rate)
    elif asset_class_type == "forex":
        inst = ForexPair(symbol=symbol, pip_size=0.0001, lot_size=1000.0)
    else:
        inst = Stock(symbol=symbol, tick_size=0.01, lot_size=1.0, taker_fee_rate=fee_rate)

    cost_model = TransactionCostModel(linear_fee_rate=fee_rate, bid_ask_half_spread=0.0002, impact_coefficient=0.0005)

    strat = AlphaMasterStrategy(
        name=f"{symbol}_AlphaMaster",
        fast_ema_period=fast_period,
        slow_ema_period=slow_period,
        atr_period=14,
        atr_multiplier=atr_mult,
        max_leverage=max_leverage,
        target_annual_vol=target_vol,
        cost_model=cost_model
    )

    engine = BacktestEngine(initial_cash=100000.0)
    res = engine.run(strat, inst, bars, timeframe_desc="1-Hour (AlphaMaster)")

    alpha = res.metrics.total_return_pct - bnh_return_pct

    return AlphaMasterComparison(
        symbol=symbol,
        asset_name=asset_name,
        sector=sector,
        bot_result=res,
        buy_and_hold_return_pct=bnh_return_pct,
        alpha_pct=alpha
    )


def main():
    print("\n" + "=" * 120)
    print("ALPHAMASTER PROBABILISTIC STRATEGY VS BUY-AND-HOLD MARKET BENCHMARK")
    print("=" * 120)

    test_universe = [
        # (Symbol, Name, Sector, Class, FastEMA, SlowEMA, ATRMult, MaxLev, TargetVol, FeeRate)
        ("SPY", "S&P 500 ETF", "Broad Market", "stock", 21, 50, 2.5, 1.3, 0.20, 0.0002),
        ("QQQ", "Nasdaq 100 ETF", "Broad Market", "stock", 21, 50, 2.5, 1.3, 0.24, 0.0002),
        ("AAPL", "Apple Inc.", "Technology", "stock", 21, 50, 2.5, 1.3, 0.26, 0.0005),
        ("MSFT", "Microsoft Corp.", "Technology", "stock", 21, 50, 2.5, 1.3, 0.24, 0.0005),
        ("NVDA", "NVIDIA Corp.", "Semiconductors", "stock", 18, 45, 2.2, 1.3, 0.38, 0.0005),
        ("AMD", "Advanced Micro Devices", "Semiconductors", "stock", 18, 45, 2.2, 1.3, 0.40, 0.0005),
        ("TSLA", "Tesla Inc.", "Automotive/Tech", "stock", 18, 45, 2.2, 1.3, 0.40, 0.0005),
        ("AMZN", "Amazon.com", "Consumer", "stock", 21, 50, 2.5, 1.3, 0.28, 0.0005),
        ("META", "Meta Platforms", "Communication", "stock", 21, 50, 2.5, 1.3, 0.30, 0.0005),
        ("GOOGL", "Alphabet Inc.", "Communication", "stock", 21, 50, 2.5, 1.3, 0.25, 0.0005),
        ("JPM", "JPMorgan Chase", "Financials", "stock", 21, 50, 2.5, 1.3, 0.20, 0.0005),
        ("BAC", "Bank of America", "Financials", "stock", 21, 50, 2.5, 1.3, 0.22, 0.0005),
        ("LLY", "Eli Lilly", "Healthcare", "stock", 21, 50, 2.5, 1.3, 0.28, 0.0005),
        ("XOM", "Exxon Mobil", "Energy", "stock", 21, 50, 2.5, 1.3, 0.22, 0.0005),
        ("GLD", "Gold ETF", "Commodities", "commodity", 21, 50, 2.5, 1.3, 0.18, 0.0003),
        ("SLV", "Silver ETF", "Commodities", "commodity", 21, 50, 2.5, 1.3, 0.28, 0.0003),
        ("BTC-USD", "Bitcoin", "Crypto L1", "crypto", 24, 60, 2.2, 1.1, 0.45, 0.0006),
        ("ETH-USD", "Ethereum", "Crypto L1", "crypto", 24, 60, 2.2, 1.1, 0.50, 0.0006),
        ("SOL-USD", "Solana", "Crypto L1", "crypto", 24, 60, 2.2, 1.1, 0.55, 0.0006),
        ("BNB-USD", "Binance Coin", "Crypto L1", "crypto", 24, 60, 2.2, 1.1, 0.40, 0.0006),
    ]

    comparisons: List[AlphaMasterComparison] = []

    for item in test_universe:
        sym, name, sec, aclass, fe, se, atrm, ml, tv, fee = item
        comp = test_asset(sym, name, sec, aclass, fe, se, atrm, ml, tv, fee)
        if comp:
            comparisons.append(comp)

    print(f"{'Asset / Symbol':<18} | {'Sector':<16} | {'Bot Return':<12} | {'Buy & Hold':<12} | {'Alpha (Excess)':<16} | {'Bot Sharpe':<10} | {'Inaction Eff':<12}")
    print("-" * 120)

    total_bot_ret = 0.0
    total_bnh_ret = 0.0
    total_alpha = 0.0
    total_sharpe = 0.0
    total_inaction = 0.0
    profitable_count = 0
    beat_market_count = 0

    for c in comparisons:
        m = c.bot_result.metrics
        bot_ret = m.total_return_pct
        bnh_ret = c.buy_and_hold_return_pct
        alpha = c.alpha_pct

        total_bot_ret += bot_ret
        total_bnh_ret += bnh_ret
        total_alpha += alpha
        total_sharpe += m.sharpe_ratio
        total_inaction += m.inaction_efficiency_pct
        if bot_ret > 0:
            profitable_count += 1
        if alpha > 0:
            beat_market_count += 1

        alpha_str = f"{alpha:>+13.2f}%"
        ret_str = f"{bot_ret:>+9.2f}%"
        bnh_str = f"{bnh_ret:>+9.2f}%"

        print(f"{c.symbol:<18} | {c.sector:<16} | {ret_str:<12} | {bnh_str:<12} | {alpha_str:<16} | {m.sharpe_ratio:>10.2f} | {m.inaction_efficiency_pct:>10.1f}%")

    n = len(comparisons)
    avg_bot_ret = total_bot_ret / n
    avg_bnh_ret = total_bnh_ret / n
    avg_alpha = total_alpha / n
    avg_sharpe = total_sharpe / n
    avg_inaction = total_inaction / n

    print("=" * 120)
    print(f"{'AVERAGE OVERALL':<18} | {'All Sectors':<16} | {avg_bot_ret:>+9.2f}%   | {avg_bnh_ret:>+9.2f}%   | {avg_alpha:>+13.2f}%   | {avg_sharpe:>10.2f} | {avg_inaction:>10.1f}%")
    print(f"Profitable Assets: {profitable_count}/{n} ({profitable_count/n*100:.1f}%) | Beating Market Rate: {beat_market_count}/{n} ({beat_market_count/n*100:.1f}%)")
    print(f"Average Alpha Generated: {avg_alpha:+.2f}% Over Market Buy & Hold Rate")
    print("=" * 120)

    # Generate dedicated TradingView report
    report_path = "reports/alpha_master_vs_market_report.html"
    os.makedirs("reports", exist_ok=True)
    all_results = [c.bot_result for c in comparisons]
    ReportGenerator.generate_html_report(
        results=all_results,
        title="AlphaMaster High-Performance Probabilistic Strategy vs Buy & Hold",
        output_path=report_path
    )
    print(f"\n[+] Interactive TradingView Report: {os.path.abspath(report_path)}\n")


if __name__ == "__main__":
    main()
