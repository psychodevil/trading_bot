#!/usr/bin/env python3
"""
Downloads real historical data from Yahoo Finance for Stocks, Crypto, and Forex.
"""

import os
import sys

# Ensure trading_bot is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.data.historical_loader import HistoricalDataLoader


def main():
    print("=" * 70)
    print("FETCHING REAL HISTORICAL MARKET DATA (STOCKS, CRYPTO, FOREX)")
    print("=" * 70)

    symbols_to_fetch = [
        ("SPY", "1h", "1y"),
        ("SPY", "5m", "1mo"),
        ("AAPL", "1h", "1y"),
        ("BTC-USD", "1h", "1y"),
        ("ETH-USD", "1h", "1y"),
        ("EURUSD=X", "1h", "1y"),
    ]

    for sym, interval, period in symbols_to_fetch:
        print(f"[*] Fetching real market data for {sym} (Interval: {interval}, Period: {period}) ...")
        bars = HistoricalDataLoader.fetch_yahoo_bars(symbol=sym, interval=interval, range_period=period)
        print(f"    -> Successfully cached {len(bars)} bars for {sym} (First: {bars[0].close:.2f}, Last: {bars[-1].close:.2f})\n")

    print("[+] All historical datasets downloaded and cached to data/historical/\n")


if __name__ == "__main__":
    main()

