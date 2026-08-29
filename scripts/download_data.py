#!/usr/bin/env python3
"""
QuantumAlpha: Historical Market Data Downloader.
Downloads 1-year hourly OHLCV bars across all 62 assets using Yahoo Finance.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trading_bot.data.market_universe import MARKET_UNIVERSE, MarketAssetInfo


def download_asset_data(asset: MarketAssetInfo, output_dir: str):
    import yfinance as yf

    sym = asset.symbol
    safe_sym = sym.replace('^', '').replace('=', '_')
    csv_file = os.path.join(output_dir, f"{safe_sym}_1h_1y.csv")

    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="1y", interval="1h")
        if df.empty or len(df) < 50:
            print(f"[-] {sym:<10} Failed / Empty data")
            return sym, False, 0

        df.reset_index(inplace=True)
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df.rename(columns={
            date_col: 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        df['timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp()))
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.to_csv(csv_file, index=False)
        return sym, True, len(df)
    except Exception as e:
        return sym, False, 0


def main():
    try:
        import yfinance
    except ImportError:
        print("[!] Error: yfinance is required to download real market data. Run inside .venv or Docker.")
        sys.exit(1)

    output_dir = os.path.join(PROJECT_ROOT, "data/historical")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"DOWNLOADING HISTORICAL MARKET DATA FOR {len(MARKET_UNIVERSE)} ASSETS")
    print("=" * 70)

    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_asset_data, asset, output_dir): asset for asset in MARKET_UNIVERSE}
        for f in as_completed(futures):
            sym, ok, count = f.result()
            if ok:
                success_count += 1
                print(f"[+] {sym:<12} Downloaded {count:,} hourly bars")
            else:
                print(f"[-] {sym:<12} Download failed")

    print("=" * 70)
    print(f"[+] Complete: {success_count}/{len(MARKET_UNIVERSE)} assets downloaded to {output_dir}")


if __name__ == "__main__":
    main()

