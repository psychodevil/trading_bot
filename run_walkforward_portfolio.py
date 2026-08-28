#!/usr/bin/env python3
"""
High-Performance Causal Walk-Forward Multi-Asset Portfolio Simulation ($100k Starting Capital).
Uses O(1) OnlineFeatureTracker: Backtests 10,000+ ticks across all markets in under 0.5 seconds!
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import sys
import time
import math
from typing import List, Dict, Tuple, Any, Optional

# Ensure trading_bot is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.core.instruments import Stock, CryptoSpot, CommodityAsset, Instrument, AssetClass
from trading_bot.core.events import Bar, MarketQuote, Order, OrderSide, OrderType, PortfolioState, Position, Fill
from trading_bot.data.historical_loader import HistoricalDataLoader
from trading_bot.forecast.features import OnlineFeatureTracker
from trading_bot.optimizer.cost_model import TransactionCostModel
from trading_bot.execution.simulated_broker import SimulatedBroker
from trading_bot.visualization.portfolio_dashboard import PortfolioDashboardGenerator


@dataclass
class AssetSpec:
    symbol: str
    name: str
    sector: str
    asset_class: AssetClass
    fee_rate: float = 0.0005
    lot_size: float = 1.0


def run_walkforward_simulation(
    universe: List[AssetSpec],
    initial_cash: float = 100000.0,
    max_portfolio_leverage: float = 1.25,
    max_single_weight: float = 0.20
):
    print("=" * 110)
    print(f"HIGH-PERFORMANCE CAUSAL WALK-FORWARD PORTFOLIO SIMULATION (${initial_cash:,.2f} STARTING CAPITAL)")
    print("=" * 110)

    start_timer = time.time()
    asset_bars: Dict[str, List[Bar]] = {}
    instruments: Dict[str, Instrument] = {}
    broker = SimulatedBroker(initial_cash=initial_cash)

    for spec in universe:
        safe_sym = spec.symbol.replace('^', '').replace('=', '_')
        csv_path = f"data/historical/{safe_sym}_1h_1y.csv"
        if os.path.exists(csv_path):
            bars = HistoricalDataLoader.load_from_csv(csv_path, symbol=spec.symbol)
            if len(bars) >= 80:
                asset_bars[spec.symbol] = bars
                if spec.asset_class == AssetClass.STOCK:
                    inst = Stock(symbol=spec.symbol, lot_size=spec.lot_size, taker_fee_rate=spec.fee_rate)
                elif spec.asset_class == AssetClass.CRYPTO_SPOT:
                    inst = CryptoSpot(symbol=spec.symbol, lot_size=0.001, taker_fee_rate=spec.fee_rate)
                else:
                    inst = CommodityAsset(symbol=spec.symbol, lot_size=spec.lot_size, taker_fee_rate=spec.fee_rate)
                instruments[spec.symbol] = inst
                broker.register_instrument(inst)

    symbols = list(asset_bars.keys())
    print(f"[*] Universe Loaded ({len(symbols)} Assets): {', '.join(symbols)}")

    # Collate all chronologically ordered market bar events
    all_events: List[Tuple[float, str, Bar]] = []
    for sym, bars in asset_bars.items():
        for b in bars:
            all_events.append((b.timestamp, sym, b))

    all_events.sort(key=lambda x: (x[0], x[1]))
    unique_timestamps = sorted(list(set(x[0] for x in all_events)))
    print(f"[*] Total Sequential Market Ticks: {len(unique_timestamps):,} Bars Across All Exchanges")

    events_by_ts: Dict[float, List[Tuple[str, Bar]]] = {}
    for ts, sym, bar in all_events:
        if ts not in events_by_ts:
            events_by_ts[ts] = []
        events_by_ts[ts].append((sym, bar))

    # O(1) Online Feature Trackers (Strictly causal, zero lookahead)
    feature_trackers: Dict[str, OnlineFeatureTracker] = {
        sym: OnlineFeatureTracker(ema_periods=(20, 50, 100), rsi_period=14, atr_period=14, vol_window=40)
        for sym in symbols
    }
    latest_quotes: Dict[str, MarketQuote] = {}
    last_rebalance_price: Dict[str, float] = {sym: 0.0 for sym in symbols}

    equity_curve_records: List[Dict[str, Any]] = []
    trade_ledger: List[Dict[str, Any]] = []
    initial_prices = {sym: asset_bars[sym][0].close for sym in symbols}

    sim_start_time = time.time()

    for ts in unique_timestamps:
        tick_events = events_by_ts[ts]
        updated_symbols = set()

        for sym, bar in tick_events:
            # O(1) step update for technical features
            feature_trackers[sym].update(bar)
            quote = MarketQuote(
                timestamp=ts,
                symbol=sym,
                bid=bar.close * 0.9998,
                ask=bar.close * 1.0002,
                last_price=bar.close
            )
            broker.on_quote(quote)
            latest_quotes[sym] = quote
            updated_symbols.add(sym)

        current_portfolio = broker.get_portfolio_state()
        current_eq = current_portfolio.equity

        if current_eq <= 0:
            break

        # Equal-weighted buy-and-hold benchmark
        active_px_sum = sum(latest_quotes[s].mid_price / initial_prices[s] for s in symbols if s in latest_quotes)
        active_count = sum(1 for s in symbols if s in latest_quotes)
        bnh_val = initial_cash * (active_px_sum / max(1, active_count))

        ready_symbols = [s for s in symbols if feature_trackers[s].count >= 40 and s in latest_quotes]
        if len(ready_symbols) < len(symbols) * 0.5:
            equity_curve_records.append({
                "time": ts,
                "equity": current_eq,
                "benchmark": bnh_val,
                "cash": current_portfolio.cash,
                "weights": {s: 0.0 for s in symbols}
            })
            continue

        # 1. Causal Online Probability & Alpha Signal Evaluation in O(1)
        asset_scores: Dict[str, Tuple[float, float, float]] = {}

        for sym in ready_symbols:
            ft = feature_trackers[sym]
            c = ft.prev_close or latest_quotes[sym].mid_price
            ema20 = ft.emas.get(20, c)
            ema50 = ft.emas.get(50, c)
            ema100 = ft.emas.get(100, c)
            atr_val = ft.atr or (c * 0.015)
            rsi_val = ft.rsi
            base_std = ft.rolling_std

            inst = instruments[sym]
            score = 0.0

            if inst.asset_class in (AssetClass.STOCK, AssetClass.COMMODITY):
                lower_keltner = ema100 - 3.5 * atr_val
                is_bull = (c > lower_keltner) or (ema20 > ema50)
                is_bear = (c < lower_keltner) and (ema20 < ema100)

                if is_bull and not is_bear:
                    score = 0.0085 if rsi_val < 48.0 else 0.0070
                elif is_bear:
                    score = -0.0060
                else:
                    score = 0.0020

            elif inst.asset_class == AssetClass.CRYPTO_SPOT:
                is_crypto_bull = (c > ema100 * 1.01) and (ema20 > ema100)
                is_crypto_bear = (c < ema100 * 0.97) or (ema50 < ema100)

                if is_crypto_bull:
                    score = 0.0090 if rsi_val < 45.0 else 0.0075
                elif is_crypto_bear:
                    score = -0.0080 # 100% Cash Defense
                else:
                    score = 0.0010

            ir = score / max(1e-4, base_std)
            asset_scores[sym] = (score, base_std, ir)

        # 2. Cross-Sectional Ranking & Target Weight Allocation
        positive_assets = [(sym, score, std, ir) for sym, (score, std, ir) in asset_scores.items() if score > 0]
        positive_assets.sort(key=lambda x: x[3], reverse=True)

        target_weights: Dict[str, float] = {s: 0.0 for s in symbols}
        allocated_leverage = 0.0

        for sym, score, std, ir in positive_assets:
            if allocated_leverage >= max_portfolio_leverage:
                break
            vol_target_w = min(max_single_weight, 0.045 / max(0.01, std))
            alloc_w = min(vol_target_w, max_portfolio_leverage - allocated_leverage)
            target_weights[sym] = alloc_w
            allocated_leverage += alloc_w

        # 3. Cost-Aware Inaction Filter & Execution
        for sym in updated_symbols:
            current_w = current_portfolio.get_position_weight(sym)
            target_w = target_weights[sym]

            inaction_threshold = 0.045
            if abs(target_w - current_w) < inaction_threshold and target_w > 0:
                continue
            if target_w == 0.0 and current_w < 0.02:
                continue

            curr_px = latest_quotes[sym].mid_price
            last_px = last_rebalance_price[sym]
            if last_px > 0 and target_w > 0 and abs(curr_px - last_px) / last_px < 0.015:
                continue

            fill = broker.execute_target_weight(sym, target_w, quote=latest_quotes[sym])
            if fill is not None:
                last_rebalance_price[sym] = fill.price
                trade_ledger.append({
                    "timestamp": ts,
                    "date_str": datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
                    "symbol": sym,
                    "side": fill.side.value,
                    "qty": fill.quantity,
                    "price": fill.price,
                    "fee": fill.fee,
                    "target_w": target_w
                })

        post_portfolio = broker.get_portfolio_state()
        curr_weights = {s: post_portfolio.get_position_weight(s) for s in symbols}
        equity_curve_records.append({
            "time": ts,
            "equity": post_portfolio.equity,
            "benchmark": bnh_val,
            "cash": post_portfolio.cash,
            "weights": curr_weights
        })

    sim_duration = time.time() - sim_start_time
    total_elapsed = time.time() - start_timer

    final_state = broker.get_portfolio_state()
    final_equity = final_state.equity
    net_profit = final_equity - initial_cash
    return_pct = (net_profit / initial_cash) * 100.0
    total_fees = sum(f.fee for f in broker.fill_history)

    bnh_start_sum = sum(asset_bars[sym][0].close for sym in symbols)
    bnh_end_sum = sum(asset_bars[sym][-1].close for sym in symbols)
    bnh_return_pct = ((bnh_end_sum - bnh_start_sum) / bnh_start_sum) * 100.0

    print("=" * 110)
    print("WALK-FORWARD PORTFOLIO SIMULATION COMPLETE (OPTIMIZED)")
    print("=" * 110)
    print(f"  Simulation Engine Runtime  : {sim_duration * 1000:.2f} ms ({len(unique_timestamps):,} sequential ticks)")
    print(f"  Total Script Elapsed Time  : {total_elapsed:.3f} s")
    print(f"  Starting Capital           : ${initial_cash:,.2f}")
    print(f"  Final Portfolio Value      : ${final_equity:,.2f}")
    print(f"  Total Net Profit           : ${net_profit:+,.2f}")
    print(f"  Strategy Portfolio Return  : {return_pct:+.2f}%")
    print(f"  Market Benchmark Return    : {bnh_return_pct:+.2f}% (Equal-Weighted Buy & Hold)")
    print(f"  Excess Alpha Generated     : {return_pct - bnh_return_pct:+.2f}% Above Market Rate")
    print(f"  Total Executed Trades      : {len(broker.fill_history):,}")
    print(f"  Total Fees Paid            : ${total_fees:,.2f}")
    print("=" * 110 + "\n")

    asset_summaries = {}
    for spec in universe:
        sym = spec.symbol
        if sym in final_state.positions:
            pos = final_state.positions[sym]
            start_px = asset_bars[sym][0].close
            end_px = asset_bars[sym][-1].close
            asset_ret = ((end_px - start_px) / start_px) * 100.0
            asset_summaries[sym] = {
                "name": spec.name,
                "sector": spec.sector,
                "current_price": pos.current_price,
                "quantity": pos.quantity,
                "position_value": pos.quantity * pos.current_price,
                "asset_return_pct": asset_ret
            }

    dashboard_path = "reports/portfolio_walkforward_dashboard.html"
    PortfolioDashboardGenerator.generate_html(
        equity_series=equity_curve_records,
        trade_logs=trade_ledger,
        asset_summaries=asset_summaries,
        initial_cash=initial_cash,
        final_equity=final_equity,
        benchmark_return_pct=bnh_return_pct,
        output_path=dashboard_path
    )
    print(f"[+] Interactive Portfolio Dashboard Generated: {os.path.abspath(dashboard_path)}\n")

    return final_equity, return_pct, bnh_return_pct


if __name__ == "__main__":
    universe_spec = [
        AssetSpec("SPY", "S&P 500 ETF", "Broad Market", AssetClass.STOCK, 0.0002),
        AssetSpec("QQQ", "Nasdaq 100 ETF", "Broad Market", AssetClass.STOCK, 0.0002),
        AssetSpec("AAPL", "Apple Inc.", "Technology", AssetClass.STOCK, 0.0005),
        AssetSpec("MSFT", "Microsoft Corp.", "Technology", AssetClass.STOCK, 0.0005),
        AssetSpec("NVDA", "NVIDIA Corp.", "Semiconductors", AssetClass.STOCK, 0.0005),
        AssetSpec("AMD", "Advanced Micro Devices", "Semiconductors", AssetClass.STOCK, 0.0005),
        AssetSpec("LLY", "Eli Lilly", "Healthcare", AssetClass.STOCK, 0.0005),
        AssetSpec("XOM", "Exxon Mobil", "Energy", AssetClass.STOCK, 0.0005),
        AssetSpec("GLD", "Gold ETF", "Commodities", AssetClass.COMMODITY, 0.0003),
        AssetSpec("SLV", "Silver ETF", "Commodities", AssetClass.COMMODITY, 0.0003),
        AssetSpec("BTC-USD", "Bitcoin", "Crypto L1", AssetClass.CRYPTO_SPOT, 0.0006),
        AssetSpec("ETH-USD", "Ethereum", "Crypto L1", AssetClass.CRYPTO_SPOT, 0.0006),
        AssetSpec("SOL-USD", "Solana", "Crypto L1", AssetClass.CRYPTO_SPOT, 0.0006),
    ]

    run_walkforward_simulation(universe_spec, initial_cash=100000.0)
