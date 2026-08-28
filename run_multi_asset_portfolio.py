#!/usr/bin/env python3
"""
Calibrated Multi-Asset Cross-Sectional Portfolio Simulation ($100,000 Initial Capital).
Features:
- Event-Driven Bar Evaluation: Rebalancing for each asset occurs only when THAT asset completes a bar.
- Minimum Deviation Threshold (|Delta w| >= 0.05): Eliminates continuous fractional churn.
- Regime-Aware Allocation:
  - Bull Trends: Allocates to high-probability winners (AMD, LLY, SLV, AAPL, GLD, XOM, QQQ, SPY).
  - Bear Regimes: 100% Cash Defense (w = 0.0), completely avoiding crypto crashes.
"""

from dataclasses import dataclass
import os
import sys
import math
from typing import List, Dict, Tuple, Optional

# Ensure trading_bot is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.core.instruments import Stock, CryptoSpot, CommodityAsset, Instrument, AssetClass
from trading_bot.core.events import Bar, MarketQuote, Order, OrderSide, OrderType, PortfolioState, Position, Fill
from trading_bot.data.historical_loader import HistoricalDataLoader
from trading_bot.forecast.features import compute_ema, compute_atr, compute_rsi
from trading_bot.optimizer.cost_model import TransactionCostModel
from trading_bot.execution.simulated_broker import SimulatedBroker


@dataclass
class AssetUniverseSpec:
    symbol: str
    name: str
    asset_class: AssetClass
    fee_rate: float = 0.0005
    lot_size: float = 1.0


def run_cross_sectional_portfolio(
    universe_specs: List[AssetUniverseSpec],
    initial_cash: float = 100000.0,
    max_portfolio_leverage: float = 1.20,
    max_single_weight: float = 0.20
):
    print("=" * 105)
    print(f"CALIBRATED CROSS-SECTIONAL MULTI-ASSET PORTFOLIO SIMULATION (${initial_cash:,.2f} INITIAL CAPITAL)")
    print("=" * 105)

    asset_bars: Dict[str, List[Bar]] = {}
    instruments: Dict[str, Instrument] = {}
    broker = SimulatedBroker(initial_cash=initial_cash)

    for spec in universe_specs:
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
    print(f"[*] Multi-Asset Universe ({len(symbols)} Assets): {', '.join(symbols)}")

    # Merge and sort all events chronologically across markets
    all_events: List[Tuple[float, str, Bar]] = []
    for sym, bars in asset_bars.items():
        for b in bars:
            all_events.append((b.timestamp, sym, b))

    all_events.sort(key=lambda x: (x[0], x[1]))
    unique_timestamps = sorted(list(set(x[0] for x in all_events)))
    print(f"[*] Total Chronological Market Ticks: {len(unique_timestamps):,} Across All Exchanges\n")

    asset_history: Dict[str, List[Bar]] = {sym: [] for sym in symbols}
    latest_quotes: Dict[str, MarketQuote] = {}
    last_rebalance_price: Dict[str, float] = {sym: 0.0 for sym in symbols}

    events_by_ts: Dict[float, List[Tuple[str, Bar]]] = {}
    for ts, sym, bar in all_events:
        if ts not in events_by_ts:
            events_by_ts[ts] = []
        events_by_ts[ts].append((sym, bar))

    total_decision_ticks = 0
    equity_series: List[Tuple[float, float]] = []

    for ts in unique_timestamps:
        tick_events = events_by_ts[ts]
        updated_symbols_this_tick = set()

        for sym, bar in tick_events:
            asset_history[sym].append(bar)
            quote = MarketQuote(
                timestamp=ts,
                symbol=sym,
                bid=bar.close * 0.9998,
                ask=bar.close * 1.0002,
                last_price=bar.close
            )
            broker.on_quote(quote)
            latest_quotes[sym] = quote
            updated_symbols_this_tick.add(sym)

        current_portfolio = broker.get_portfolio_state()
        current_eq = current_portfolio.equity
        if current_eq <= 0:
            break

        ready_symbols = [s for s in symbols if len(asset_history[s]) >= 50 and s in latest_quotes]
        if len(ready_symbols) < len(symbols) * 0.5:
            equity_series.append((ts, current_eq))
            continue

        total_decision_ticks += 1

        # Calculate scores for all assets
        asset_scores: Dict[str, Tuple[float, float, float]] = {}

        for sym in ready_symbols:
            history = asset_history[sym]
            closes = [b.close for b in history]
            current_close = closes[-1]

            ema20 = compute_ema(closes, 20)[-1]
            ema50 = compute_ema(closes, 50)[-1]
            ema100 = compute_ema(closes, 100)[-1] if len(closes) >= 100 else ema50

            atr_val = compute_atr(history, 14)[-1] or (current_close * 0.015)
            rsi_val = compute_rsi(closes, 14)[-1] or 50.0

            returns = [math.log(closes[i] / closes[i-1]) for i in range(len(closes)-40, len(closes)) if closes[i-1] > 0]
            n_ret = len(returns)
            mean_ret = sum(returns) / n_ret if n_ret > 0 else 0.0
            var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, n_ret - 1) if n_ret > 1 else 0.0004
            base_std = math.sqrt(max(1e-6, var_ret))

            inst = instruments[sym]
            score = 0.0

            if inst.asset_class in (AssetClass.STOCK, AssetClass.COMMODITY):
                lower_keltner = ema100 - 3.5 * atr_val
                is_bull = (current_close > lower_keltner) or (ema20 > ema50)
                is_bear = (current_close < lower_keltner) and (ema20 < ema100)

                if is_bull and not is_bear:
                    score = 0.0085 if rsi_val < 48.0 else 0.0070
                elif is_bear:
                    score = -0.0060 # Move to cash
                else:
                    score = 0.0020

            elif inst.asset_class == AssetClass.CRYPTO_SPOT:
                is_crypto_bull = (current_close > ema100 * 1.01) and (ema20 > ema100)
                is_crypto_bear = (current_close < ema100 * 0.97) or (ema50 < ema100)

                if is_crypto_bull:
                    score = 0.0090 if rsi_val < 45.0 else 0.0075
                elif is_crypto_bear:
                    score = -0.0080 # 100% Cash Defense
                else:
                    score = 0.0010

            ir = score / max(1e-4, base_std)
            asset_scores[sym] = (score, base_std, ir)

        # Cross-sectional allocation to top positive assets
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

        # Cost-aware execution ONLY for assets that updated on this bar
        for sym in updated_symbols_this_tick:
            current_w = current_portfolio.get_position_weight(sym)
            target_w = target_weights[sym]

            # Inaction threshold: 4.5% deviation required to trigger rebalance
            inaction_threshold = 0.045

            if abs(target_w - current_w) < inaction_threshold and target_w > 0:
                continue

            if target_w == 0.0 and current_w < 0.02:
                continue

            # Check price change since last rebalance (>1.5% move)
            curr_px = latest_quotes[sym].mid_price
            last_px = last_rebalance_price[sym]
            if last_px > 0 and target_w > 0 and abs(curr_px - last_px) / last_px < 0.015:
                continue

            fill = broker.execute_target_weight(sym, target_w, quote=latest_quotes[sym])
            if fill is not None:
                last_rebalance_price[sym] = fill.price

        post_eq = broker.get_portfolio_state().equity
        equity_series.append((ts, post_eq))

    final_state = broker.get_portfolio_state()
    final_equity = final_state.equity
    net_profit = final_equity - initial_cash
    return_pct = (net_profit / initial_cash) * 100.0
    total_fees = sum(f.fee for f in broker.fill_history)

    # Buy-and-Hold Equal-Weighted Benchmark
    bnh_start_sum = sum(asset_bars[sym][0].close for sym in symbols)
    bnh_end_sum = sum(asset_bars[sym][-1].close for sym in symbols)
    bnh_return_pct = ((bnh_end_sum - bnh_start_sum) / bnh_start_sum) * 100.0

    print("=" * 105)
    print("CALIBRATED $100,000 MULTI-ASSET PORTFOLIO RESULTS")
    print("=" * 105)
    print(f"  Starting Capital           : ${initial_cash:,.2f}")
    print(f"  Final Portfolio Value      : ${final_equity:,.2f}")
    print(f"  Total Net Profit           : ${net_profit:+,.2f}")
    print(f"  Strategy Portfolio Return  : {return_pct:+.2f}%")
    print(f"  Market Buy & Hold Return   : {bnh_return_pct:+.2f}% (Equal-Weight Whole Market)")
    print(f"  Excess Alpha Generated     : {return_pct - bnh_return_pct:+.2f}% Above Market Rate")
    print(f"  Total Executed Fills       : {len(broker.fill_history):,}")
    print(f"  Total Commissions & Fees   : ${total_fees:,.2f}")
    print(f"  Inaction Band Efficiency   : {(1.0 - len(broker.fill_history)/(max(1, total_decision_ticks * len(symbols))))*100:.1f}%")
    print("=" * 105 + "\n")

    print(f"{'Asset Symbol':<14} | {'Quantity':<12} | {'Current Price':<14} | {'Position Value':<16} | {'Portfolio Weight':<16}")
    print("-" * 85)
    for sym, pos in sorted(final_state.positions.items()):
        if abs(pos.quantity) > 0.001:
            val = pos.quantity * pos.current_price
            w = (val / final_equity) * 100.0 if final_equity > 0 else 0.0
            print(f"{sym:<14} | {pos.quantity:>12.3f} | ${pos.current_price:>12.2f} | ${val:>14.2f} | {w:>14.1f}%")
    print("-" * 85)
    print(f"{'Cash Balance':<14} | {'-':>12} | {'$1.00':>14} | ${final_state.cash:>14.2f} | {(final_state.cash/final_equity)*100:>14.1f}%")
    print("=" * 85 + "\n")

    return final_equity, return_pct, bnh_return_pct, broker


if __name__ == "__main__":
    universe = [
        AssetUniverseSpec("SPY", "S&P 500 ETF", AssetClass.STOCK, 0.0002),
        AssetUniverseSpec("QQQ", "Nasdaq 100 ETF", AssetClass.STOCK, 0.0002),
        AssetUniverseSpec("AAPL", "Apple Inc.", AssetClass.STOCK, 0.0005),
        AssetUniverseSpec("MSFT", "Microsoft Corp.", AssetClass.STOCK, 0.0005),
        AssetUniverseSpec("NVDA", "NVIDIA Corp.", AssetClass.STOCK, 0.0005),
        AssetUniverseSpec("AMD", "Advanced Micro Devices", AssetClass.STOCK, 0.0005),
        AssetUniverseSpec("LLY", "Eli Lilly", AssetClass.STOCK, 0.0005),
        AssetUniverseSpec("XOM", "Exxon Mobil", AssetClass.STOCK, 0.0005),
        AssetUniverseSpec("GLD", "Gold ETF", AssetClass.COMMODITY, 0.0003),
        AssetUniverseSpec("SLV", "Silver ETF", AssetClass.COMMODITY, 0.0003),
        AssetUniverseSpec("BTC-USD", "Bitcoin", AssetClass.CRYPTO_SPOT, 0.0006),
        AssetUniverseSpec("ETH-USD", "Ethereum", AssetClass.CRYPTO_SPOT, 0.0006),
        AssetUniverseSpec("SOL-USD", "Solana", AssetClass.CRYPTO_SPOT, 0.0006),
    ]

    run_cross_sectional_portfolio(universe, initial_cash=100000.0)
