"""
REST API Endpoints for QuantumAlpha Web Dashboard & Simulation Engine.
"""

from __future__ import annotations
import json
import os
import sys
import math
from typing import Dict, Any, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trading_bot.data.market_universe import MARKET_UNIVERSE
from trading_bot.core.instruments import Stock, CryptoSpot, CommodityAsset, ForexPair, AssetClass
from trading_bot.data.historical_loader import HistoricalDataLoader
from trading_bot.core.events import Bar, MarketQuote
from trading_bot.execution.simulated_broker import SimulatedBroker
from trading_bot.forecast.features import OnlineFeatureTracker


def get_market_universe_data() -> List[Dict[str, Any]]:
    """Returns metadata and statistics for all 62 market assets."""
    assets = []
    for m in MARKET_UNIVERSE:
        safe_sym = m.symbol.replace('^', '').replace('=', '_')
        csv_path = os.path.join(PROJECT_ROOT, f"data/historical/{safe_sym}_1h_1y.csv")
        bars_count = 0
        latest_px = 0.0
        return_pct = 0.0

        if os.path.exists(csv_path):
            try:
                bars = HistoricalDataLoader.load_from_csv(csv_path, symbol=m.symbol)
                bars_count = len(bars)
                if bars_count > 0:
                    latest_px = bars[-1].close
                    start_px = bars[0].close
                    return_pct = ((latest_px - start_px) / max(1e-6, start_px)) * 100.0
            except Exception:
                pass

        assets.append({
            "symbol": m.symbol,
            "name": m.name,
            "sector": m.sector,
            "asset_class": m.asset_class.value,
            "bars_count": bars_count,
            "latest_price": round(latest_px, 4),
            "return_pct": round(return_pct, 2)
        })
    return assets


def get_asset_bars_data(symbol: str) -> List[Dict[str, Any]]:
    """Returns historical OHLCV candlestick bars for a given symbol."""
    safe_sym = symbol.replace('^', '').replace('=', '_')
    csv_path = os.path.join(PROJECT_ROOT, f"data/historical/{safe_sym}_1h_1y.csv")
    if not os.path.exists(csv_path):
        return []

    bars = HistoricalDataLoader.load_from_csv(csv_path, symbol=symbol)
    result = []
    for b in bars:
        result.append({
            "time": int(b.timestamp),
            "open": round(b.open, 4),
            "high": round(b.high, 4),
            "low": round(b.low, 4),
            "close": round(b.close, 4),
            "volume": round(b.volume, 2)
        })
    return result


def run_portfolio_simulation_api(
    initial_cash: float = 100000.0,
    max_leverage: float = 1.35,
    top_k: int = 4,
    selected_symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes an institutional low-turnover momentum walk-forward simulation.
    """
    symbols = selected_symbols or [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "LLY", "XOM", "GLD", "SLV", "BTC-USD", "ETH-USD", "SOL-USD"
    ]

    asset_bars: Dict[str, List[Bar]] = {}
    instruments: Dict[str, Any] = {}
    broker = SimulatedBroker(initial_cash=initial_cash)

    for sym in symbols:
        safe_sym = sym.replace('^', '').replace('=', '_')
        csv_path = os.path.join(PROJECT_ROOT, f"data/historical/{safe_sym}_1h_1y.csv")
        if os.path.exists(csv_path):
            bars = HistoricalDataLoader.load_from_csv(csv_path, symbol=sym)
            if len(bars) >= 60:
                asset_bars[sym] = bars
                if "USD" in sym:
                    inst = CryptoSpot(symbol=sym, lot_size=0.001, taker_fee_rate=0.0006)
                elif sym in ("GLD", "SLV", "USO"):
                    inst = CommodityAsset(symbol=sym, lot_size=1.0, taker_fee_rate=0.0003)
                else:
                    inst = Stock(symbol=sym, lot_size=1.0, taker_fee_rate=0.0005)
                instruments[sym] = inst
                broker.register_instrument(inst)

    active_symbols = list(asset_bars.keys())
    all_events: List[Tuple[float, str, Bar]] = []
    for sym, bars in asset_bars.items():
        for b in bars:
            all_events.append((b.timestamp, sym, b))

    all_events.sort(key=lambda x: (x[0], x[1]))
    unique_timestamps = sorted(list(set(x[0] for x in all_events)))

    events_by_ts: Dict[float, List[Tuple[str, Bar]]] = {}
    for ts, sym, bar in all_events:
        if ts not in events_by_ts:
            events_by_ts[ts] = []
        events_by_ts[ts].append((sym, bar))

    feature_trackers: Dict[str, OnlineFeatureTracker] = {
        sym: OnlineFeatureTracker(ema_periods=(20, 50, 100), rsi_period=14, atr_period=14, vol_window=40)
        for sym in active_symbols
    }
    latest_quotes: Dict[str, MarketQuote] = {}
    price_ring_80: Dict[str, List[float]] = {sym: [] for sym in active_symbols}
    highest_price_tracking: Dict[str, float] = {sym: 0.0 for sym in active_symbols}
    current_held_leaders: Set[str] = set()

    equity_records: List[Dict[str, Any]] = []
    trade_ledger: List[Dict[str, Any]] = []
    initial_prices = {sym: asset_bars[sym][0].close for sym in active_symbols}

    tick_count = 0
    rebalance_interval = 120

    for ts in unique_timestamps:
        tick_count += 1
        tick_events = events_by_ts[ts]
        updated_symbols = set()

        for sym, bar in tick_events:
            feature_trackers[sym].update(bar)
            ring = price_ring_80[sym]
            ring.append(bar.close)
            if len(ring) > 80:
                ring.pop(0)

            quote = MarketQuote(timestamp=ts, symbol=sym, bid=bar.close * 0.9998, ask=bar.close * 1.0002, last_price=bar.close)
            broker.on_quote(quote)
            latest_quotes[sym] = quote
            updated_symbols.add(sym)

        current_portfolio = broker.get_portfolio_state()
        current_eq = current_portfolio.equity
        if current_eq <= 0:
            break

        active_px_sum = sum(latest_quotes[s].mid_price / initial_prices[s] for s in active_symbols if s in latest_quotes)
        bnh_val = initial_cash * (active_px_sum / max(1, len(latest_quotes)))

        ready_symbols = [s for s in active_symbols if feature_trackers[s].count >= 60 and s in latest_quotes]
        if len(ready_symbols) < len(active_symbols) * 0.5:
            equity_records.append({
                "time": int(ts),
                "equity": round(current_eq, 2),
                "benchmark": round(bnh_val, 2),
                "cash": round(current_portfolio.cash, 2),
                "weights": {s: 0.0 for s in active_symbols}
            })
            continue

        # Hourly Trailing Stop Checks
        stopped_out: Set[str] = set()
        for sym in list(current_held_leaders):
            if sym in ready_symbols:
                c = latest_quotes[sym].mid_price
                ft = feature_trackers[sym]
                atr_val = ft.atr or (c * 0.015)
                highest_price_tracking[sym] = max(highest_price_tracking[sym], c)
                trailing_stop = highest_price_tracking[sym] - 3.8 * atr_val
                ema100 = ft.emas.get(100, c)

                if c < trailing_stop or c < ema100 * 0.96:
                    stopped_out.add(sym)
                    current_held_leaders.remove(sym)
                    fill = broker.execute_target_weight(sym, 0.0, quote=latest_quotes[sym])
                    if fill is not None:
                        trade_ledger.append({
                            "timestamp": int(ts),
                            "symbol": sym,
                            "side": fill.side.value.upper(),
                            "quantity": round(fill.quantity, 4),
                            "price": round(fill.price, 4),
                            "fee": round(fill.fee, 4),
                            "target_weight": 0.0
                        })

        # Scheduled Rebalancing
        if tick_count % rebalance_interval == 0 or len(current_held_leaders) == 0:
            scored_universe = []
            for sym in ready_symbols:
                ft = feature_trackers[sym]
                c = latest_quotes[sym].mid_price
                ema20 = ft.emas.get(20, c)
                ema50 = ft.emas.get(50, c)
                ema100 = ft.emas.get(100, c)
                rsi_val = ft.rsi
                inst = instruments[sym]

                ring = price_ring_80[sym]
                past_px = ring[0] if len(ring) >= 40 else c
                ret_80 = (c - past_px) / past_px

                is_trend_intact = (c > ema100 * 0.98) and (ema20 > ema50 or c > ema50)
                if inst.asset_class == AssetClass.CRYPTO_SPOT:
                    is_trend_intact = is_trend_intact and (c > ema100 * 1.02) and (ema50 > ema100)

                if is_trend_intact and ret_80 > 0.01:
                    score = ret_80 * (1.15 if 40.0 <= rsi_val <= 65.0 else 0.85)
                    scored_universe.append((sym, score))

            scored_universe.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [sym for sym, _ in scored_universe[:top_k]]
            top_candidate_pool = set(sym for sym, _ in scored_universe[:top_k + 2])

            new_leaders = set()
            for sym in current_held_leaders:
                if sym in top_candidate_pool and sym not in stopped_out:
                    new_leaders.add(sym)

            for sym in top_candidates:
                if len(new_leaders) < top_k and sym not in stopped_out:
                    new_leaders.add(sym)

            current_held_leaders = new_leaders

            target_weights: Dict[str, float] = {s: 0.0 for s in active_symbols}
            if current_held_leaders:
                target_w = min(0.35, max_leverage / len(current_held_leaders))
                for sym in current_held_leaders:
                    target_weights[sym] = target_w

            for sym in active_symbols:
                if sym not in latest_quotes:
                    continue
                curr_w = current_portfolio.get_position_weight(sym)
                targ_w = target_weights[sym]

                if abs(targ_w - curr_w) > 0.06:
                    fill = broker.execute_target_weight(sym, targ_w, quote=latest_quotes[sym])
                    if fill is not None:
                        highest_price_tracking[sym] = fill.price
                        trade_ledger.append({
                            "timestamp": int(ts),
                            "symbol": sym,
                            "side": fill.side.value.upper(),
                            "quantity": round(fill.quantity, 4),
                            "price": round(fill.price, 4),
                            "fee": round(fill.fee, 4),
                            "target_weight": round(targ_w * 100, 2)
                        })

        post_portfolio = broker.get_portfolio_state()
        curr_weights = {s: round(post_portfolio.get_position_weight(s) * 100, 2) for s in active_symbols}
        equity_records.append({
            "time": int(ts),
            "equity": round(post_portfolio.equity, 2),
            "benchmark": round(bnh_val, 2),
            "cash": round(post_portfolio.cash, 2),
            "weights": curr_weights
        })

    final_state = broker.get_portfolio_state()
    final_equity = final_state.equity
    net_profit = final_equity - initial_cash
    return_pct = (net_profit / initial_cash) * 100.0
    total_fees = sum(f.fee for f in broker.fill_history)

    bnh_start_sum = sum(asset_bars[sym][0].close for sym in active_symbols)
    bnh_end_sum = sum(asset_bars[sym][-1].close for sym in active_symbols)
    bnh_return_pct = ((bnh_end_sum - bnh_start_sum) / bnh_start_sum) * 100.0

    # Current positions
    positions_summary = []
    for sym, pos in final_state.positions.items():
        if abs(pos.quantity) > 0.001:
            val = pos.quantity * pos.current_price
            w = (val / final_equity) * 100.0 if final_equity > 0 else 0.0
            positions_summary.append({
                "symbol": sym,
                "quantity": round(pos.quantity, 4),
                "current_price": round(pos.current_price, 2),
                "position_value": round(val, 2),
                "portfolio_weight": round(w, 2)
            })

    return {
        "kpis": {
            "initial_cash": initial_cash,
            "final_equity": round(final_equity, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(return_pct, 2),
            "benchmark_return_pct": round(bnh_return_pct, 2),
            "alpha_pct": round(return_pct - bnh_return_pct, 2),
            "total_trades": len(broker.fill_history),
            "total_fees": round(total_fees, 2),
            "cash_balance": round(final_state.cash, 2)
        },
        "equity_curve": equity_records,
        "trades": trade_ledger,
        "positions": positions_summary
    }
