"""
REST API Endpoints for QuantumAlpha Web Dashboard & Simulation Engine.
"""

from __future__ import annotations
import json
import os
import sys
import math
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        csv_path = f"data/historical/{safe_sym}_1h_1y.csv"
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
    csv_path = f"data/historical/{safe_sym}_1h_1y.csv"
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
    max_leverage: float = 1.25,
    selected_symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes a fast, strictly causal walk-forward portfolio simulation and returns results.
    """
    symbols = selected_symbols or [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "LLY", "XOM", "GLD", "SLV", "BTC-USD", "ETH-USD", "SOL-USD"
    ]

    asset_bars: Dict[str, List[Bar]] = {}
    instruments: Dict[str, Any] = {}
    broker = SimulatedBroker(initial_cash=initial_cash)

    for sym in symbols:
        safe_sym = sym.replace('^', '').replace('=', '_')
        csv_path = f"data/historical/{safe_sym}_1h_1y.csv"
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
    last_rebalance_price: Dict[str, float] = {sym: 0.0 for sym in active_symbols}

    equity_records: List[Dict[str, Any]] = []
    trade_ledger: List[Dict[str, Any]] = []
    initial_prices = {sym: asset_bars[sym][0].close for sym in active_symbols}

    for ts in unique_timestamps:
        tick_events = events_by_ts[ts]
        updated_symbols = set()

        for sym, bar in tick_events:
            feature_trackers[sym].update(bar)
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

        ready_symbols = [s for s in active_symbols if feature_trackers[s].count >= 40 and s in latest_quotes]
        if len(ready_symbols) < len(active_symbols) * 0.5:
            equity_records.append({
                "time": int(ts),
                "equity": round(current_eq, 2),
                "benchmark": round(bnh_val, 2),
                "cash": round(current_portfolio.cash, 2),
                "weights": {s: 0.0 for s in active_symbols}
            })
            continue

        # Score assets in O(1)
        asset_scores = {}
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
                    score = -0.0080
                else:
                    score = 0.0010

            ir = score / max(1e-4, base_std)
            asset_scores[sym] = (score, base_std, ir)

        positive_assets = [(sym, score, std, ir) for sym, (score, std, ir) in asset_scores.items() if score > 0]
        positive_assets.sort(key=lambda x: x[3], reverse=True)

        target_weights: Dict[str, float] = {s: 0.0 for s in active_symbols}
        allocated_leverage = 0.0

        for sym, score, std, ir in positive_assets:
            if allocated_leverage >= max_leverage:
                break
            vol_target_w = min(0.20, 0.045 / max(0.01, std))
            alloc_w = min(vol_target_w, max_leverage - allocated_leverage)
            target_weights[sym] = alloc_w
            allocated_leverage += alloc_w

        for sym in updated_symbols:
            current_w = current_portfolio.get_position_weight(sym)
            target_w = target_weights[sym]

            if abs(target_w - current_w) < 0.045 and target_w > 0:
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
                    "timestamp": int(ts),
                    "symbol": sym,
                    "side": fill.side.value.upper(),
                    "quantity": round(fill.quantity, 4),
                    "price": round(fill.price, 4),
                    "fee": round(fill.fee, 4),
                    "target_weight": round(target_w * 100, 2)
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

