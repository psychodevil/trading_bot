"""
Multi-timeframe and sampling engine supporting fixed time views (1m, 1h, 1d),
information-driven bars (tick, volume, dollar), and stochastic/randomized time sampling.
"""

from __future__ import annotations
from enum import Enum
import math
import random
from typing import List, Optional, Callable

from trading_bot.core.events import Bar, MarketQuote


class SamplingScheme(str, Enum):
    FIXED_TIME = "fixed_time"          # Standard time bars (e.g. 60s, 3600s)
    RANDOM_POISSON = "random_poisson"  # Poisson arrival intervals dt ~ Exp(lambda)
    RANDOM_UNIFORM = "random_uniform"  # Uniform random intervals [min_dt, max_dt]
    TICK_BAR = "tick_bar"              # Fixed N ticks per bar
    VOLUME_BAR = "volume_bar"          # Fixed traded volume per bar
    DOLLAR_BAR = "dollar_bar"          # Fixed dollar turnover per bar


class MultiTimeframeResampler:
    """
    Resamples high-resolution ticks or quotes into flexible timeframes and sampling modes.
    """

    def __init__(
        self,
        scheme: SamplingScheme = SamplingScheme.FIXED_TIME,
        fixed_seconds: float = 60.0,
        poisson_mean_seconds: float = 60.0,
        random_range_seconds: tuple[float, float] = (30.0, 120.0),
        tick_count_threshold: int = 50,
        volume_threshold: float = 1000.0,
        dollar_threshold: float = 100000.0,
        seed: Optional[int] = None
    ):
        self.scheme = scheme
        self.fixed_seconds = fixed_seconds
        self.poisson_mean_seconds = poisson_mean_seconds
        self.random_range_seconds = random_range_seconds
        self.tick_count_threshold = tick_count_threshold
        self.volume_threshold = volume_threshold
        self.dollar_threshold = dollar_threshold
        self.rng = random.Random(seed) if seed is not None else random

    def resample_quotes(self, quotes: List[MarketQuote]) -> List[Bar]:
        """Resamples quote sequence according to the configured sampling scheme."""
        if not quotes:
            return []

        if self.scheme == SamplingScheme.FIXED_TIME:
            return self._resample_fixed_time(quotes, self.fixed_seconds)
        elif self.scheme == SamplingScheme.RANDOM_POISSON:
            return self._resample_random_poisson(quotes)
        elif self.scheme == SamplingScheme.RANDOM_UNIFORM:
            return self._resample_random_uniform(quotes)
        elif self.scheme == SamplingScheme.TICK_BAR:
            return self._resample_tick_bars(quotes)
        elif self.scheme == SamplingScheme.VOLUME_BAR:
            return self._resample_volume_bars(quotes)
        elif self.scheme == SamplingScheme.DOLLAR_BAR:
            return self._resample_dollar_bars(quotes)
        else:
            raise ValueError(f"Unknown sampling scheme: {self.scheme}")

    def _resample_fixed_time(self, quotes: List[MarketQuote], interval: float) -> List[Bar]:
        bars: List[Bar] = []
        if not quotes:
            return bars

        cur_start = quotes[0].timestamp - (quotes[0].timestamp % interval)
        cur_end = cur_start + interval
        o = h = l = c = quotes[0].mid_price
        vol = 0.0
        n_trades = 0
        sym = quotes[0].symbol

        for q in quotes:
            if q.timestamp >= cur_end:
                bars.append(Bar(
                    timestamp=cur_start,
                    symbol=sym,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    timeframe_seconds=interval,
                    trades_count=n_trades
                ))
                cur_start = q.timestamp - (q.timestamp % interval)
                cur_end = cur_start + interval
                o = h = l = c = q.mid_price
                vol = 0.0
                n_trades = 0

            p = q.mid_price
            h = max(h, p)
            l = min(l, p)
            c = p
            vol += (q.bid_size + q.ask_size) * 0.1
            n_trades += 1

        if n_trades > 0:
            bars.append(Bar(
                timestamp=cur_start,
                symbol=sym,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                timeframe_seconds=interval,
                trades_count=n_trades
            ))
        return bars

    def _resample_random_poisson(self, quotes: List[MarketQuote]) -> List[Bar]:
        bars: List[Bar] = []
        if not quotes:
            return bars

        cur_start = quotes[0].timestamp
        next_dt = self.rng.expovariate(1.0 / self.poisson_mean_seconds)
        next_boundary = cur_start + max(1.0, next_dt)

        o = h = l = c = quotes[0].mid_price
        vol = 0.0
        n_trades = 0
        sym = quotes[0].symbol

        for q in quotes:
            if q.timestamp >= next_boundary:
                bars.append(Bar(
                    timestamp=cur_start,
                    symbol=sym,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    timeframe_seconds=max(1.0, q.timestamp - cur_start),
                    trades_count=n_trades
                ))
                cur_start = q.timestamp
                next_dt = self.rng.expovariate(1.0 / self.poisson_mean_seconds)
                next_boundary = cur_start + max(1.0, next_dt)
                o = h = l = c = q.mid_price
                vol = 0.0
                n_trades = 0

            p = q.mid_price
            h = max(h, p)
            l = min(l, p)
            c = p
            vol += (q.bid_size + q.ask_size) * 0.1
            n_trades += 1

        if n_trades > 0:
            bars.append(Bar(
                timestamp=cur_start,
                symbol=sym,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                timeframe_seconds=max(1.0, quotes[-1].timestamp - cur_start),
                trades_count=n_trades
            ))
        return bars

    def _resample_random_uniform(self, quotes: List[MarketQuote]) -> List[Bar]:
        bars: List[Bar] = []
        if not quotes:
            return bars

        min_dt, max_dt = self.random_range_seconds
        cur_start = quotes[0].timestamp
        next_dt = self.rng.uniform(min_dt, max_dt)
        next_boundary = cur_start + next_dt

        o = h = l = c = quotes[0].mid_price
        vol = 0.0
        n_trades = 0
        sym = quotes[0].symbol

        for q in quotes:
            if q.timestamp >= next_boundary:
                bars.append(Bar(
                    timestamp=cur_start,
                    symbol=sym,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    timeframe_seconds=max(1.0, q.timestamp - cur_start),
                    trades_count=n_trades
                ))
                cur_start = q.timestamp
                next_dt = self.rng.uniform(min_dt, max_dt)
                next_boundary = cur_start + next_dt
                o = h = l = c = q.mid_price
                vol = 0.0
                n_trades = 0

            p = q.mid_price
            h = max(h, p)
            l = min(l, p)
            c = p
            vol += (q.bid_size + q.ask_size) * 0.1
            n_trades += 1

        if n_trades > 0:
            bars.append(Bar(
                timestamp=cur_start,
                symbol=sym,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                timeframe_seconds=max(1.0, quotes[-1].timestamp - cur_start),
                trades_count=n_trades
            ))
        return bars

    def _resample_tick_bars(self, quotes: List[MarketQuote]) -> List[Bar]:
        bars: List[Bar] = []
        if not quotes:
            return bars

        cur_start = quotes[0].timestamp
        o = h = l = c = quotes[0].mid_price
        vol = 0.0
        n_trades = 0
        sym = quotes[0].symbol

        for q in quotes:
            p = q.mid_price
            h = max(h, p)
            l = min(l, p)
            c = p
            vol += (q.bid_size + q.ask_size) * 0.1
            n_trades += 1

            if n_trades >= self.tick_count_threshold:
                bars.append(Bar(
                    timestamp=cur_start,
                    symbol=sym,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    timeframe_seconds=max(1.0, q.timestamp - cur_start),
                    trades_count=n_trades
                ))
                cur_start = q.timestamp
                o = h = l = c = q.mid_price
                vol = 0.0
                n_trades = 0

        if n_trades > 0:
            bars.append(Bar(
                timestamp=cur_start,
                symbol=sym,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                timeframe_seconds=max(1.0, quotes[-1].timestamp - cur_start),
                trades_count=n_trades
            ))
        return bars

    def _resample_volume_bars(self, quotes: List[MarketQuote]) -> List[Bar]:
        bars: List[Bar] = []
        if not quotes:
            return bars

        cur_start = quotes[0].timestamp
        o = h = l = c = quotes[0].mid_price
        vol = 0.0
        n_trades = 0
        sym = quotes[0].symbol

        for q in quotes:
            p = q.mid_price
            h = max(h, p)
            l = min(l, p)
            c = p
            trade_vol = (q.bid_size + q.ask_size) * 0.1
            vol += trade_vol
            n_trades += 1

            if vol >= self.volume_threshold:
                bars.append(Bar(
                    timestamp=cur_start,
                    symbol=sym,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    timeframe_seconds=max(1.0, q.timestamp - cur_start),
                    trades_count=n_trades
                ))
                cur_start = q.timestamp
                o = h = l = c = q.mid_price
                vol = 0.0
                n_trades = 0

        if n_trades > 0:
            bars.append(Bar(
                timestamp=cur_start,
                symbol=sym,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                timeframe_seconds=max(1.0, quotes[-1].timestamp - cur_start),
                trades_count=n_trades
            ))
        return bars

    def _resample_dollar_bars(self, quotes: List[MarketQuote]) -> List[Bar]:
        bars: List[Bar] = []
        if not quotes:
            return bars

        cur_start = quotes[0].timestamp
        o = h = l = c = quotes[0].mid_price
        dollar_turnover = 0.0
        vol = 0.0
        n_trades = 0
        sym = quotes[0].symbol

        for q in quotes:
            p = q.mid_price
            h = max(h, p)
            l = min(l, p)
            c = p
            trade_vol = (q.bid_size + q.ask_size) * 0.1
            vol += trade_vol
            dollar_turnover += trade_vol * p
            n_trades += 1

            if dollar_turnover >= self.dollar_threshold:
                bars.append(Bar(
                    timestamp=cur_start,
                    symbol=sym,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    timeframe_seconds=max(1.0, q.timestamp - cur_start),
                    trades_count=n_trades
                ))
                cur_start = q.timestamp
                o = h = l = c = q.mid_price
                vol = 0.0
                dollar_turnover = 0.0
                n_trades = 0

        if n_trades > 0:
            bars.append(Bar(
                timestamp=cur_start,
                symbol=sym,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                timeframe_seconds=max(1.0, quotes[-1].timestamp - cur_start),
                trades_count=n_trades
            ))
        return bars

