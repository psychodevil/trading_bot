"""
Synthetic Market Data Generators for Monte Carlo Simulation and Stress-Testing.
Includes Geometric Brownian Motion, Merton Jump-Diffusion, Heston Stochastic Volatility,
Ornstein-Uhlenbeck (Mean-Reverting), and GARCH(1,1) dynamic volatility processes.
"""

from __future__ import annotations
import math
import random
import time
from typing import List, Tuple, Optional

from trading_bot.core.events import Bar, MarketQuote


def generate_geometric_brownian_motion(
    s0: float = 100.0,
    mu_annual: float = 0.08,
    sigma_annual: float = 0.20,
    dt_seconds: float = 60.0,
    n_steps: int = 1000,
    start_timestamp: float = 1700000000.0,
    symbol: str = "SYNTH_STOCK",
    seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Simulates price path under Geometric Brownian Motion (GBM):
    S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    Returns list of (timestamp, price).
    """
    rng = random.Random(seed) if seed is not None else random
    seconds_in_year = 365.0 * 86400.0
    dt = dt_seconds / seconds_in_year
    drift = (mu_annual - 0.5 * sigma_annual * sigma_annual) * dt
    vol_step = sigma_annual * math.sqrt(dt)

    current_price = s0
    current_time = start_timestamp
    path = [(current_time, current_price)]

    for _ in range(n_steps):
        z = rng.gauss(0.0, 1.0)
        current_price *= math.exp(drift + vol_step * z)
        current_time += dt_seconds
        path.append((current_time, current_price))

    return path


def generate_merton_jump_diffusion(
    s0: float = 100.0,
    mu_annual: float = 0.08,
    sigma_annual: float = 0.20,
    jump_lambda_annual: float = 10.0, # Average 10 jumps per year
    jump_mean: float = -0.02,         # Average -2% jump size
    jump_std: float = 0.04,           # Jump size volatility
    dt_seconds: float = 60.0,
    n_steps: int = 1000,
    start_timestamp: float = 1700000000.0,
    seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Simulates Merton Jump-Diffusion process:
    Continuous Brownian motion plus discrete compound Poisson jumps.
    """
    rng = random.Random(seed) if seed is not None else random
    seconds_in_year = 365.0 * 86400.0
    dt = dt_seconds / seconds_in_year
    drift = (mu_annual - 0.5 * sigma_annual * sigma_annual) * dt
    vol_step = sigma_annual * math.sqrt(dt)
    jump_prob = jump_lambda_annual * dt

    current_price = s0
    current_time = start_timestamp
    path = [(current_time, current_price)]

    for _ in range(n_steps):
        z = rng.gauss(0.0, 1.0)
        continuous_return = drift + vol_step * z
        jump_return = 0.0
        if rng.random() < jump_prob:
            jump_return = rng.gauss(jump_mean, jump_std)

        current_price *= math.exp(continuous_return + jump_return)
        current_time += dt_seconds
        path.append((current_time, current_price))

    return path


def generate_heston_stochastic_vol(
    s0: float = 100.0,
    v0: float = 0.04,         # Initial variance (20% vol squared)
    mu_annual: float = 0.08,
    kappa: float = 2.0,       # Rate of mean reversion of variance
    theta: float = 0.04,      # Long-term variance target
    xi: float = 0.3,          # Volatility of variance
    rho: float = -0.7,        # Correlation between asset return and volatility shocks (leverage effect)
    dt_seconds: float = 60.0,
    n_steps: int = 1000,
    start_timestamp: float = 1700000000.0,
    seed: Optional[int] = None
) -> List[Tuple[float, float, float]]:
    """
    Simulates Heston Stochastic Volatility model:
    dS = mu * S * dt + sqrt(v) * S * dW_S
    dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v
    Returns list of (timestamp, spot_price, variance).
    """
    rng = random.Random(seed) if seed is not None else random
    seconds_in_year = 365.0 * 86400.0
    dt = dt_seconds / seconds_in_year
    sqrt_dt = math.sqrt(dt)

    current_s = s0
    current_v = v0
    current_time = start_timestamp
    path = [(current_time, current_s, current_v)]

    for _ in range(n_steps):
        z1 = rng.gauss(0.0, 1.0)
        z2_indep = rng.gauss(0.0, 1.0)
        z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * z2_indep

        # Full truncation Euler scheme for non-negative variance
        v_pos = max(0.0, current_v)
        sqrt_v = math.sqrt(v_pos)

        # Asset price step
        drift = (mu_annual - 0.5 * v_pos) * dt
        current_s *= math.exp(drift + sqrt_v * sqrt_dt * z1)

        # Variance step
        dv = kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z2
        current_v = max(1e-6, current_v + dv)

        current_time += dt_seconds
        path.append((current_time, current_s, current_v))

    return path


def generate_ornstein_uhlenbeck(
    s0: float = 0.0,
    mean_target_theta: float = 0.0,
    mean_reversion_kappa: float = 5.0, # Reversion speed
    volatility_sigma: float = 0.5,
    dt_seconds: float = 60.0,
    n_steps: int = 1000,
    start_timestamp: float = 1700000000.0,
    seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Simulates Ornstein-Uhlenbeck mean-reverting spread process:
    dX_t = kappa * (theta - X_t) * dt + sigma * dW_t
    """
    rng = random.Random(seed) if seed is not None else random
    seconds_in_year = 365.0 * 86400.0
    dt = dt_seconds / seconds_in_year
    decay = math.exp(-mean_reversion_kappa * dt)
    variance = (volatility_sigma ** 2) / (2.0 * mean_reversion_kappa) * (1.0 - math.exp(-2.0 * mean_reversion_kappa * dt))
    sd = math.sqrt(max(1e-12, variance))

    current_x = s0
    current_time = start_timestamp
    path = [(current_time, current_x)]

    for _ in range(n_steps):
        z = rng.gauss(0.0, 1.0)
        current_x = current_x * decay + mean_target_theta * (1.0 - decay) + sd * z
        current_time += dt_seconds
        path.append((current_time, current_x))

    return path


def convert_ticks_to_quotes(
    price_path: List[Tuple[float, float]],
    symbol: str,
    spread_bps: float = 5.0, # 5 basis points half-spread = 0.05%
    depth: float = 1000.0
) -> List[MarketQuote]:
    """
    Converts (timestamp, mid_price) stream into realistic MarketQuote objects with bid/ask spread.
    """
    quotes = []
    spread_multiplier = spread_bps / 10000.0

    for ts, mid in price_path:
        half_spread = mid * spread_multiplier
        bid = mid - half_spread
        ask = mid + half_spread
        quotes.append(MarketQuote(
            timestamp=ts,
            symbol=symbol,
            bid=bid,
            ask=ask,
            bid_size=depth,
            ask_size=depth,
            last_price=mid
        ))
    return quotes


def convert_quotes_to_bars(
    quotes: List[MarketQuote],
    timeframe_seconds: float = 60.0
) -> List[Bar]:
    """
    Aggregates high-frequency quotes into OHLCV candlestick bars.
    """
    if not quotes:
        return []

    bars = []
    current_bar_start = quotes[0].timestamp - (quotes[0].timestamp % timeframe_seconds)
    current_bar_end = current_bar_start + timeframe_seconds

    open_p = quotes[0].mid_price
    high_p = open_p
    low_p = open_p
    close_p = open_p
    volume = 0.0
    trades_count = 0
    symbol = quotes[0].symbol

    for q in quotes:
        if q.timestamp >= current_bar_end:
            # Finalize previous bar
            bars.append(Bar(
                timestamp=current_bar_start,
                symbol=symbol,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                timeframe_seconds=timeframe_seconds,
                trades_count=trades_count
            ))
            # Start new bar
            current_bar_start = q.timestamp - (q.timestamp % timeframe_seconds)
            current_bar_end = current_bar_start + timeframe_seconds
            open_p = q.mid_price
            high_p = open_p
            low_p = open_p
            close_p = open_p
            volume = 0.0
            trades_count = 0

        p = q.mid_price
        high_p = max(high_p, p)
        low_p = min(low_p, p)
        close_p = p
        volume += (q.bid_size + q.ask_size) * 0.1
        trades_count += 1

    # Add final bar
    if trades_count > 0:
        bars.append(Bar(
            timestamp=current_bar_start,
            symbol=symbol,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=volume,
            timeframe_seconds=timeframe_seconds,
            trades_count=trades_count
        ))

    return bars

