"""
Instrument and Financial Vehicle definitions across Stocks, Crypto Spot/Perpetuals,
Forex, Futures, Options, and Commodities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Dict, Any

from trading_bot.core.math_utils import black_scholes_price, black_scholes_greeks, implied_volatility


class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    FOREX = "forex"
    FUTURES = "futures"
    OPTION = "option"
    COMMODITY = "commodity"


@dataclass
class Instrument:
    """
    Base instrument definition providing universal execution and margin semantics.
    """
    symbol: str
    asset_class: AssetClass
    quote_currency: str = "USD"
    tick_size: float = 0.01
    lot_size: float = 1.0
    multiplier: float = 1.0
    margin_requirement: float = 1.0  # 1.0 = fully funded (1x leverage), 0.05 = 20x leverage
    maker_fee_rate: float = 0.0002   # 2 bps
    taker_fee_rate: float = 0.0005   # 5 bps
    borrow_rate_annual: float = 0.03 # 3% annual cost to borrow for shorting

    def round_price(self, price: float) -> float:
        """Round price to the instrument's minimum tick size."""
        if self.tick_size <= 0:
            return price
        steps = round(price / self.tick_size)
        raw = steps * self.tick_size
        decimals = max(0, -int(math.floor(math.log10(self.tick_size) + 1e-9))) if self.tick_size < 1.0 else 0
        return round(raw, decimals)

    def round_quantity(self, quantity: float) -> float:
        """Round quantity to the instrument's minimum lot size."""
        if self.lot_size <= 0:
            return quantity
        steps = round(quantity / self.lot_size)
        raw = steps * self.lot_size
        decimals = max(0, -int(math.floor(math.log10(self.lot_size) + 1e-9))) if self.lot_size < 1.0 else 0
        return round(raw, decimals)

    def notional_value(self, quantity: float, price: float) -> float:
        """Compute the total notional cash value of a position."""
        return abs(quantity) * price * self.multiplier

    def margin_required(self, quantity: float, price: float) -> float:
        """Compute the required initial/maintenance margin to hold this position."""
        return self.notional_value(quantity, price) * self.margin_requirement

    def calculate_transaction_fee(self, quantity: float, price: float, is_taker: bool = True) -> float:
        """Calculate linear execution exchange fee."""
        rate = self.taker_fee_rate if is_taker else self.maker_fee_rate
        return self.notional_value(quantity, price) * rate

    def calculate_borrow_cost(self, quantity: float, price: float, elapsed_seconds: float) -> float:
        """Calculate borrowing fee accrued for short positions over elapsed time."""
        if quantity >= 0 or self.borrow_rate_annual <= 0:
            return 0.0
        annual_fraction = elapsed_seconds / (365.0 * 86400.0)
        return self.notional_value(quantity, price) * self.borrow_rate_annual * annual_fraction


@dataclass
class Stock(Instrument):
    """
    Equities / Stock vehicle with dividend yield and shorting availability.
    """
    asset_class: AssetClass = field(default=AssetClass.STOCK, init=False)
    dividend_yield_annual: float = 0.015
    is_shortable: bool = True
    margin_requirement: float = 0.5  # Reg-T 50% initial margin standard in US


@dataclass
class CryptoSpot(Instrument):
    """
    Crypto Spot instrument (e.g. BTC/USDT) with 24/7 trading and fractional lots.
    """
    asset_class: AssetClass = field(default=AssetClass.CRYPTO_SPOT, init=False)
    base_asset: str = "BTC"
    quote_currency: str = "USDT"
    tick_size: float = 0.01
    lot_size: float = 0.0001
    margin_requirement: float = 1.0


@dataclass
class CryptoPerp(Instrument):
    """
    Crypto Perpetual Contract with dynamic periodic funding rate settlement.
    """
    asset_class: AssetClass = field(default=AssetClass.CRYPTO_PERP, init=False)
    base_asset: str = "BTC"
    quote_currency: str = "USDT"
    tick_size: float = 0.1
    lot_size: float = 0.001
    margin_requirement: float = 0.05  # Up to 20x leverage
    funding_interval_seconds: float = 8 * 3600.0  # 8 hours

    def calculate_funding_payment(self, quantity: float, mark_price: float, funding_rate_per_interval: float) -> float:
        """
        Positive funding rate: Longs pay shorts (cash outflow for long, inflow for short).
        Returns net cashflow (positive means received cash, negative means paid fee).
        """
        notional = quantity * mark_price * self.multiplier
        return -notional * funding_rate_per_interval


@dataclass
class ForexPair(Instrument):
    """
    Foreign Exchange pair with pip conventions and overnight interest swap rates.
    """
    asset_class: AssetClass = field(default=AssetClass.FOREX, init=False)
    base_currency: str = "EUR"
    quote_currency: str = "USD"
    pip_size: float = 0.0001
    tick_size: float = 0.0001
    lot_size: float = 1000.0   # 1 micro-lot = 1,000 units
    margin_requirement: float = 0.02 # 50:1 leverage standard
    swap_long_annual: float = -0.015 # Swap cost for long EUR/USD
    swap_short_annual: float = 0.005 # Swap credit for short EUR/USD

    def calculate_swap_cost(self, quantity: float, price: float, elapsed_seconds: float) -> float:
        """Calculate interest differential carry cost/benefit."""
        if quantity == 0:
            return 0.0
        annual_fraction = elapsed_seconds / (365.0 * 86400.0)
        swap_rate = self.swap_long_annual if quantity > 0 else self.swap_short_annual
        # Net swap cashflow (positive is credit, negative is debit)
        return self.notional_value(quantity, price) * swap_rate * annual_fraction


@dataclass
class FuturesContract(Instrument):
    """
    Commodity or Index Futures with contract multiplier and expiration date.
    """
    asset_class: AssetClass = field(default=AssetClass.FUTURES, init=False)
    underlying_symbol: str = "ES"
    multiplier: float = 50.0  # e.g. E-mini S&P is $50 per index point
    expiry_timestamp: Optional[float] = None
    margin_requirement: float = 0.10


@dataclass
class OptionContract(Instrument):
    """
    Option Contract (Call or Put) with analytical pricing, Greeks, and theta decay.
    """
    asset_class: AssetClass = field(default=AssetClass.OPTION, init=False)
    underlying_symbol: str = "SPY"
    strike: float = 100.0
    expiry_timestamp: float = 0.0
    is_call: bool = True
    is_european: bool = True
    multiplier: float = 100.0  # Standard 100 shares per option contract
    margin_requirement: float = 0.20

    def time_to_expiry_years(self, current_timestamp: float) -> float:
        """Compute remaining time to expiration in fraction of years."""
        diff_seconds = max(0.0, self.expiry_timestamp - current_timestamp)
        return diff_seconds / (365.0 * 86400.0)

    def price(
        self,
        underlying_spot: float,
        current_timestamp: float,
        rate: float = 0.04,
        volatility: float = 0.20,
        dividend_yield: float = 0.0
    ) -> float:
        """Compute theoretical Black-Scholes premium."""
        t_years = self.time_to_expiry_years(current_timestamp)
        return black_scholes_price(
            spot=underlying_spot,
            strike=self.strike,
            time_to_expiry=t_years,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            is_call=self.is_call
        )

    def greeks(
        self,
        underlying_spot: float,
        current_timestamp: float,
        rate: float = 0.04,
        volatility: float = 0.20,
        dividend_yield: float = 0.0
    ) -> Dict[str, float]:
        """Compute Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho)."""
        t_years = self.time_to_expiry_years(current_timestamp)
        return black_scholes_greeks(
            spot=underlying_spot,
            strike=self.strike,
            time_to_expiry=t_years,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            is_call=self.is_call
        )

    def solve_iv(
        self,
        market_option_price: float,
        underlying_spot: float,
        current_timestamp: float,
        rate: float = 0.04,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """Solve for implied volatility given observed option market price."""
        t_years = self.time_to_expiry_years(current_timestamp)
        return implied_volatility(
            target_price=market_option_price,
            spot=underlying_spot,
            strike=self.strike,
            time_to_expiry=t_years,
            rate=rate,
            dividend_yield=dividend_yield,
            is_call=self.is_call
        )


@dataclass
class CommodityAsset(Instrument):
    """
    Physical or Spot Commodity with physical storage / holding carry costs.
    """
    asset_class: AssetClass = field(default=AssetClass.COMMODITY, init=False)
    storage_cost_annual: float = 0.015 # 1.5% annual storage/insurance fee
