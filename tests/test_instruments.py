"""
Unit tests for Financial Instruments, Vehicle specifications, and Black-Scholes Greeks.
"""

import math
import unittest

from trading_bot.core.instruments import (
    AssetClass, Stock, CryptoSpot, CryptoPerp, ForexPair, FuturesContract, OptionContract, CommodityAsset
)
from trading_bot.core.math_utils import black_scholes_price, black_scholes_greeks, implied_volatility


class TestInstruments(unittest.TestCase):

    def test_stock_specifications(self):
        stock = Stock(symbol="AAPL", tick_size=0.01, lot_size=1.0, taker_fee_rate=0.0005)
        self.assertEqual(stock.asset_class, AssetClass.STOCK)
        self.assertEqual(stock.round_price(150.1234), 150.12)
        self.assertEqual(stock.round_quantity(12.7), 13.0)
        self.assertEqual(stock.notional_value(10, 150.0), 1500.0)
        self.assertEqual(stock.margin_required(10, 150.0), 750.0) # 50% margin
        self.assertAlmostEqual(stock.calculate_transaction_fee(10, 150.0), 0.75, places=4)

    def test_crypto_perp_funding(self):
        perp = CryptoPerp(symbol="BTC/USDT-PERP", tick_size=0.1, lot_size=0.001)
        self.assertEqual(perp.asset_class, AssetClass.CRYPTO_PERP)
        # Long 2 BTC at $50,000, funding rate +0.0001 (1 bp) -> Long pays $10
        cashflow = perp.calculate_funding_payment(quantity=2.0, mark_price=50000.0, funding_rate_per_interval=0.0001)
        self.assertEqual(cashflow, -10.0)
        # Short 2 BTC at $50,000, funding rate +0.0001 -> Short receives $10
        cashflow_short = perp.calculate_funding_payment(quantity=-2.0, mark_price=50000.0, funding_rate_per_interval=0.0001)
        self.assertEqual(cashflow_short, 10.0)

    def test_forex_specs_and_swap(self):
        forex = ForexPair(symbol="EUR/USD", pip_size=0.0001, lot_size=1000.0, swap_long_annual=-0.02)
        self.assertEqual(forex.asset_class, AssetClass.FOREX)
        self.assertEqual(forex.round_price(1.08543), 1.0854)
        # Long 10,000 EUR/USD for 1 year with -2% swap rate
        swap_cost = forex.calculate_swap_cost(quantity=10000.0, price=1.10, elapsed_seconds=365 * 86400)
        self.assertAlmostEqual(swap_cost, -220.0, places=2)

    def test_option_black_scholes_and_greeks(self):
        spot = 100.0
        strike = 100.0
        time_to_exp = 1.0 # 1 year
        rate = 0.05
        vol = 0.20

        # European Call price
        call_price = black_scholes_price(spot, strike, time_to_exp, rate, vol, is_call=True)
        # Theoretical BS Call price for S=100, K=100, T=1, r=0.05, vol=0.20 is approx 10.4506
        self.assertAlmostEqual(call_price, 10.4506, places=2)

        # European Put price via Put-Call Parity: C - P = S - K * exp(-r*T)
        put_price = black_scholes_price(spot, strike, time_to_exp, rate, vol, is_call=False)
        parity_diff = (call_price - put_price) - (spot - strike * math.exp(-rate * time_to_exp))
        self.assertAlmostEqual(parity_diff, 0.0, places=4)

        # Greeks
        greeks = black_scholes_greeks(spot, strike, time_to_exp, rate, vol, is_call=True)
        self.assertGreater(greeks["delta"], 0.5) # Call delta > 0.5 for ATM with r > 0
        self.assertGreater(greeks["gamma"], 0.0)
        self.assertGreater(greeks["vega"], 0.0)
        self.assertLess(greeks["theta"], 0.0) # Theta decay is negative

        # Implied Volatility inversion
        solved_iv = implied_volatility(target_price=call_price, spot=spot, strike=strike, time_to_expiry=time_to_exp, rate=rate, is_call=True)
        self.assertIsNotNone(solved_iv)
        self.assertAlmostEqual(solved_iv, 0.20, places=3)

    def test_option_contract_wrapper(self):
        opt = OptionContract(
            symbol="SPY_C100",
            underlying_symbol="SPY",
            strike=100.0,
            expiry_timestamp=1700000000.0 + 30 * 86400,
            is_call=True
        )
        p = opt.price(underlying_spot=105.0, current_timestamp=1700000000.0, rate=0.04, volatility=0.25)
        self.assertGreater(p, 5.0)


if __name__ == "__main__":
    unittest.main()

