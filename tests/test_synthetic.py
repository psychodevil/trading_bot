"""
Unit tests for Synthetic Market Data Generators (GBM, Merton Jump, Heston, OU).
"""

import unittest

from trading_bot.data.synthetic import (
    generate_geometric_brownian_motion,
    generate_merton_jump_diffusion,
    generate_heston_stochastic_vol,
    generate_ornstein_uhlenbeck,
    convert_ticks_to_quotes,
    convert_quotes_to_bars
)


class TestSyntheticGenerators(unittest.TestCase):

    def test_gbm_generator(self):
        path = generate_geometric_brownian_motion(s0=100.0, mu_annual=0.10, sigma_annual=0.20, n_steps=200, seed=123)
        self.assertEqual(len(path), 201)
        self.assertEqual(path[0][1], 100.0)
        # All prices must be strictly positive
        self.assertTrue(all(p > 0 for _, p in path))

    def test_merton_jump_diffusion(self):
        path = generate_merton_jump_diffusion(s0=50.0, jump_lambda_annual=20.0, n_steps=200, seed=123)
        self.assertEqual(len(path), 201)
        self.assertTrue(all(p > 0 for _, p in path))

    def test_heston_stochastic_vol(self):
        path = generate_heston_stochastic_vol(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, n_steps=200, seed=123)
        self.assertEqual(len(path), 201)
        for _, s, v in path:
            self.assertGreater(s, 0.0)
            self.assertGreaterEqual(v, 0.0)

    def test_ornstein_uhlenbeck(self):
        path = generate_ornstein_uhlenbeck(s0=0.0, mean_target_theta=0.0, mean_reversion_kappa=5.0, n_steps=200, seed=123)
        self.assertEqual(len(path), 201)
        # Mean should remain bounded near zero
        mean_val = sum(p for _, p in path) / len(path)
        self.assertAlmostEqual(mean_val, 0.0, delta=0.5)

    def test_quote_and_bar_conversion(self):
        path = generate_geometric_brownian_motion(s0=100.0, n_steps=120, dt_seconds=1.0, seed=123)
        quotes = convert_ticks_to_quotes(path, symbol="SYM", spread_bps=4.0)
        self.assertEqual(len(quotes), 121)
        self.assertTrue(all(q.bid < q.ask for q in quotes))

        bars = convert_quotes_to_bars(quotes, timeframe_seconds=60.0)
        self.assertGreaterEqual(len(bars), 2)
        for b in bars:
            self.assertGreaterEqual(b.high, b.low)
            self.assertGreaterEqual(b.high, b.open)
            self.assertGreaterEqual(b.high, b.close)


if __name__ == "__main__":
    unittest.main()

