"""
Unit tests for Multi-Timeframe Resampler and Randomized Sampling schemes.
"""

import unittest

from trading_bot.core.events import MarketQuote
from trading_bot.data.resampler import SamplingScheme, MultiTimeframeResampler


class TestResampler(unittest.TestCase):

    def setUp(self):
        # Generate 120 synthetic quotes 1 second apart (aligned to 60s boundary)
        self.quotes = []
        base_ts = 1700000040.0
        p = 100.0
        for i in range(120):
            p += (0.05 if i % 2 == 0 else -0.04)
            self.quotes.append(MarketQuote(
                timestamp=base_ts + i,
                symbol="TEST",
                bid=p - 0.01,
                ask=p + 0.01,
                bid_size=50.0,
                ask_size=50.0,
                last_price=p
            ))

    def test_fixed_time_resampling(self):
        # 120 seconds into 60s bars -> 2 bars
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.FIXED_TIME, fixed_seconds=60.0)
        bars = resampler.resample_quotes(self.quotes)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].timeframe_seconds, 60.0)
        self.assertEqual(bars[0].trades_count, 60)

    def test_poisson_random_resampling(self):
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.RANDOM_POISSON, poisson_mean_seconds=30.0, seed=42)
        bars = resampler.resample_quotes(self.quotes)
        self.assertGreater(len(bars), 1)
        for b in bars:
            self.assertGreater(b.timeframe_seconds, 0)
            self.assertGreater(b.trades_count, 0)

    def test_tick_bar_resampling(self):
        # 120 quotes with 30 ticks per bar -> 4 bars
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.TICK_BAR, tick_count_threshold=30)
        bars = resampler.resample_quotes(self.quotes)
        self.assertEqual(len(bars), 4)
        for b in bars:
            self.assertEqual(b.trades_count, 30)

    def test_dollar_bar_resampling(self):
        resampler = MultiTimeframeResampler(scheme=SamplingScheme.DOLLAR_BAR, dollar_threshold=20000.0)
        bars = resampler.resample_quotes(self.quotes)
        self.assertGreater(len(bars), 0)


if __name__ == "__main__":
    unittest.main()
