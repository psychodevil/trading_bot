"""
Integration tests for the Backtesting Engine and HTML Reporting.
"""

import os
import tempfile
import unittest

from trading_bot.core.instruments import Stock, CryptoPerp, ForexPair
from trading_bot.data.synthetic import generate_geometric_brownian_motion, convert_ticks_to_quotes, convert_quotes_to_bars
from trading_bot.strategies.probabilistic_trend import ProbabilisticTrendStrategy
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.visualization.report_generator import ReportGenerator


class TestBacktestEngine(unittest.TestCase):

    def test_full_backtest_lifecycle(self):
        # 1. Generate market data
        stock = Stock(symbol="MSFT", tick_size=0.01, lot_size=1.0)
        ticks = generate_geometric_brownian_motion(s0=300.0, mu_annual=0.15, sigma_annual=0.25, dt_seconds=10.0, n_steps=600, seed=42)
        quotes = convert_ticks_to_quotes(ticks, symbol="MSFT")
        bars = convert_quotes_to_bars(quotes, timeframe_seconds=60.0)

        # 2. Run backtest
        strategy = ProbabilisticTrendStrategy(name="MSFT_Prob_Trend", horizon_seconds=1800.0)
        engine = BacktestEngine(initial_cash=100000.0)
        result = engine.run(strategy=strategy, instrument=stock, bars=bars, timeframe_desc="1m")

        # 3. Verify metrics & logs
        self.assertEqual(result.strategy_name, "MSFT_Prob_Trend")
        self.assertEqual(len(result.equity_curve), len(bars))
        self.assertGreater(result.metrics.final_equity, 0.0)
        self.assertGreater(result.total_decisions, 0)
        self.assertGreaterEqual(result.metrics.inaction_efficiency_pct, 0.0)

        # 4. Verify HTML Report Generation
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            html = ReportGenerator.generate_html_report([result], output_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 500)
            self.assertIn("MSFT_Prob_Trend", html)
            self.assertIn("Inaction Efficiency", html)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

