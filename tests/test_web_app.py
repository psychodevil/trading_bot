"""
Unit and Integration Tests for QuantumAlpha Web Application & REST API.
"""

import unittest
import json
import os
import sys
from io import BytesIO

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import StandaloneAppServer, create_flask_app
from web.routes.api import (
    get_market_universe_data,
    get_asset_bars_data,
    run_portfolio_simulation_api
)


class TestWebApiEndpoints(unittest.TestCase):
    """Tests the REST API data functions."""

    def test_market_universe_data(self):
        assets = get_market_universe_data()
        self.assertIsInstance(assets, list)
        self.assertGreater(len(assets), 50)
        spy = next((a for a in assets if a["symbol"] == "SPY"), None)
        self.assertIsNotNone(spy)
        self.assertEqual(spy["sector"], "Broad Market")
        self.assertIn("latest_price", spy)
        self.assertIn("return_pct", spy)

    def test_asset_bars_data(self):
        bars = get_asset_bars_data("SPY")
        self.assertIsInstance(bars, list)
        self.assertGreater(len(bars), 100)
        b0 = bars[0]
        self.assertIn("time", b0)
        self.assertIn("open", b0)
        self.assertIn("high", b0)
        self.assertIn("low", b0)
        self.assertIn("close", b0)
        self.assertIn("volume", b0)

    def test_asset_bars_nonexistent(self):
        bars = get_asset_bars_data("NONEXISTENT_XYZ")
        self.assertEqual(bars, [])

    def test_portfolio_simulation_api(self):
        res = run_portfolio_simulation_api(
            initial_cash=100000.0,
            max_leverage=1.25,
            selected_symbols=["SPY", "QQQ", "AAPL", "GLD"]
        )
        self.assertIn("kpis", res)
        self.assertIn("equity_curve", res)
        self.assertIn("trades", res)
        self.assertIn("positions", res)

        kpis = res["kpis"]
        self.assertEqual(kpis["initial_cash"], 100000.0)
        self.assertGreater(kpis["final_equity"], 0)
        self.assertIn("total_return_pct", kpis)
        self.assertIn("alpha_pct", kpis)


class TestStandaloneAppServer(unittest.TestCase):
    """Tests the Standalone WSGI & HTTP Server."""

    def _mock_request(self, path: str, method: str = "GET", body: bytes = b""):
        status_code = [None]
        headers = [{}]

        def start_response(status, response_headers):
            status_code[0] = status
            for k, v in response_headers:
                headers[0][k] = v

        environ = {
            "PATH_INFO": path,
            "REQUEST_METHOD": method,
            "wsgi.input": BytesIO(body),
            "CONTENT_LENGTH": str(len(body))
        }

        response = StandaloneAppServer.handle_request(environ, start_response)
        response_body = b"".join(response)
        return status_code[0], headers[0], response_body

    def test_index_page(self):
        status, headers, body = self._mock_request("/")
        self.assertEqual(status, "200 OK")
        self.assertIn(b"Executive Portfolio Dashboard", body)
        self.assertIn(b"QuantumAlpha", body)

    def test_portfolio_page(self):
        status, headers, body = self._mock_request("/portfolio")
        self.assertEqual(status, "200 OK")
        self.assertIn(b"Portfolio & Asset Allocation Timeline", body)

    def test_markets_page(self):
        status, headers, body = self._mock_request("/markets")
        self.assertEqual(status, "200 OK")
        self.assertIn(b"Market Universe Explorer", body)

    def test_simulator_page(self):
        status, headers, body = self._mock_request("/simulator")
        self.assertEqual(status, "200 OK")
        self.assertIn(b"Interactive Quantitative Simulation Lab", body)

    def test_trades_page(self):
        status, headers, body = self._mock_request("/trades")
        self.assertEqual(status, "200 OK")
        self.assertIn(b"Chronological Rebalancing Decision Ledger", body)

    def test_static_css(self):
        status, headers, body = self._mock_request("/static/css/style.css")
        self.assertEqual(status, "200 OK")
        self.assertEqual(headers.get("Content-Type"), "text/css")
        self.assertIn(b":root", body)

    def test_static_js(self):
        status, headers, body = self._mock_request("/static/js/app.js")
        self.assertEqual(status, "200 OK")
        self.assertEqual(headers.get("Content-Type"), "application/javascript")

    def test_api_markets_endpoint(self):
        status, headers, body = self._mock_request("/api/markets")
        self.assertEqual(status, "200 OK")
        self.assertEqual(headers.get("Content-Type"), "application/json")
        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data, list)

    def test_api_market_bars_endpoint(self):
        status, headers, body = self._mock_request("/api/market/SPY/bars")
        self.assertEqual(status, "200 OK")
        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_api_simulate_post(self):
        payload = json.dumps({
            "initial_cash": 50000,
            "max_leverage": 1.1,
            "symbols": ["SPY", "GLD"]
        }).encode("utf-8")
        status, headers, body = self._mock_request("/api/simulate", method="POST", body=payload)
        self.assertEqual(status, "200 OK")
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["kpis"]["initial_cash"], 50000.0)

    def test_404_not_found(self):
        status, headers, body = self._mock_request("/invalid_unknown_route")
        self.assertEqual(status, "404 Not Found")


if __name__ == "__main__":
    unittest.main()
