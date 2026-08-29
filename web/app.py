#!/usr/bin/env python3
"""
QuantumAlpha Web Application & REST API Server.
Supports both standard Flask (when installed) and a built-in zero-dependency WSGI/HTTP server fallback.
"""

import json
import os
import sys
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.routes.api import (
    get_market_universe_data,
    get_asset_bars_data,
    run_portfolio_simulation_api
)

# Global in-memory cache for latest simulation run
LATEST_SIMULATION_CACHE = None


def get_latest_simulation() -> dict:
    global LATEST_SIMULATION_CACHE
    if LATEST_SIMULATION_CACHE is None:
        LATEST_SIMULATION_CACHE = run_portfolio_simulation_api(initial_cash=100000.0, max_leverage=1.25)
    return LATEST_SIMULATION_CACHE


# =============================================================================
# 1. FLASK APPLICATION FACTORY (When Flask is Installed)
# =============================================================================
def create_flask_app():
    try:
        from flask import Flask, render_template, jsonify, request, send_from_directory
    except ImportError:
        return None

    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, "web", "templates"),
        static_folder=os.path.join(PROJECT_ROOT, "web", "static")
    )

    @app.route("/")
    def index():
        return render_template("index.html", active_page="dashboard", title="QuantumAlpha | Executive Dashboard")

    @app.route("/portfolio")
    def portfolio():
        return render_template("portfolio.html", active_page="portfolio", title="QuantumAlpha | Portfolio Allocation")

    @app.route("/markets")
    def markets():
        return render_template("markets.html", active_page="markets", title="QuantumAlpha | Markets Explorer")

    @app.route("/simulator")
    def simulator():
        return render_template("simulator.html", active_page="simulator", title="QuantumAlpha | Simulation Lab")

    @app.route("/trades")
    def trades():
        return render_template("trades.html", active_page="trades", title="QuantumAlpha | Trade Ledger")

    # --- REST API Endpoints ---
    @app.route("/api/simulation/latest")
    def api_latest_simulation():
        return jsonify(get_latest_simulation())

    @app.route("/api/markets")
    def api_markets():
        return jsonify(get_market_universe_data())

    @app.route("/api/market/<symbol>/bars")
    def api_market_bars(symbol):
        return jsonify(get_asset_bars_data(symbol))

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        data = request.get_json() or {}
        initial_cash = float(data.get("initial_cash", 100000.0))
        max_leverage = float(data.get("max_leverage", 1.25))
        symbols = data.get("symbols", None)

        res = run_portfolio_simulation_api(initial_cash=initial_cash, max_leverage=max_leverage, selected_symbols=symbols)
        global LATEST_SIMULATION_CACHE
        LATEST_SIMULATION_CACHE = res
        return jsonify(res)

    return app


# =============================================================================
# 2. STANDALONE ZERO-DEPENDENCY WSGI / HTTP SERVER (Pure Python Standard Library)
# =============================================================================
def render_simple_template(template_name: str, active_page: str, title: str) -> str:
    """Zero-dependency template renderer for standalone standard library server."""
    base_path = os.path.join(PROJECT_ROOT, "web", "templates", "base.html")
    child_path = os.path.join(PROJECT_ROOT, "web", "templates", template_name)

    with open(base_path, "r", encoding="utf-8") as f:
        base_html = f.read()
    with open(child_path, "r", encoding="utf-8") as f:
        child_html = f.read()

    # Extract block content
    content_start = child_html.find("{% block content %}")
    content_end = child_html.find("{% endblock %}")
    block_content = child_html[content_start + 19:content_end] if content_start != -1 else child_html

    # Extract block scripts
    scripts_start = child_html.find("{% block scripts %}")
    scripts_end = child_html.rfind("{% endblock %}")
    block_scripts = child_html[scripts_start + 19:scripts_end] if scripts_start != -1 and scripts_start != content_start else ""

    # Replace in base
    rendered = base_html.replace('{{ title if title else "QuantumAlpha | Probabilistic Trading Platform" }}', title)
    for p in ["dashboard", "portfolio", "markets", "simulator", "trades"]:
        rendered = rendered.replace(f"{{{{ 'active' if active_page == '{p}' else '' }}}}", "active" if p == active_page else "")

    rendered = rendered.replace("{% block content %}{% endblock %}", block_content)
    rendered = rendered.replace("{% block scripts %}{% endblock %}", block_scripts)
    return rendered


class StandaloneAppServer:
    """Pure Python WSGI & HTTP application for serving QuantumAlpha web UI without third-party packages."""

    @staticmethod
    def handle_request(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET")

        # Static assets
        if path.startswith("/static/"):
            rel_path = path[8:] # Strip /static/
            file_path = os.path.join(PROJECT_ROOT, "web", "static", rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                content_type = "text/css" if file_path.endswith(".css") else ("application/javascript" if file_path.endswith(".js") else "text/plain")
                start_response("200 OK", [("Content-Type", content_type), ("Cache-Control", "max-age=3600")])
                with open(file_path, "rb") as f:
                    return [f.read()]
            else:
                start_response("404 Not Found", [("Content-Type", "text/plain")])
                return [b"Static file not found"]

        # HTML Pages
        page_map = {
            "/": ("index.html", "dashboard", "QuantumAlpha | Executive Dashboard"),
            "/portfolio": ("portfolio.html", "portfolio", "QuantumAlpha | Portfolio Allocation"),
            "/markets": ("markets.html", "markets", "QuantumAlpha | Markets Explorer"),
            "/simulator": ("simulator.html", "simulator", "QuantumAlpha | Simulation Lab"),
            "/trades": ("trades.html", "trades", "QuantumAlpha | Trade Ledger"),
        }

        if path in page_map:
            tpl, active, title = page_map[path]
            html = render_simple_template(tpl, active, title)
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
            return [html.encode("utf-8")]

        # REST API Endpoints
        if path == "/api/simulation/latest":
            data = get_latest_simulation()
            start_response("200 OK", [("Content-Type", "application/json")])
            return [json.dumps(data).encode("utf-8")]

        if path == "/api/markets":
            data = get_market_universe_data()
            start_response("200 OK", [("Content-Type", "application/json")])
            return [json.dumps(data).encode("utf-8")]

        if path.startswith("/api/market/") and path.endswith("/bars"):
            parts = path.split("/")
            symbol = parts[3]
            data = get_asset_bars_data(symbol)
            start_response("200 OK", [("Content-Type", "application/json")])
            return [json.dumps(data).encode("utf-8")]

        if path == "/api/simulate" and method == "POST":
            try:
                content_length = int(environ.get("CONTENT_LENGTH", 0))
                body = environ["wsgi.input"].read(content_length)
                post_data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                post_data = {}

            initial_cash = float(post_data.get("initial_cash", 100000.0))
            max_leverage = float(post_data.get("max_leverage", 1.25))
            symbols = post_data.get("symbols", None)

            res = run_portfolio_simulation_api(initial_cash=initial_cash, max_leverage=max_leverage, selected_symbols=symbols)
            global LATEST_SIMULATION_CACHE
            LATEST_SIMULATION_CACHE = res

            start_response("200 OK", [("Content-Type", "application/json")])
            return [json.dumps(res).encode("utf-8")]

        # 404
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"404 Not Found"]


# =============================================================================
# 3. ENTRYPOINT & LAUNCHER
# =============================================================================
def main():
    port = int(os.environ.get("PORT", 8088))
    host = "0.0.0.0"

    print("=" * 70)
    print("  🌌 QuantumAlpha Web Application Server")
    print("=" * 70)
    print(f"[*] Starting Server on http://{host}:{port}...")

    flask_app = create_flask_app()
    if flask_app is not None:
        print("[+] Flask framework detected. Launching via Flask WSGI engine.")
        flask_app.run(host=host, port=port, debug=False)
    else:
        print("[+] Pure-Python environment detected. Launching via built-in WSGI server.")
        from wsgiref.simple_server import make_server
        httpd = make_server(host, port, StandaloneAppServer.handle_request)
        print(f"[+] Server live and listening at http://localhost:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutting down cleanly.")


if __name__ == "__main__":
    main()

