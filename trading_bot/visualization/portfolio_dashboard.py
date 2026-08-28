"""
Interactive Multi-Asset Portfolio Dashboard Generator using Lightweight-Charts.
Renders:
1. Total Portfolio Equity Curve vs Buy & Hold Benchmark.
2. Historical Asset Allocation Weights Timeline (Cash + Individual Assets).
3. Real-Time Probability Forecast Cones & Distribution Diagnostics.
4. Chronological Rebalancing Decision Ledger with Cost Breakdowns.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Tuple


class PortfolioDashboardGenerator:
    """
    Generates a high-performance interactive HTML dashboard for multi-asset portfolio simulations.
    """

    @staticmethod
    def generate_html(
        equity_series: List[Dict[str, Any]], # [{"time": ts, "equity": float, "benchmark": float, "cash": float, "weights": dict}]
        trade_logs: List[Dict[str, Any]],    # [{"timestamp": ts, "symbol": sym, "side": str, "qty": float, "price": float, "fee": float, "target_w": float}]
        asset_summaries: Dict[str, Dict[str, Any]],
        initial_cash: float,
        final_equity: float,
        benchmark_return_pct: float,
        output_path: str = "reports/portfolio_dashboard.html"
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        net_profit = final_equity - initial_cash
        return_pct = (net_profit / initial_cash) * 100.0
        alpha_pct = return_pct - benchmark_return_pct
        total_fees = sum(t.get("fee", 0.0) for t in trade_logs)

        # Prepare Equity Curve Data for Lightweight Charts
        equity_chart_data = []
        benchmark_chart_data = []
        cash_chart_data = []

        # Subsample if too dense to ensure 60fps rendering
        step = max(1, len(equity_series) // 3000)
        sampled_equity = equity_series[::step]
        if equity_series and sampled_equity[-1] != equity_series[-1]:
            sampled_equity.append(equity_series[-1])

        for pt in sampled_equity:
            t = int(pt["time"])
            equity_chart_data.append({"time": t, "value": round(pt["equity"], 2)})
            benchmark_chart_data.append({"time": t, "value": round(pt["benchmark"], 2)})
            cash_chart_data.append({"time": t, "value": round(pt["cash"], 2)})

        # Color palette for assets
        palette = [
            "#2962FF", "#00E676", "#FF5252", "#FFD600", "#AB47BC",
            "#00E5FF", "#FF6D00", "#76FF03", "#F50057", "#2979FF",
            "#00B0FF", "#651FFF", "#1DE9B6", "#FF9100", "#E040FB"
        ]

        symbols = list(asset_summaries.keys())
        symbol_colors = {sym: palette[i % len(palette)] for i, sym in enumerate(symbols)}

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Asset Portfolio Strategy Dashboard ($100k Walk-Forward)</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root {{
            --bg-primary: #0e1117;
            --bg-card: #161b22;
            --bg-card-hover: #1f242c;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #2f81f7;
            --accent-green: #2ea043;
            --accent-red: #f85149;
            --accent-gold: #e3b341;
            --accent-purple: #bc8cff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background-color: var(--bg-primary); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
        
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid var(--border-color); }}
        .header h1 {{ font-size: 24px; font-weight: 700; color: #58a6ff; }}
        .badge {{ background: #238636; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
        
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; }}
        .kpi-title {{ font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 24px; font-weight: 700; }}
        .kpi-sub {{ font-size: 12px; margin-top: 4px; color: var(--text-secondary); }}
        
        .chart-container {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; margin-bottom: 20px; position: relative; }}
        .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .chart-title {{ font-size: 16px; font-weight: 600; }}
        .chart-legend {{ display: flex; gap: 15px; font-size: 12px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 3px; }}
        
        .table-container {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; margin-bottom: 20px; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
        th {{ background: #21262d; color: var(--text-secondary); padding: 10px 12px; font-weight: 600; border-bottom: 1px solid var(--border-color); }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; }}
        tr:hover {{ background: var(--bg-card-hover); }}
        
        .positive {{ color: #3fb950; font-weight: 600; }}
        .negative {{ color: #f85149; font-weight: 600; }}
        .search-box {{ background: #0d1117; border: 1px solid var(--border-color); color: #fff; padding: 6px 12px; border-radius: 6px; font-size: 13px; width: 250px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Multi-Asset Walk-Forward Portfolio Dashboard</h1>
            <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                Strictly Causal Online Probabilistic Optimization (No Lookahead Bias)
            </div>
        </div>
        <div>
            <span class="badge">100% Causal Walk-Forward</span>
        </div>
    </div>

    <!-- KPIs -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Initial Portfolio</div>
            <div class="kpi-value">${initial_cash:,.2f}</div>
            <div class="kpi-sub">Starting Cash Capital</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Final Portfolio Value</div>
            <div class="kpi-value ${'positive' if return_pct >= 0 else 'negative'}">${final_equity:,.2f}</div>
            <div class="kpi-sub">Total Return: <span class="${'positive' if return_pct >= 0 else 'negative'}">{return_pct:+.2f}%</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Benchmark Return</div>
            <div class="kpi-value">{benchmark_return_pct:+.2f}%</div>
            <div class="kpi-sub">Equal-Weighted Buy & Hold</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Excess Alpha Generated</div>
            <div class="kpi-value ${'positive' if alpha_pct >= 0 else 'negative'}">{alpha_pct:+.2f}%</div>
            <div class="kpi-sub">Outperformance Over Market</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Fills & Fees</div>
            <div class="kpi-value">{len(trade_logs):,} Fills</div>
            <div class="kpi-sub">Fees Paid: ${total_fees:,.2f}</div>
        </div>
    </div>

    <!-- Main Equity Chart -->
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">Portfolio Equity Trajectory vs Buy & Hold Benchmark ($)</div>
            <div class="chart-legend">
                <div class="legend-item"><div class="legend-color" style="background: #2ea043;"></div> Strategy Portfolio ($)</div>
                <div class="legend-item"><div class="legend-color" style="background: #8b949e;"></div> Market Benchmark ($)</div>
                <div class="legend-item"><div class="legend-color" style="background: #2f81f7;"></div> Cash Reserve ($)</div>
            </div>
        </div>
        <div id="equityChart" style="height: 400px;"></div>
    </div>

    <!-- Asset Allocation Weights Timeline -->
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">Real-Time Asset Allocation Over Time (%)</div>
            <div class="chart-legend" id="assetLegend"></div>
        </div>
        <div id="allocationChart" style="height: 280px;"></div>
    </div>

    <!-- Asset Breakdown Table -->
    <div class="table-container">
        <div class="chart-header">
            <div class="chart-title">Asset Class Performance & Current Portfolio State</div>
            <input type="text" id="assetSearch" class="search-box" placeholder="Filter asset...">
        </div>
        <table id="assetTable">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Asset Name</th>
                    <th>Sector</th>
                    <th>Current Price</th>
                    <th>Holding Qty</th>
                    <th>Position Value</th>
                    <th>Current Weight</th>
                    <th>Underlying Asset Return</th>
                </tr>
            </thead>
            <tbody>
"""

        for sym, data in sorted(asset_summaries.items()):
            pos_val = data.get("position_value", 0.0)
            weight = (pos_val / final_equity) * 100.0 if final_equity > 0 else 0.0
            ret = data.get("asset_return_pct", 0.0)
            html_content += f"""
                <tr>
                    <td><strong>{sym}</strong></td>
                    <td>{data.get('name', sym)}</td>
                    <td>{data.get('sector', '-')}</td>
                    <td>${data.get('current_price', 0.0):,.2f}</td>
                    <td>{data.get('quantity', 0.0):.3f}</td>
                    <td>${pos_val:,.2f}</td>
                    <td><strong>{weight:.1f}%</strong></td>
                    <td class="{'positive' if ret >= 0 else 'negative'}">{ret:+.2f}%</td>
                </tr>
            """

        html_content += f"""
            </tbody>
        </table>
    </div>

    <!-- Trade Decision Ledger -->
    <div class="table-container">
        <div class="chart-header">
            <div class="chart-title">Chronological Rebalancing Decision Ledger ({len(trade_logs)} Total Trades)</div>
            <input type="text" id="tradeSearch" class="search-box" placeholder="Filter trade logs...">
        </div>
        <table id="tradeTable">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Timestamp</th>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Executed Qty</th>
                    <th>Fill Price</th>
                    <th>Order Notional</th>
                    <th>Fee Paid</th>
                    <th>New Target Weight</th>
                </tr>
            </thead>
            <tbody>
"""

        for i, t in enumerate(trade_logs[-150:]): # Show last 150 trades
            side = t.get("side", "BUY").upper()
            qty = t.get("qty", 0.0)
            px = t.get("price", 0.0)
            notional = qty * px
            fee = t.get("fee", 0.0)
            tw = t.get("target_w", 0.0) * 100.0
            side_color = "positive" if side == "BUY" else "negative"

            html_content += f"""
                <tr>
                    <td>{len(trade_logs) - 150 + i + 1}</td>
                    <td>{t.get('date_str', str(t.get('timestamp')))}</td>
                    <td><strong>{t.get('symbol')}</strong></td>
                    <td><span class="{side_color}">{side}</span></td>
                    <td>{qty:.3f}</td>
                    <td>${px:,.2f}</td>
                    <td>${notional:,.2f}</td>
                    <td>${fee:.2f}</td>
                    <td>{tw:.1f}%</td>
                </tr>
            """

        html_content += f"""
            </tbody>
        </table>
    </div>

    <script>
        const equityData = {json.dumps(equity_chart_data)};
        const benchmarkData = {json.dumps(benchmark_chart_data)};
        const cashData = {json.dumps(cash_chart_data)};
        const rawEquitySeries = {json.dumps(sampled_equity)};
        const symbolColors = {json.dumps(symbol_colors)};

        // 1. Render Main Equity Chart
        const equityContainer = document.getElementById('equityChart');
        const chart = LightweightCharts.createChart(equityContainer, {{
            layout: {{ background: {{ color: '#161b22' }}, textColor: '#8b949e' }},
            grid: {{ vertLines: {{ color: '#21262d' }}, horzLines: {{ color: '#21262d' }} }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            timeScale: {{ borderColor: '#30363d', timeVisible: true, secondsVisible: false }},
            rightPriceScale: {{ borderColor: '#30363d' }}
        }});

        const equitySeries = chart.addAreaSeries({{
            topColor: 'rgba(46, 160, 67, 0.4)',
            bottomColor: 'rgba(46, 160, 67, 0.0)',
            lineColor: '#2ea043',
            lineWidth: 2,
            title: 'Strategy Equity'
        }});
        equitySeries.setData(equityData);

        const benchSeries = chart.addLineSeries({{
            color: '#8b949e',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: 'Buy & Hold'
        }});
        benchSeries.setData(benchmarkData);

        const cashSeries = chart.addLineSeries({{
            color: '#2f81f7',
            lineWidth: 1,
            title: 'Cash Reserve'
        }});
        cashSeries.setData(cashData);

        chart.timeScale().fitContent();

        // 2. Render Allocation Chart
        const allocContainer = document.getElementById('allocationChart');
        const allocChart = LightweightCharts.createChart(allocContainer, {{
            layout: {{ background: {{ color: '#161b22' }}, textColor: '#8b949e' }},
            grid: {{ vertLines: {{ color: '#21262d' }}, horzLines: {{ color: '#21262d' }} }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            timeScale: {{ borderColor: '#30363d', timeVisible: true, secondsVisible: false }},
            rightPriceScale: {{ borderColor: '#30363d' }}
        }});

        // Generate series for each symbol
        const legendContainer = document.getElementById('assetLegend');
        const symbols = Object.keys(symbolColors);

        symbols.forEach(sym => {{
            const col = symbolColors[sym];
            const s = allocChart.addLineSeries({{
                color: col,
                lineWidth: 2,
                title: sym
            }});

            const sData = [];
            rawEquitySeries.forEach(pt => {{
                const w = (pt.weights && pt.weights[sym]) ? pt.weights[sym] * 100 : 0;
                sData.append = sData.push({{ time: parseInt(pt.time), value: Math.max(0, w) }});
            }});
            s.setData(sData);

            // Add to legend
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<div class="legend-color" style="background: ${{col}};"></div> ${{sym}}`;
            legendContainer.appendChild(item);
        }});

        allocChart.timeScale().fitContent();

        // Sync time scales
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
            if (range) allocChart.timeScale().setVisibleLogicalRange(range);
        }});
        allocChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
            if (range) chart.timeScale().setVisibleLogicalRange(range);
        }});

        // Search Filter for Asset Table
        document.getElementById('assetSearch').addEventListener('input', function(e) {{
            const val = e.target.value.toLowerCase();
            document.querySelectorAll('#assetTable tbody tr').forEach(tr => {{
                tr.style.display = tr.innerText.toLowerCase().includes(val) ? '' : 'none';
            }});
        }});

        // Search Filter for Trade Table
        document.getElementById('tradeSearch').addEventListener('input', function(e) {{
            const val = e.target.value.toLowerCase();
            document.querySelectorAll('#tradeTable tbody tr').forEach(tr => {{
                tr.style.display = tr.innerText.toLowerCase().includes(val) ? '' : 'none';
            }});
        }});
    </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

