"""
Interactive Lightweight-Charts Dashboard and Whole Market HTML Report Generator.
Leverages TradingView's lightweight-charts for interactive candlestick charts,
volume histograms, execution markers, probability forecast cones, inaction bands,
and synchronized equity/drawdown sub-panels across the entire market.
"""

from __future__ import annotations
import json
import math
from typing import List, Optional, Dict, Any
from pathlib import Path

from trading_bot.backtest.engine import BacktestResult


class ReportGenerator:
    """
    Generates rich, standalone interactive dashboards powered by TradingView Lightweight-Charts.
    """

    @classmethod
    def generate_html_report(
        cls,
        results: List[BacktestResult],
        title: str = "Probabilistic Trading Bot: Whole Market Analysis",
        output_path: Optional[str] = None
    ) -> str:
        """
        Builds a multi-dataset interactive TradingView lightweight-charts dashboard in pure HTML/CSS/JS.
        """
        if not results:
            raise ValueError("No results provided to generate report")

        # Prepare JSON datasets for all results to allow instant client-side switching
        datasets: Dict[str, Any] = {}
        table_rows_html = ""

        for idx, res in enumerate(results):
            key = f"{res.instrument.symbol}_{res.timeframe_description.replace(' ', '_')}"
            
            # Format candles for lightweight-charts: { time: unix_sec, open, high, low, close }
            candles = []
            volume_data = []
            if getattr(res, "bars", None):
                for b in res.bars:
                    t = int(b.timestamp)
                    candles.append({
                        "time": t,
                        "open": round(b.open, 4),
                        "high": round(b.high, 4),
                        "low": round(b.low, 4),
                        "close": round(b.close, 4)
                    })
                    vol_color = "rgba(63, 185, 80, 0.4)" if b.close >= b.open else "rgba(248, 81, 73, 0.4)"
                    volume_data.append({
                        "time": t,
                        "value": round(b.volume, 2),
                        "color": vol_color
                    })
            else:
                for ts, p in res.price_history:
                    t = int(ts)
                    candles.append({
                        "time": t,
                        "open": round(p, 4),
                        "high": round(p, 4),
                        "low": round(p, 4),
                        "close": round(p, 4)
                    })

            # Format fills as markers
            markers = []
            for fill in res.fills:
                t = int(fill.timestamp)
                is_buy = (fill.side.value == "buy")
                markers.append({
                    "time": t,
                    "position": "belowBar" if is_buy else "aboveBar",
                    "color": "#3fb950" if is_buy else "#f85149",
                    "shape": "arrowUp" if is_buy else "arrowDown",
                    "text": f"{'BUY' if is_buy else 'SELL'} {fill.quantity:.3f} @ ${fill.price:.2f}"
                })

            # Format weights and inaction bands
            weights_data = [{"time": int(ts), "value": round(w, 4)} for ts, w in res.weights_history]
            inaction_upper = [{"time": int(ts), "value": round(high, 4)} for ts, low, high in res.inaction_bands_history]
            inaction_lower = [{"time": int(ts), "value": round(low, 4)} for ts, low, high in res.inaction_bands_history]
            frictionless_targets = [{"time": int(ts), "value": round(w, 4)} for ts, w in res.frictionless_targets_history]

            # Format equity curve
            equity_data = [{"time": int(ts), "value": round(eq, 2)} for ts, eq in res.equity_curve]

            # Format probability forecast cones
            forecast_mean = []
            forecast_upper = []
            forecast_lower = []
            for i, (ts, mu, sig) in enumerate(res.forecasts_history):
                t = int(ts)
                ref_price = res.price_history[i][1] if i < len(res.price_history) else 100.0
                expected_p = ref_price * math.exp(mu)
                upper_p = ref_price * math.exp(mu + 1.96 * sig)
                lower_p = ref_price * math.exp(mu - 1.96 * sig)
                forecast_mean.append({"time": t, "value": round(expected_p, 4)})
                forecast_upper.append({"time": t, "value": round(upper_p, 4)})
                forecast_lower.append({"time": t, "value": round(lower_p, 4)})

            # Metrics
            m = res.metrics
            datasets[key] = {
                "key": key,
                "strategy_name": res.strategy_name,
                "symbol": res.instrument.symbol,
                "asset_class": res.instrument.asset_class.value,
                "timeframe": res.timeframe_description,
                "candles": candles,
                "volume": volume_data,
                "markers": markers,
                "weights": weights_data,
                "inaction_upper": inaction_upper,
                "inaction_lower": inaction_lower,
                "frictionless_targets": frictionless_targets,
                "equity": equity_data,
                "forecast_mean": forecast_mean,
                "forecast_upper": forecast_upper,
                "forecast_lower": forecast_lower,
                "metrics": {
                    "total_return_pct": m.total_return_pct,
                    "cagr_pct": m.cagr_pct,
                    "sharpe_ratio": m.sharpe_ratio,
                    "sortino_ratio": m.sortino_ratio,
                    "calmar_ratio": m.calmar_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "max_drawdown_days": m.max_drawdown_duration_days,
                    "win_rate_pct": m.win_rate_pct,
                    "profit_factor": m.profit_factor,
                    "total_trades": m.total_trades,
                    "total_turnover": m.total_turnover_dollars,
                    "total_fees": m.total_fees_paid_dollars,
                    "fee_drag_pct": m.fee_drag_pct,
                    "inaction_efficiency_pct": m.inaction_efficiency_pct,
                    "cvar_95_pct": m.cvar_95_daily_pct,
                    "initial_equity": m.initial_equity,
                    "final_equity": m.final_equity
                }
            }

            ret_cls = "positive" if m.total_return_pct >= 0 else "negative"
            table_rows_html += f"""
            <tr onclick="switchDataset('{key}')" style="cursor:pointer;">
                <td><strong>{res.instrument.symbol}</strong></td>
                <td><span class="badge">{res.instrument.asset_class.value}</span></td>
                <td>{res.timeframe_description}</td>
                <td class="{ret_cls}">{m.total_return_pct:+.2f}%</td>
                <td><strong>{m.sharpe_ratio:.2f}</strong></td>
                <td class="negative">{m.max_drawdown_pct:.2f}%</td>
                <td>{m.win_rate_pct:.1f}%</td>
                <td>${m.total_fees_paid_dollars:,.2f}</td>
                <td><span class="highlight">{m.inaction_efficiency_pct:.1f}%</span></td>
            </tr>
            """

        datasets_json = json.dumps(datasets)
        dataset_keys = list(datasets.keys())
        default_key = dataset_keys[0]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- TradingView Lightweight-Charts -->
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --border-color: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-purple: #bc8cff;
            --accent-orange: #f0883e;
            --accent-cyan: #39c5cf;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
        }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 12px;
        }}
        h1 {{ font-size: 22px; font-weight: 600; }}
        .header-controls {{ display: flex; gap: 12px; align-items: center; }}
        select.control-select {{
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            outline: none;
            cursor: pointer;
        }}
        select.control-select:focus {{ border-color: var(--accent-blue); }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
        }}
        .kpi-label {{ font-size: 11px; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.5px; margin-bottom: 4px; }}
        .kpi-val {{ font-size: 22px; font-weight: 700; color: var(--text-primary); }}
        .kpi-val.green {{ color: var(--accent-green); }}
        .kpi-val.red {{ color: var(--accent-red); }}
        .kpi-val.blue {{ color: var(--accent-blue); }}
        .kpi-val.orange {{ color: var(--accent-orange); }}
        .kpi-val.cyan {{ color: var(--accent-cyan); }}
        .kpi-sub {{ font-size: 11px; color: var(--text-secondary); margin-top: 4px; }}

        /* Chart Panels */
        .chart-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            position: relative;
        }}
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .panel-title {{ font-size: 14px; font-weight: 600; color: var(--text-primary); display: flex; align-items: center; gap: 8px; }}
        .badge {{
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent-blue);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }}
        .legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

        .chart-box {{ width: 100%; height: 420px; }}
        .chart-sub-box {{ width: 100%; height: 200px; }}

        /* Table */
        .table-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            max-height: 480px;
            overflow-y: auto;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: var(--bg-card); color: var(--text-secondary); position: sticky; top: 0; }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        .positive {{ color: var(--accent-green); font-weight: 600; }}
        .negative {{ color: var(--accent-red); font-weight: 600; }}
        .highlight {{ color: var(--accent-cyan); font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>{title}</h1>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                    TradingView Lightweight-Charts &bull; Probabilistic Distribution Cones &bull; Dynamic Inaction Bands
                </div>
            </div>
            <div class="header-controls">
                <label style="font-size: 12px; color: var(--text-secondary);">Select Asset ({len(results)} Total):</label>
                <select id="datasetSelector" class="control-select" onchange="switchDataset(this.value)">
                    <!-- Options populated via JS -->
                </select>
            </div>
        </header>

        <!-- KPI Summary Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Return</div>
                <div id="kpiReturn" class="kpi-val green">+0.00%</div>
                <div id="kpiCagr" class="kpi-sub">CAGR: 0.00%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Sharpe Ratio</div>
                <div id="kpiSharpe" class="kpi-val blue">0.00</div>
                <div id="kpiSortino" class="kpi-sub">Sortino: 0.00 | Calmar: 0.00</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Max Drawdown</div>
                <div id="kpiMaxDD" class="kpi-val red">0.00%</div>
                <div id="kpiRecovery" class="kpi-sub">Duration: 0.0 days</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Inaction Efficiency</div>
                <div id="kpiInactionEff" class="kpi-val orange">0.0%</div>
                <div id="kpiTradesAvoided" class="kpi-sub">Turnover Churn Suppressed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Fee & Slippage Drag</div>
                <div id="kpiFeeDrag" class="kpi-val red">$0.00</div>
                <div id="kpiFeePct" class="kpi-sub">0.00% of Portfolio</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tail Risk (CVaR 95%)</div>
                <div id="kpiCvar" class="kpi-val cyan">0.00%</div>
                <div id="kpiWinRate" class="kpi-sub">Win Rate: 0.0%</div>
            </div>
        </div>

        <!-- Main Candlestick Chart with Volume, Cones, & Order Fills -->
        <div class="chart-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <span id="chartSymbol">AAPL</span>
                    <span id="chartTimeframe" class="badge">1-Minute Bars</span>
                </div>
                <div class="legend">
                    <div class="legend-item"><span class="dot" style="background:#3fb950;"></span> Buy Marker</div>
                    <div class="legend-item"><span class="dot" style="background:#f85149;"></span> Sell Marker</div>
                    <div class="legend-item"><span class="dot" style="background:#39c5cf;"></span> Forecast Mean</div>
                    <div class="legend-item"><span class="dot" style="background:#bc8cff;"></span> &plusmn;2&sigma; Probability Cone</div>
                </div>
            </div>
            <div id="mainChartContainer" class="chart-box"></div>
        </div>

        <!-- Position Weight & Inaction Bands Sub-Chart -->
        <div class="chart-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <span>Position Weight Allocation & Dynamic No-Trade Inaction Region [w_lower, w_upper]</span>
                </div>
                <div class="legend">
                    <div class="legend-item"><span class="dot" style="background:#58a6ff;"></span> Actual Executed Weight</div>
                    <div class="legend-item"><span class="dot" style="background:#f0883e;"></span> Frictionless Target</div>
                    <div class="legend-item"><span class="dot" style="background:#bc8cff;"></span> Inaction Upper / Lower Bounds</div>
                </div>
            </div>
            <div id="weightChartContainer" class="chart-sub-box"></div>
        </div>

        <!-- Portfolio Equity Growth Sub-Chart -->
        <div class="chart-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <span>Portfolio Equity Growth ($)</span>
                    <span id="equityBadge" class="badge">$100,000 &rarr; $100,000</span>
                </div>
            </div>
            <div id="equityChartContainer" class="chart-sub-box"></div>
        </div>

        <!-- Market Universe Overview Table -->
        <div class="table-panel">
            <div style="font-size: 14px; font-weight: 600; margin-bottom: 10px;">Market Universe Cross-Section (Click any row to load chart)</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Class</th>
                        <th>Timeframe</th>
                        <th>Return</th>
                        <th>Sharpe</th>
                        <th>Max DD</th>
                        <th>Win Rate</th>
                        <th>Fees Paid</th>
                        <th>Inaction Efficiency</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const allDatasets = {datasets_json};
        const selector = document.getElementById("datasetSelector");

        // Populate dropdown
        for (const [k, d] of Object.entries(allDatasets)) {{
            const opt = document.createElement("option");
            opt.value = k;
            opt.textContent = `${{d.symbol}} (${{d.asset_class}}) - ${{d.strategy_name}}`;
            selector.appendChild(opt);
        }}

        // Initialize Lightweight Charts
        const chartTheme = {{
            layout: {{
                background: {{ color: '#161b22' }},
                textColor: '#8b949e',
            }},
            grid: {{
                vertLines: {{ color: '#21262d' }},
                horzLines: {{ color: '#21262d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            timeScale: {{
                borderColor: '#30363d',
                timeVisible: true,
                secondsVisible: false,
            }},
        }};

        // 1. Main Candlestick Chart
        const mainChart = LightweightCharts.createChart(document.getElementById("mainChartContainer"), {{
            ...chartTheme,
            rightPriceScale: {{ borderColor: '#30363d' }},
        }});
        const candleSeries = mainChart.addCandlestickSeries({{
            upColor: '#3fb950',
            downColor: '#f85149',
            borderVisible: false,
            wickUpColor: '#3fb950',
            wickDownColor: '#f85149',
        }});
        const volumeSeries = mainChart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: '',
            scaleMargins: {{ top: 0.8, bottom: 0 }},
        }});
        const forecastMeanSeries = mainChart.addLineSeries({{
            color: '#39c5cf',
            lineWidth: 1.5,
            lineStyle: LightweightCharts.LineStyle.Solid,
            title: 'Forecast Mean',
        }});
        const forecastUpperSeries = mainChart.addLineSeries({{
            color: 'rgba(188, 140, 255, 0.6)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: '+2σ Cone',
        }});
        const forecastLowerSeries = mainChart.addLineSeries({{
            color: 'rgba(188, 140, 255, 0.6)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: '-2σ Cone',
        }});

        // 2. Weight & Inaction Bands Sub-Chart
        const weightChart = LightweightCharts.createChart(document.getElementById("weightChartContainer"), {{
            ...chartTheme,
            rightPriceScale: {{ borderColor: '#30363d' }},
        }});
        const weightSeries = weightChart.addLineSeries({{
            color: '#58a6ff',
            lineWidth: 2.5,
            title: 'Actual Weight',
        }});
        const frictionlessSeries = weightChart.addLineSeries({{
            color: '#f0883e',
            lineWidth: 1.5,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: 'Frictionless Target',
        }});
        const inactionUpperSeries = weightChart.addLineSeries({{
            color: 'rgba(188, 140, 255, 0.8)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            title: 'Inaction Upper',
        }});
        const inactionLowerSeries = weightChart.addLineSeries({{
            color: 'rgba(188, 140, 255, 0.8)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            title: 'Inaction Lower',
        }});

        // 3. Equity Sub-Chart
        const equityChart = LightweightCharts.createChart(document.getElementById("equityChartContainer"), {{
            ...chartTheme,
            rightPriceScale: {{ borderColor: '#30363d' }},
        }});
        const equitySeries = equityChart.addAreaSeries({{
            topColor: 'rgba(63, 185, 80, 0.4)',
            bottomColor: 'rgba(63, 185, 80, 0.02)',
            lineColor: '#3fb950',
            lineWidth: 2,
            title: 'Equity',
        }});

        // Synchronize TimeScales across all 3 charts
        mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
            if (range) {{
                weightChart.timeScale().setVisibleLogicalRange(range);
                equityChart.timeScale().setVisibleLogicalRange(range);
            }}
        }});
        weightChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
            if (range) {{
                mainChart.timeScale().setVisibleLogicalRange(range);
                equityChart.timeScale().setVisibleLogicalRange(range);
            }}
        }});
        equityChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
            if (range) {{
                mainChart.timeScale().setVisibleLogicalRange(range);
                weightChart.timeScale().setVisibleLogicalRange(range);
            }}
        }});

        // Responsive resize
        window.addEventListener('resize', () => {{
            const w = document.getElementById("mainChartContainer").clientWidth;
            mainChart.resize(w, 420);
            weightChart.resize(w, 200);
            equityChart.resize(w, 200);
        }});

        function switchDataset(key) {{
            const data = allDatasets[key];
            if (!data) return;

            // Sync dropdown value
            selector.value = key;

            // Update Titles & Badges
            document.getElementById("chartSymbol").textContent = `${{data.symbol}} (${{data.asset_class}}) - ${{data.strategy_name}}`;
            document.getElementById("chartTimeframe").textContent = data.timeframe;
            document.getElementById("equityBadge").textContent = `$${{data.metrics.initial_equity.toLocaleString()}} → $${{data.metrics.final_equity.toLocaleString(undefined, {{maximumFractionDigits: 2}})}}`;

            // Update KPI Cards
            const retEl = document.getElementById("kpiReturn");
            retEl.textContent = (data.metrics.total_return_pct >= 0 ? "+" : "") + data.metrics.total_return_pct.toFixed(2) + "%";
            retEl.className = "kpi-val " + (data.metrics.total_return_pct >= 0 ? "green" : "red");
            document.getElementById("kpiCagr").textContent = `CAGR: ${{data.metrics.cagr_pct.toFixed(2)}}%`;
            document.getElementById("kpiSharpe").textContent = data.metrics.sharpe_ratio.toFixed(2);
            document.getElementById("kpiSortino").textContent = `Sortino: ${{data.metrics.sortino_ratio.toFixed(2)}} | Calmar: ${{data.metrics.calmar_ratio.toFixed(2)}}`;
            document.getElementById("kpiMaxDD").textContent = data.metrics.max_drawdown_pct.toFixed(2) + "%";
            document.getElementById("kpiRecovery").textContent = `Duration: ${{data.metrics.max_drawdown_days.toFixed(1)}} days`;
            document.getElementById("kpiInactionEff").textContent = data.metrics.inaction_efficiency_pct.toFixed(1) + "%";
            document.getElementById("kpiTradesAvoided").textContent = `${{data.metrics.total_trades}} Total Executed Fills`;
            document.getElementById("kpiFeeDrag").textContent = `$${{data.metrics.total_fees.toFixed(2)}}`;
            document.getElementById("kpiFeePct").textContent = `${{data.metrics.fee_drag_pct.toFixed(2)}}% Equity Drag`;
            document.getElementById("kpiCvar").textContent = `${{data.metrics.cvar_95_pct.toFixed(2)}}%`;
            document.getElementById("kpiWinRate").textContent = `Win Rate: ${{data.metrics.win_rate_pct.toFixed(1)}}%`;

            // Feed data to lightweight-charts
            candleSeries.setData(data.candles);
            volumeSeries.setData(data.volume);
            candleSeries.setMarkers(data.markers);

            forecastMeanSeries.setData(data.forecast_mean);
            forecastUpperSeries.setData(data.forecast_upper);
            forecastLowerSeries.setData(data.forecast_lower);

            weightSeries.setData(data.weights);
            frictionlessSeries.setData(data.frictionless_targets);
            inactionUpperSeries.setData(data.inaction_upper);
            inactionLowerSeries.setData(data.inaction_lower);

            equitySeries.setData(data.equity);

            mainChart.timeScale().fitContent();
            weightChart.timeScale().fitContent();
            equityChart.timeScale().fitContent();
        }}

        // Initialize with default dataset
        switchDataset("{default_key}");
    </script>
</body>
</html>
        """

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html_content, encoding="utf-8")

        return html_content
