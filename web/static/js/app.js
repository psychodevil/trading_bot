/**
 * QuantumAlpha Web Application Dynamic UI & API Integrator
 */

// Color Palette for Multi-Asset Timeline
const ASSET_PALETTE = [
    "#388bfd", "#2ea043", "#f85149", "#d29922", "#a371f7",
    "#00d2ff", "#ff7a00", "#76ff03", "#f50057", "#2979ff",
    "#00b0ff", "#651fff", "#1de9b6", "#ff9100", "#e040fb"
];

// =============================================================================
// 1. Dashboard View
// =============================================================================
async function initDashboardCharts() {
    try {
        const resp = await fetch('/api/simulation/latest');
        const data = await resp.json();

        // Render KPIs
        if (data.kpis) {
            document.getElementById('kpi-equity').innerText = `$${data.kpis.final_equity.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            document.getElementById('kpi-return').innerText = `${data.kpis.total_return_pct >= 0 ? '+' : ''}${data.kpis.total_return_pct.toFixed(2)}%`;
            document.getElementById('kpi-benchmark').innerText = `${data.kpis.benchmark_return_pct >= 0 ? '+' : ''}${data.kpis.benchmark_return_pct.toFixed(2)}%`;
            document.getElementById('kpi-alpha').innerText = `${data.kpis.alpha_pct >= 0 ? '+' : ''}${data.kpis.alpha_pct.toFixed(2)}%`;
            document.getElementById('kpi-trades').innerText = data.kpis.total_trades.toLocaleString();
        }

        // Render Chart
        if (data.equity_curve && data.equity_curve.length > 0) {
            const eqData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.equity }));
            const bnhData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.benchmark }));
            const cashData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.cash }));
            renderEquityComparisonChart('mainEquityChart', eqData, bnhData, cashData);
        }

        // Render Active Positions Table
        if (data.positions && data.positions.length > 0) {
            const tbody = document.getElementById('positionsTableBody');
            tbody.innerHTML = '';
            data.positions.forEach(pos => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${pos.symbol}</strong></td>
                    <td>${pos.quantity.toFixed(3)}</td>
                    <td>$${pos.current_price.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
                    <td>$${pos.position_value.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
                    <td><span class="badge blue">${pos.portfolio_weight.toFixed(1)}%</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Failed to load dashboard data:", e);
    }
}

// =============================================================================
// 2. Portfolio Allocation View
// =============================================================================
async function initPortfolioView() {
    try {
        const resp = await fetch('/api/simulation/latest');
        const data = await resp.json();

        if (data.equity_curve && data.equity_curve.length > 0) {
            const chart = createLightweightChart('portfolioAllocationChart', 380);
            const legendContainer = document.getElementById('allocationLegend');
            legendContainer.innerHTML = '';

            const sampleWeights = data.equity_curve[data.equity_curve.length - 1].weights || {};
            const symbols = Object.keys(sampleWeights);

            symbols.forEach((sym, idx) => {
                const col = ASSET_PALETTE[idx % ASSET_PALETTE.length];
                const s = chart.addLineSeries({
                    color: col,
                    lineWidth: 2,
                    title: sym
                });

                const sData = data.equity_curve.map(pt => ({
                    time: pt.time,
                    value: (pt.weights && pt.weights[sym]) ? pt.weights[sym] : 0
                }));
                s.setData(sData);

                // Add to legend
                const item = document.createElement('div');
                item.className = 'legend-badge';
                item.innerHTML = `<span class="dot" style="background: ${col};"></span> ${sym}`;
                legendContainer.appendChild(item);
            });

            chart.timeScale().fitContent();
        }

        // Render Positions
        if (data.positions) {
            const tbody = document.getElementById('fullPositionsTableBody');
            tbody.innerHTML = '';
            data.positions.forEach(pos => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${pos.symbol}</strong></td>
                    <td>${pos.symbol.includes('USD') ? 'Crypto' : (pos.symbol.includes('GLD') || pos.symbol.includes('SLV') ? 'Commodity' : 'Stock')}</td>
                    <td>${pos.quantity.toFixed(4)}</td>
                    <td>$${pos.current_price.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
                    <td>$${pos.position_value.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
                    <td><strong>${pos.portfolio_weight.toFixed(2)}%</strong></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Failed to load portfolio view:", e);
    }
}

// =============================================================================
// 3. Markets Explorer View
// =============================================================================
let globalMarketAssets = [];

async function initMarketsView() {
    try {
        const resp = await fetch('/api/markets');
        globalMarketAssets = await resp.json();
        renderMarketAssetsTable(globalMarketAssets);

        // Load default candlestick chart for SPY
        loadAssetCandlesticks('SPY', 'S&P 500 ETF', 'Broad Market', 'STOCK', 18.64);
    } catch (e) {
        console.error("Failed to load market universe:", e);
    }
}

function renderMarketAssetsTable(assets) {
    const tbody = document.getElementById('marketAssetsTableBody');
    tbody.innerHTML = '';

    assets.forEach(a => {
        const tr = document.createElement('tr');
        const retClass = a.return_pct >= 0 ? 'positive' : 'negative';
        tr.innerHTML = `
            <td><strong>${a.symbol}</strong></td>
            <td>${a.name}</td>
            <td>${a.sector}</td>
            <td><span class="badge blue">${a.asset_class}</span></td>
            <td>${a.bars_count.toLocaleString()}</td>
            <td>$${a.latest_price.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
            <td><span class="${retClass}">${a.return_pct >= 0 ? '+' : ''}${a.return_pct.toFixed(2)}%</span></td>
            <td>
                <button class="btn btn-outline" style="padding: 3px 8px; font-size: 11px;" 
                    onclick="loadAssetCandlesticks('${a.symbol}', '${a.name.replace(/'/g, "\\'")}', '${a.sector}', '${a.asset_class}', ${a.return_pct})">
                    📈 View Chart
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function filterMarketTable() {
    const sector = document.getElementById('marketSectorFilter').value;
    const search = document.getElementById('marketSearchInput').value.toLowerCase();

    const filtered = globalMarketAssets.filter(a => {
        const matchSector = (sector === 'ALL' || a.sector === sector);
        const matchSearch = (a.symbol.toLowerCase().includes(search) || a.name.toLowerCase().includes(search));
        return matchSector && matchSearch;
    });
    renderMarketAssetsTable(filtered);
}

async function loadAssetCandlesticks(symbol, name, sector, assetClass, returnPct) {
    document.getElementById('selectedAssetTitle').innerText = `${symbol} - ${name} (1-Hour Candlesticks)`;
    document.getElementById('selectedAssetSubtitle').innerText = `${sector} • Live historical OHLCV data`;
    document.getElementById('selectedAssetClass').innerText = assetClass;
    
    const retBadge = document.getElementById('selectedAssetReturn');
    retBadge.innerText = `${returnPct >= 0 ? '+' : ''}${returnPct.toFixed(2)}%`;
    retBadge.className = `badge ${returnPct >= 0 ? 'green' : 'red'}`;

    try {
        const resp = await fetch(`/api/market/${symbol}/bars`);
        const bars = await resp.json();
        if (bars.length > 0) {
            renderCandlestickChart('marketCandlestickChart', bars);
        }
    } catch (e) {
        console.error("Failed to load candlestick bars:", e);
    }
}

// =============================================================================
// 4. Simulator Lab View
// =============================================================================
async function initSimulatorView() {
    try {
        const resp = await fetch('/api/simulation/latest');
        const data = await resp.json();
        if (data.equity_curve) {
            const eqData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.equity }));
            const bnhData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.benchmark }));
            renderEquityComparisonChart('simEquityChart', eqData, bnhData, null);
        }
    } catch (e) {
        console.error("Failed to load simulator default data:", e);
    }
}

async function runCustomSimulation() {
    const initialCash = parseFloat(document.getElementById('simInitialCash').value) || 100000;
    const maxLeverage = parseFloat(document.getElementById('simMaxLeverage').value) || 1.25;
    
    const checkboxes = document.querySelectorAll('input[name="simAsset"]:checked');
    const symbols = Array.from(checkboxes).map(cb => cb.value);

    const btn = document.getElementById('runSimBtn');
    const badge = document.getElementById('simStatusBadge');
    btn.disabled = true;
    btn.innerText = "⏳ Running Quantitative Simulation...";
    badge.innerText = "Simulating...";
    badge.className = "badge blue";

    try {
        const resp = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                initial_cash: initialCash,
                max_leverage: maxLeverage,
                symbols: symbols
            })
        });
        const data = await resp.json();

        // Update KPIs
        if (data.kpis) {
            document.getElementById('simResEquity').innerText = `$${data.kpis.final_equity.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            document.getElementById('simResReturn').innerText = `${data.kpis.total_return_pct >= 0 ? '+' : ''}${data.kpis.total_return_pct.toFixed(2)}%`;
            document.getElementById('simResBench').innerText = `${data.kpis.benchmark_return_pct >= 0 ? '+' : ''}${data.kpis.benchmark_return_pct.toFixed(2)}%`;
            document.getElementById('simResAlpha').innerText = `${data.kpis.alpha_pct >= 0 ? '+' : ''}${data.kpis.alpha_pct.toFixed(2)}%`;
            document.getElementById('simResTrades').innerText = data.kpis.total_trades.toLocaleString();
            document.getElementById('simResFees').innerText = `$${data.kpis.total_fees.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            document.getElementById('simResCash').innerText = `$${data.kpis.cash_balance.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
        }

        // Render Chart
        if (data.equity_curve) {
            const eqData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.equity }));
            const bnhData = data.equity_curve.map(pt => ({ time: pt.time, value: pt.benchmark }));
            renderEquityComparisonChart('simEquityChart', eqData, bnhData, null);
        }

        badge.innerText = "Complete";
        badge.className = "badge green";
    } catch (e) {
        console.error("Simulation error:", e);
        badge.innerText = "Error";
        badge.className = "badge red";
    } finally {
        btn.disabled = false;
        btn.innerText = "🚀 Run Quantitative Simulation";
    }
}

// =============================================================================
// 5. Trades View
// =============================================================================
let globalTrades = [];

async function initTradesView() {
    try {
        const resp = await fetch('/api/simulation/latest');
        const data = await resp.json();
        if (data.trades) {
            globalTrades = data.trades;
            document.getElementById('tradeCountSpan').innerText = globalTrades.length.toLocaleString();
            renderTradeTable(globalTrades.slice(-200).reverse());
        }
    } catch (e) {
        console.error("Failed to load trade ledger:", e);
    }
}

function renderTradeTable(trades) {
    const tbody = document.getElementById('tradeLedgerTableBody');
    tbody.innerHTML = '';

    trades.forEach(t => {
        const tr = document.createElement('tr');
        const sideClass = t.side === 'BUY' ? 'positive' : 'negative';
        const dateStr = new Date(t.timestamp * 1000).toISOString().replace('T', ' ').substring(0, 16);
        tr.innerHTML = `
            <td>${dateStr}</td>
            <td><strong>${t.symbol}</strong></td>
            <td><span class="${sideClass}"><strong>${t.side}</strong></span></td>
            <td>${t.quantity.toFixed(4)}</td>
            <td>$${t.price.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
            <td>$${(t.quantity * t.price).toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
            <td>$${t.fee.toFixed(2)}</td>
            <td><strong>${t.target_weight.toFixed(1)}%</strong></td>
        `;
        tbody.appendChild(tr);
    });
}

function filterTradeTable() {
    const query = document.getElementById('tradeSearchInput').value.toLowerCase();
    const filtered = globalTrades.filter(t => {
        return t.symbol.toLowerCase().includes(query) || t.side.toLowerCase().includes(query);
    });
    renderTradeTable(filtered.slice(-200).reverse());
}

async function triggerFastBacktest() {
    alert("Executing fast walk-forward simulation across multi-asset universe...");
    await runCustomSimulation();
    window.location.href = '/';
}
