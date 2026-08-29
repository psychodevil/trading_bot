/**
 * TradingView Lightweight-Charts Integration Helper
 */

function createLightweightChart(containerId, height = 400) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    container.innerHTML = ""; // Clear existing

    const chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#161b24' },
            textColor: '#9aa5b5',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        },
        grid: {
            vertLines: { color: '#262c38' },
            horzLines: { color: '#262c38' }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        timeScale: {
            borderColor: '#262c38',
            timeVisible: true,
            secondsVisible: false
        },
        rightPriceScale: {
            borderColor: '#262c38',
            scaleMargins: {
                top: 0.1,
                bottom: 0.1
            }
        }
    });

    // Auto-resize on window resize
    window.addEventListener('resize', () => {
        if (container.clientWidth > 0) {
            chart.applyOptions({ width: container.clientWidth });
        }
    });

    return chart;
}

function renderEquityComparisonChart(containerId, equitySeries, benchmarkSeries, cashSeries) {
    const chart = createLightweightChart(containerId, 440);
    if (!chart) return;

    const eqSeries = chart.addAreaSeries({
        topColor: 'rgba(46, 160, 67, 0.35)',
        bottomColor: 'rgba(46, 160, 67, 0.0)',
        lineColor: '#2ea043',
        lineWidth: 2,
        title: 'Strategy Equity'
    });
    eqSeries.setData(equitySeries);

    const bnhSeries = chart.addLineSeries({
        color: '#8b949e',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        title: 'Buy & Hold'
    });
    bnhSeries.setData(benchmarkSeries);

    if (cashSeries && cashSeries.length > 0) {
        const cSeries = chart.addLineSeries({
            color: '#388bfd',
            lineWidth: 1,
            title: 'Cash Reserve'
        });
        cSeries.setData(cashSeries);
    }

    chart.timeScale().fitContent();
    return chart;
}

function renderCandlestickChart(containerId, ohlcvData) {
    const chart = createLightweightChart(containerId, 420);
    if (!chart) return;

    const candleSeries = chart.addCandlestickSeries({
        upColor: '#2ea043',
        downColor: '#f85149',
        borderVisible: false,
        wickUpColor: '#2ea043',
        wickDownColor: '#f85149'
    });
    candleSeries.setData(ohlcvData);

    chart.timeScale().fitContent();
    return chart;
}

