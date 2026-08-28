# 📈 Quantitative Trading Strategies Guide

QuantumAlpha provides concrete, production-ready multi-asset strategies designed for distinct market regimes.

---

## 1. `AlphaPortfolioStrategy` (Master Strategy)
* **Target Instruments**: Equities, Tech Leaders, Commodities, Crypto Spot.
* **Core Logic**:
  - **Secular Bull Compounding ($1.35x - 1.50x$)**: Maintains compounding exposure when $\text{Price} > \text{EMA}_{100} - 3.5 \times \text{ATR}$ or $\text{EMA}_{20} > \text{EMA}_{50}$.
  - **High-Conviction Dip Accumulation**: Increases leverage when RSI dips between $35$ and $48$ during an established secular uptrend.
  - **100% Cash Defense in Bear Regimes**: Transitions entirely to Cash ($w = 0.0$) when prices break below trailing volatility stops, completely avoiding crypto/tech crashes.
  - **Calibrated Inaction Filter**: Freezes trades when current weights are within $4.5\%$ of target, avoiding fee drag.

---

## 2. `SecularTrendAlphaStrategy`
* **Target Instruments**: S&P 500 (`SPY`), Nasdaq 100 (`QQQ`), Mega-Cap Tech (`AAPL`, `MSFT`, `NVDA`, `AMD`).
* **Core Logic**:
  - Anchors macro regime on 150-bar and 35-bar moving averages.
  - Sits long during macro expansions and moves to cash during prolonged drawdowns.

---

## 3. `CryptoPerpFundingStrategy`
* **Target Instruments**: Perpetual Futures (`BTC-PERP`, `ETH-PERP`, `SOL-PERP`).
* **Core Logic**:
  - Harvests 8-hour funding rate cashflows when market skew is overextended.
  - Delta-hedges directional risk using spot or inverse futures.

---

## 4. `OptionsVolHarvestStrategy`
* **Target Instruments**: Equity / Index Options.
* **Core Logic**:
  - Models implied volatility surface vs historical realized volatility.
  - Sells rich variance via delta-neutral option straddles/strangles with strict stop losses.

---

## 5. `ForexMeanReversionStrategy`
* **Target Instruments**: Currency Pairs (`EUR/USD`, `GBP/USD`, `USD/JPY`).
* **Core Logic**:
  - Trades Bollinger Band statistical extremes ($z > 2.0$ or $z < -2.0$).
  - Captures carry interest rate differentials across central bank rate disparities.

