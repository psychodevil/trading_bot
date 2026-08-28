# 🏛️ QuantumAlpha System Architecture

QuantumAlpha is an institutional-grade, event-driven quantitative trading framework. It formulates trading decisions as a continuous-state, discrete-time stochastic control problem where asset allocation weights are jointly optimized under heavy-tailed return uncertainty and non-linear transaction frictions.

---

## 1. Core Mathematical Framework

### 1.1 Probability Distribution Formulation

Given historical filtration $\mathcal{F}_t$, the forward return of asset $i$ over horizon $H$ is modeled as a generalized Student-$t$ distribution:

$$P\left(R_{i, t \to t+H} \mid \mathcal{F}_t\right) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi(\nu-2)}\sigma_i \Gamma\left(\frac{\nu}{2}\right)} \left[ 1 + \frac{(R - \mu_i)^2}{(\nu-2)\sigma_i^2} \right]^{-\frac{\nu+1}{2}}$$

- **Drift $\mu_i$**: Dynamically estimated from multi-horizon trend alignment ($\text{EMA}_{20}, \text{EMA}_{50}, \text{EMA}_{100}$), RSI momentum, and Bollinger Band $z$-score deviations.
- **Scale $\sigma_i$**: Estimated via exponential-weighted moving average (EWMA) and localized ATR volatility clustering.
- **Degrees of Freedom $\nu_i$**: Dynamically parameterized by rolling sample excess kurtosis $\kappa$:

$$\nu_i = \max\left(4.1, \; \frac{6}{\kappa_i} + 4\right)$$

---

### 1.2 Non-Linear Transaction & Holding Cost Engine

Rebalancing from current weight $w_{\text{curr}}$ to target weight $w_{\text{target}}$ incurs multi-factor execution friction:

$$\mathcal{C}(\Delta w) = \underbrace{c_{\text{linear}} \cdot |\Delta w|}_{\text{Exchange Fees}} + \underbrace{\frac{s_{\text{spread}}}{2} \cdot |\Delta w|}_{\text{Bid-Ask Crossing}} + \underbrace{\eta \cdot |\Delta w|^{1.5}}_{\text{Square-Root Market Impact}}$$

In addition, holding positions over interval $\Delta t$ incurs continuous carry costs:

$$\mathcal{C}_{\text{holding}}(w, \Delta t) = \left( r_{\text{borrow}} \cdot \mathbb{I}_{w < 0} + r_{\text{funding}} + \theta_{\text{options}} \right) |w| \Delta t$$

---

### 1.3 Analytical Inaction / No-Trade Boundary Derivation

To prevent fee drag from small noisy price shifts, the optimizer solves for the inaction half-width $\Delta w_{\text{inaction}}$:

$$\Delta w_{\text{inaction}} = \frac{c_{\text{fee}} + s/2}{\gamma \cdot \sigma^2 + \epsilon}$$

$$\text{Inaction Zone} = [\underline{w}, \bar{w}] = \left[ w^* - \Delta w_{\text{inaction}}, \; w^* + \Delta w_{\text{inaction}} \right]$$

If $w_{\text{curr}} \in [\underline{w}, \bar{w}]$, the trade is suppressed, saving $100\%$ of turnover costs.

---

## 2. $O(1)$ Stateful Indicator Engine

Traditional backtesting engines recompute historical indicators on every bar, leading to $O(N^2)$ quadratic slowdowns. QuantumAlpha implements an $O(1)$ stateful online update pipeline:

- **EMA Update**: $\text{EMA}_t = \alpha P_t + (1 - \alpha)\text{EMA}_{t-1}$
- **Wilder ATR Update**: $\text{ATR}_t = \frac{\text{ATR}_{t-1} \cdot 13 + \text{TR}_t}{14}$
- **Wilder RSI Update**: $\text{AvgGain}_t = \frac{\text{AvgGain}_{t-1} \cdot 13 + \text{Gain}_t}{14}$
- **Circular Variance Ring Buffer**: Updates mean and variance in $O(1)$ memory operations.

This architecture enables evaluating 210,000+ bars across 62 assets in **under 75 seconds**.
