Options Pricing: Black-Scholes vs Monte Carlo | Python

Implemented closed-form Black-Scholes pricing and Monte Carlo GBM simulation (100k paths) for European options; demonstrated convergence to within $0.012 of analytical price
Computed all 5 Greeks and verified Put-Call Parity to zero arbitrage error across parameter scenarios
# Options Pricing: Black-Scholes vs Monte Carlo Simulation

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?logo=scipy)
![License](https://img.shields.io/badge/License-MIT-green)

A quantitative finance project that prices European options using two methods — the **Black-Scholes closed-form formula** and **Monte Carlo simulation** — and compares their accuracy, convergence behaviour, and sensitivity across market parameters.

---

## Overview

Options are financial derivatives that give the holder the right (but not the obligation) to buy or sell an asset at a fixed price (the *strike*) before a certain date (*expiry*). Pricing them correctly is fundamental to market risk management at investment banks.

This project implements and compares two core industry approaches:

| Method | Type | Accuracy | Use Case |
|---|---|---|---|
| Black-Scholes | Analytical (closed-form) | Exact | European vanilla options |
| Monte Carlo | Numerical (simulation) | Approximate (improves with N) | Exotic options, complex payoffs |

---

## Key Results

Running with benchmark parameters: **S=100, K=105, T=0.5yr, r=5%, σ=20%**

| | Call Price | Put Price |
|---|---|---|
| **Black-Scholes** | $4.5817 | $6.9892 |
| **Monte Carlo (100k paths)** | $4.5696 | $6.9927 |
| **Error** | $0.0121 | $0.0035 |

Monte Carlo converges to within **$0.01** of the analytical price — demonstrating the Law of Large Numbers in a financial context.

**Put-Call Parity verified:** `C - P = S - K·e^(-rT)` holds to zero error ✓

---

## Features

- **Black-Scholes Pricer** — Analytical call & put pricing using `scipy.stats.norm`
- **Monte Carlo Pricer** — Geometric Brownian Motion simulation with vectorised NumPy operations
- **Convergence Analysis** — Tracks MC price accuracy as simulation count increases (100 → 50,000)
- **Greeks Calculator** — Delta, Gamma, Theta, Vega, Rho for calls and puts
- **Sensitivity Table** — Call price across volatility range (10% → 40%)
- **Put-Call Parity Check** — No-arbitrage verification
- **4-Panel Visualisation** — Stock paths, convergence, price curves, Greeks

---

## The Math

### Black-Scholes Formula

For a European **call** option:

```
C = S·N(d1) - K·e^(-rT)·N(d2)

where:
  d1 = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
  d2 = d1 - σ·√T
  N(·) = cumulative standard normal distribution
```

For a European **put** option (via Put-Call Parity):

```
P = K·e^(-rT)·N(-d2) - S·N(-d1)
```

### Monte Carlo — Geometric Brownian Motion

Stock price evolves as:

```
S(t+dt) = S(t) · exp[ (r - σ²/2)·dt + σ·√dt·Z ]

where Z ~ N(0,1) is a random shock at each time step
```

At expiry, payoffs are discounted back to present value:
```
Call payoff = max(S_T - K, 0)
Put  payoff = max(K - S_T, 0)
Price       = e^(-rT) · E[payoff]
```

---

## The Greeks

| Greek | Definition | Risk Managed |
|---|---|---|
| **Delta** | dC/dS — price sensitivity to stock move | Directional risk |
| **Gamma** | d²C/dS² — rate of change of Delta | Convexity / re-hedging cost |
| **Theta** | dC/dt — time decay per day | Time risk |
| **Vega** | dC/dσ — sensitivity to volatility | Volatility risk |
| **Rho** | dC/dr — sensitivity to interest rate | Rate risk |

Results for benchmark parameters (Call):
```
Delta :  0.4612   ← moves $0.46 per $1 stock move
Gamma :  0.0281   ← Delta changes by 0.028 per $1 stock move
Theta : -0.0211   ← loses $0.021 per day (time decay)
Vega  :  0.2808   ← gains $0.28 per 1% volatility increase
Rho   :  0.2077   ← gains $0.21 per 1% rate increase
```

---

## Output Chart

![Options Pricing Analysis](options_pricing_analysis.png)

Four panels:
1. **Simulated GBM stock paths** with strike price reference line
2. **MC convergence** to Black-Scholes price as simulations increase
3. **Option price curves** vs underlying stock price (with intrinsic value)
4. **Delta & Gamma** vs stock price showing non-linear risk profile

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/tanusreesaha/options-pricing.git
cd options-pricing

# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python options_pricing.py
```

**requirements.txt**
```
numpy>=1.24
scipy>=1.10
pandas>=1.5
matplotlib>=3.7
```

---

## Project Structure

```
options-pricing/
├── options_pricing.py          # Main script — all models and analysis
├── options_pricing_analysis.png  # Output chart (auto-generated)
├── requirements.txt
└── README.md
```

---

## Why This Matters for Risk Management

Banks use Monte Carlo extensively in their risk engines because:

- **Exotic options** (barrier, Asian, lookback) have no closed-form solution
- **Portfolio VaR** requires simulating thousands of correlated risk factor paths
- **Basel III/IV** capital requirements rely on internal models built on MC methods

This project demonstrates the foundational logic behind those systems — taking a simple European option and showing how the numerical simulation method scales to any payoff structure.

---

## Author

**Tanusree Saha**
Quantitative Finance | Python | Risk Modeling
📧 11tanusreesaha@gmail.com

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Hull, J. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- MIT OpenCourseWare — 18.S096: Topics in Mathematics with Applications in Finance
