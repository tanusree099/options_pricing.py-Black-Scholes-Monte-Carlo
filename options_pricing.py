"""
=====================================================================
  OPTIONS PRICING: Black-Scholes Model vs. Monte Carlo Simulation
  Author : Tanusree Saha
  Purpose: Quantitative Finance Portfolio Project
=====================================================================

WHAT THIS PROJECT DOES:
    1. Prices European Call & Put options using the Black-Scholes
       closed-form formula (analytical solution).
    2. Prices the same options using Monte Carlo simulation
       (numerical approximation via random paths).
    3. Compares both methods and visualises:
         - Simulated stock price paths
         - Option price convergence as simulations increase
         - Greeks (Delta, Gamma, Theta, Vega, Rho)

FINANCIAL CONCEPTS COVERED:
    - Black-Scholes Model assumptions & formula
    - Risk-neutral pricing
    - Monte Carlo simulation for derivative pricing
    - The Greeks: sensitivity of option price to parameters
    - Put-Call Parity verification

WHY THIS MATTERS FOR RISK MANAGEMENT:
    Options pricing is the foundation of market risk. Banks like UBS
    use Monte Carlo extensively for exotic derivatives and VaR
    calculations where closed-form solutions don't exist.
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  1. BLACK-SCHOLES ANALYTICAL PRICER
# ─────────────────────────────────────────────

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes closed-form option pricing formula.

    Parameters
    ----------
    S     : float  – Current stock price
    K     : float  – Strike price
    T     : float  – Time to expiry in years (e.g. 0.5 = 6 months)
    r     : float  – Risk-free interest rate (annual, decimal)
    sigma : float  – Volatility of the underlying (annual, decimal)
    option_type : str – 'call' or 'put'

    Returns
    -------
    price : float  – Theoretical option price
    d1    : float  – d1 parameter (used for Greeks)
    d2    : float  – d2 parameter
    """
    # d1 and d2: core of the Black-Scholes formula
    # d1 captures the probability-adjusted moneyness + time value
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        # Call = S*N(d1) - K*e^(-rT)*N(d2)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        # Put  = K*e^(-rT)*N(-d2) - S*N(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price, d1, d2


# ─────────────────────────────────────────────
#  2. THE GREEKS
# ─────────────────────────────────────────────

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Compute the 5 main option Greeks.

    Greeks measure sensitivity of option price to each parameter.
    Critical for risk hedging desks at investment banks.

    Delta : dPrice/dS      – How much price moves per $1 move in stock
    Gamma : d²Price/dS²    – Rate of change of Delta (convexity risk)
    Theta : dPrice/dT      – Time decay per day
    Vega  : dPrice/dSigma  – Sensitivity to volatility (per 1% move)
    Rho   : dPrice/dr      – Sensitivity to interest rate (per 1% move)
    """
    _, d1, d2 = black_scholes_price(S, K, T, r, sigma, option_type)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1   # Put delta is negative

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta (annualised, divide by 365 for daily)
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (per 1% change in volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01

    # Rho (per 1% change in interest rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta,
            'Vega': vega, 'Rho': rho}


# ─────────────────────────────────────────────
#  3. MONTE CARLO PRICER
# ─────────────────────────────────────────────

def monte_carlo_price(S, K, T, r, sigma, option_type='call',
                      n_simulations=100_000, n_steps=252, seed=42):
    """
    Price a European option using Monte Carlo simulation.

    HOW IT WORKS:
        Under risk-neutral pricing, stock price follows Geometric
        Brownian Motion (GBM):
            S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        where Z ~ N(0,1) is a random shock.

        We simulate thousands of possible future stock paths,
        compute the payoff at expiry for each path, then
        discount back to today using the risk-free rate.

    Parameters
    ----------
    n_simulations : int – Number of random paths (more = more accurate)
    n_steps       : int – Time steps per path (252 = trading days/year)
    seed          : int – For reproducibility

    Returns
    -------
    price          : float – Monte Carlo option price
    std_error      : float – Standard error (measure of accuracy)
    paths          : ndarray – Sample of simulated stock price paths
    payoffs        : ndarray – Payoff for each simulation
    """
    np.random.seed(seed)
    dt = T / n_steps

    # Simulate GBM paths using vectorised operations
    # Shape: (n_steps, n_simulations)
    Z = np.random.standard_normal((n_steps, n_simulations))
    # Increment: drift + diffusion
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    # Cumulative product to get price paths
    price_paths = S * np.exp(np.cumsum(increments, axis=0))
    # Prepend starting price
    price_paths = np.vstack([np.full(n_simulations, S), price_paths])

    # Final prices at expiry
    S_T = price_paths[-1]

    # Compute payoffs
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)   # max(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)   # max(K - S_T, 0)

    # Discount payoffs back to present value
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)

    return price, std_error, price_paths, payoffs


# ─────────────────────────────────────────────
#  4. CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────

def convergence_analysis(S, K, T, r, sigma, option_type='call',
                         max_sims=50_000):
    """
    Show how Monte Carlo price converges to Black-Scholes
    as the number of simulations increases.

    This illustrates the Law of Large Numbers in practice.
    """
    bs_price, _, _ = black_scholes_price(S, K, T, r, sigma, option_type)

    sim_counts = np.logspace(2, np.log10(max_sims), 30, dtype=int)
    mc_prices = []

    for n in sim_counts:
        price, _, _, _ = monte_carlo_price(S, K, T, r, sigma,
                                           option_type, n_simulations=n)
        mc_prices.append(price)

    return sim_counts, mc_prices, bs_price


# ─────────────────────────────────────────────
#  5. PUT-CALL PARITY CHECK
# ─────────────────────────────────────────────

def put_call_parity_check(S, K, T, r, sigma):
    """
    Verify Put-Call Parity: C - P = S - K*e^(-rT)

    This is a fundamental no-arbitrage relationship.
    If it breaks, there's a risk-free profit opportunity.
    """
    call_price, _, _ = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price,  _, _ = black_scholes_price(S, K, T, r, sigma, 'put')

    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    parity_error = abs(lhs - rhs)

    return {
        'Call Price'            : round(call_price, 4),
        'Put Price'             : round(put_price, 4),
        'C - P'                 : round(lhs, 4),
        'S - K*e^(-rT)'         : round(rhs, 4),
        'Parity Error'          : round(parity_error, 8),
        'Parity Holds'          : parity_error < 1e-6
    }


# ─────────────────────────────────────────────
#  6. VISUALISATION
# ─────────────────────────────────────────────

def plot_results(S, K, T, r, sigma, n_paths_to_plot=50):
    """
    Generate a 4-panel summary figure.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0D1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    COLOR_CALL  = '#00C896'
    COLOR_PUT   = '#FF6B6B'
    COLOR_BS    = '#FFD700'
    COLOR_GRID  = '#2A2A3A'
    COLOR_TEXT  = '#E0E0E0'
    BG          = '#0D1117'
    PANEL_BG    = '#161B22'

    # ── Panel 1: Simulated Stock Price Paths ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    _, _, paths, _ = monte_carlo_price(S, K, T, r, sigma, n_simulations=n_paths_to_plot, seed=1)
    time_axis = np.linspace(0, T, paths.shape[0])
    for i in range(n_paths_to_plot):
        ax1.plot(time_axis, paths[:, i], alpha=0.3, linewidth=0.7, color='#4A9EFF')
    ax1.axhline(K, color=COLOR_BS, linewidth=1.5, linestyle='--', label=f'Strike K={K}')
    ax1.set_title('Simulated Stock Price Paths (GBM)', color=COLOR_TEXT, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (years)', color=COLOR_TEXT, fontsize=9)
    ax1.set_ylabel('Stock Price ($)', color=COLOR_TEXT, fontsize=9)
    ax1.tick_params(colors=COLOR_TEXT)
    ax1.grid(True, color=COLOR_GRID, linewidth=0.5)
    ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=COLOR_TEXT)
    for spine in ax1.spines.values(): spine.set_color(COLOR_GRID)

    # ── Panel 2: Convergence to Black-Scholes ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    sims_c, mc_c, bs_c = convergence_analysis(S, K, T, r, sigma, 'call')
    sims_p, mc_p, bs_p = convergence_analysis(S, K, T, r, sigma, 'put')
    ax2.semilogx(sims_c, mc_c, color=COLOR_CALL, linewidth=1.5, label='MC Call')
    ax2.semilogx(sims_p, mc_p, color=COLOR_PUT,  linewidth=1.5, label='MC Put')
    ax2.axhline(bs_c, color=COLOR_CALL, linestyle='--', linewidth=1, alpha=0.7, label=f'BS Call = ${bs_c:.2f}')
    ax2.axhline(bs_p, color=COLOR_PUT,  linestyle='--', linewidth=1, alpha=0.7, label=f'BS Put  = ${bs_p:.2f}')
    ax2.set_title('Monte Carlo Convergence to Black-Scholes', color=COLOR_TEXT, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Simulations (log scale)', color=COLOR_TEXT, fontsize=9)
    ax2.set_ylabel('Option Price ($)', color=COLOR_TEXT, fontsize=9)
    ax2.tick_params(colors=COLOR_TEXT)
    ax2.grid(True, color=COLOR_GRID, linewidth=0.5)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=COLOR_TEXT)
    for spine in ax2.spines.values(): spine.set_color(COLOR_GRID)

    # ── Panel 3: Option Price vs Stock Price ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL_BG)
    stock_range = np.linspace(S * 0.5, S * 1.5, 200)
    call_prices = [black_scholes_price(s, K, T, r, sigma, 'call')[0] for s in stock_range]
    put_prices  = [black_scholes_price(s, K, T, r, sigma, 'put')[0]  for s in stock_range]
    intrinsic_call = np.maximum(stock_range - K, 0)
    intrinsic_put  = np.maximum(K - stock_range, 0)
    ax3.plot(stock_range, call_prices, color=COLOR_CALL, linewidth=2, label='Call (BS)')
    ax3.plot(stock_range, put_prices,  color=COLOR_PUT,  linewidth=2, label='Put (BS)')
    ax3.plot(stock_range, intrinsic_call, color=COLOR_CALL, linestyle=':', linewidth=1, alpha=0.5, label='Call Intrinsic')
    ax3.plot(stock_range, intrinsic_put,  color=COLOR_PUT,  linestyle=':', linewidth=1, alpha=0.5, label='Put Intrinsic')
    ax3.axvline(K, color=COLOR_BS, linestyle='--', linewidth=1, alpha=0.7, label=f'Strike K={K}')
    ax3.axvline(S, color='white',   linestyle='--', linewidth=1, alpha=0.5, label=f'Spot S={S}')
    ax3.set_title('Option Price vs Underlying Stock Price', color=COLOR_TEXT, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Stock Price ($)', color=COLOR_TEXT, fontsize=9)
    ax3.set_ylabel('Option Price ($)', color=COLOR_TEXT, fontsize=9)
    ax3.tick_params(colors=COLOR_TEXT)
    ax3.grid(True, color=COLOR_GRID, linewidth=0.5)
    ax3.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=COLOR_TEXT)
    for spine in ax3.spines.values(): spine.set_color(COLOR_GRID)

    # ── Panel 4: Greeks vs Stock Price ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)
    deltas = [compute_greeks(s, K, T, r, sigma, 'call')['Delta'] for s in stock_range]
    gammas = [compute_greeks(s, K, T, r, sigma, 'call')['Gamma'] * 100 for s in stock_range]
    ax4.plot(stock_range, deltas, color=COLOR_CALL, linewidth=2, label='Delta (Call)')
    ax4.plot(stock_range, gammas, color='#FF9F43', linewidth=2, label='Gamma × 100')
    ax4.axvline(K, color=COLOR_BS, linestyle='--', linewidth=1, alpha=0.7)
    ax4.axvline(S, color='white',   linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title('Delta & Gamma vs Stock Price', color=COLOR_TEXT, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Stock Price ($)', color=COLOR_TEXT, fontsize=9)
    ax4.set_ylabel('Greek Value', color=COLOR_TEXT, fontsize=9)
    ax4.tick_params(colors=COLOR_TEXT)
    ax4.grid(True, color=COLOR_GRID, linewidth=0.5)
    ax4.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=COLOR_TEXT)
    for spine in ax4.spines.values(): spine.set_color(COLOR_GRID)

    fig.suptitle('Options Pricing: Black-Scholes vs Monte Carlo Simulation',
                 color=COLOR_TEXT, fontsize=14, fontweight='bold', y=1.01)

    plt.savefig('/mnt/user-data/outputs/options_pricing_analysis.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("Chart saved.")


# ─────────────────────────────────────────────
#  7. MAIN — RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Parameters (realistic market scenario) ──
    S     = 100.0    # Current stock price: $100
    K     = 105.0    # Strike price: slightly out-of-the-money call
    T     = 0.5      # 6 months to expiry
    r     = 0.05     # 5% risk-free rate (approx US treasury)
    sigma = 0.20     # 20% annual volatility (typical for large-cap)

    print("=" * 60)
    print("  OPTIONS PRICING: Black-Scholes vs Monte Carlo")
    print("=" * 60)
    print(f"\n  Parameters: S={S}, K={K}, T={T}yr, r={r*100}%, σ={sigma*100}%\n")

    # ── Black-Scholes Results ──
    call_bs, d1, d2 = black_scholes_price(S, K, T, r, sigma, 'call')
    put_bs,  _,  _  = black_scholes_price(S, K, T, r, sigma, 'put')

    print("── Black-Scholes Analytical Prices ──")
    print(f"  Call Price : ${call_bs:.4f}")
    print(f"  Put Price  : ${put_bs:.4f}")
    print(f"  d1 = {d1:.4f}  |  d2 = {d2:.4f}")

    # ── Monte Carlo Results ──
    print("\n── Monte Carlo Simulation (100,000 paths) ──")
    call_mc, call_se, _, _ = monte_carlo_price(S, K, T, r, sigma, 'call', 100_000)
    put_mc,  put_se,  _, _ = monte_carlo_price(S, K, T, r, sigma, 'put',  100_000)

    print(f"  Call Price : ${call_mc:.4f}  ±  {call_se:.4f}  (std error)")
    print(f"  Put Price  : ${put_mc:.4f}  ±  {put_se:.4f}  (std error)")
    print(f"  Call Error vs BS: ${abs(call_mc - call_bs):.4f}")
    print(f"  Put  Error vs BS: ${abs(put_mc  - put_bs):.4f}")

    # ── Greeks ──
    print("\n── Option Greeks (Call) ──")
    greeks = compute_greeks(S, K, T, r, sigma, 'call')
    for name, val in greeks.items():
        print(f"  {name:6s}: {val:.6f}")

    # ── Put-Call Parity ──
    print("\n── Put-Call Parity Verification ──")
    parity = put_call_parity_check(S, K, T, r, sigma)
    for k, v in parity.items():
        print(f"  {k:20s}: {v}")

    # ── Sensitivity Table ──
    print("\n── Sensitivity: Call Price vs Volatility ──")
    print(f"  {'Volatility':>12}  {'BS Call':>10}  {'MC Call':>10}  {'Difference':>12}")
    for vol in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        bs  = black_scholes_price(S, K, T, r, vol, 'call')[0]
        mc, _, _, _ = monte_carlo_price(S, K, T, r, vol, 'call', 50_000)
        print(f"  {vol*100:>10.0f}%  ${bs:>9.4f}  ${mc:>9.4f}  ${abs(bs-mc):>11.4f}")

    # ── Plot ──
    print("\n── Generating Charts ──")
    plot_results(S, K, T, r, sigma)
    print("\nDone. Chart saved to outputs folder.")
