import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


ticker = "^NSEI"
data = yf.download(ticker, period="5y", progress=False)

closing_prices = data['Close']

log_returns = np.log(closing_prices / closing_prices.shift(1))
log_returns = log_returns.dropna()

daily_volatility = float(log_returns.std())
annualized_volatility = float(daily_volatility * np.sqrt(252))


print("\n" + "-"*50)
print("Nifty 50 (^NSEI) Volatility Analysis")
print("-"*50)
print(f"Period: Last 5 years")
print(f"Number of observations: {len(log_returns)}")
print(f"Daily Volatility: {daily_volatility:.6f}")
print(f"Annualized Volatility (%): {annualized_volatility * 100:.2f}%")
print("-"*50)


# =====================================================
# Black-Scholes Call Option Pricing Function
# =====================================================

def black_scholes_call(S, K, T, r, sigma):
   
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate call price using scipy.stats.norm CDF
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price


# =====================================================
# Monte Carlo Call Option Pricing Function
# =====================================================

def monte_carlo_call_price(S0, K, T, r, sigma, iterations=10000, method='exact'):
    """
    Price a European Call option using Monte Carlo simulation.

    method:
      - 'exact': simulate S_T directly from lognormal terminal distribution (most accurate)
      - 'vectorized': simulate path using vectorized GBM steps
      - 'loop': simulate path using a pure Python loop (reference)
    
    Returns:
    --------
    tuple
      (call_price, std_error)
    """
    if method == 'exact':
        z = np.random.standard_normal(iterations)
        S_final = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    else:
        dt = 1 / 252
        steps = int(T * 252)

        if method == 'vectorized':
            Z = np.random.standard_normal((iterations, steps))
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z
            increments = np.exp(drift + diffusion)
            S_paths = S0 * np.cumprod(increments, axis=1)
            S_final = S_paths[:, -1]

        elif method == 'loop':
            S_final = np.full(iterations, S0)
            for i in range(steps):
                Z_step = np.random.standard_normal(iterations)
                S_final *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_step)

        else:
            raise ValueError("method must be 'exact', 'vectorized', or 'loop'")

    payoffs = np.maximum(S_final - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs, ddof=1) / np.sqrt(iterations)

    return call_price, std_error


# Example usage of Black-Scholes
if __name__ == "__main__":
    # Parameters
    S_0 = 100  # Current spot price
    K = S_0    # Strike price equals current spot price (At-the-money)
    T = 1      # 1 year to expiration
    r = 0.07   # Risk-free rate: 7%
    sigma = annualized_volatility  # Use calculated volatility from Nifty 50
    
    # Calculate call option price using Black-Scholes
    call_price_bs = black_scholes_call(S_0, K, T, r, sigma)
    
    # Calculate call option price using Monte Carlo (exact terminal simulation)
    call_price_mc, std_error_mc = monte_carlo_call_price(
        S_0, K, T, r, sigma, iterations=10000, method='exact'
    )
    
    # Print Black-Scholes results
    print("\n" + "-"*50)
    print("Black-Scholes Call Option Pricing")
    print("-"*50)
    print(f"Spot Price (S): ${S_0:.2f}")
    print(f"Strike Price (K): ${K:.2f}")
    print(f"Time to Expiration (T): {T} year(s)")
    print(f"Risk-free Rate (r): {r*100:.1f}%")
    print(f"Volatility (sigma): {sigma*100:.2f}%")
    print("-"*50)
    print(f"Call Option Price (BS): ${call_price_bs:.4f}")
    print("-"*50)
    
    # Print Monte Carlo results
    print("\n" + "-"*50)
    print("Monte Carlo Call Option Pricing (10,000 simulations)")
    print("-"*50)
    print(f"Call Option Price (MC): ${call_price_mc:.4f}")
    print(f"Standard Error: ${std_error_mc:.4f}")
    print(f"95% CI: ${call_price_mc - 1.96*std_error_mc:.4f} - ${call_price_mc + 1.96*std_error_mc:.4f}")
    print("-"*50)
    
    # Compare results
    print("\n" + "-"*50)
    print("Comparison")
    print("-"*50)
    print(f"Black-Scholes Price: ${call_price_bs:.4f}")
    print(f"Monte Carlo Price:   ${call_price_mc:.4f}")
    print(f"Difference:          ${abs(call_price_bs - call_price_mc):.4f}")
    print(f"Difference (%):      {abs(call_price_bs - call_price_mc) / call_price_bs * 100:.2f}%")
    print("-"*50)
    
    # =====================================================
    # Plotting Section
    # =====================================================
    
    # Plot 1: First 10 GBM price paths
   
    prices_plot = np.zeros((10, int(T * 252) + 1))
    prices_plot[:, 0] = S_0
    
    dt = 1 / 252
    steps = int(T * 252)
    Z_plot = np.random.standard_normal((10, steps))
    
    for i in range(steps):
        prices_plot[:, i + 1] = prices_plot[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_plot[:, i]
        )
    
    # Time array for x-axis (in years)
    time_array = np.linspace(0, T, int(T * 252) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: GBM Paths
    for i in range(10):
        ax1.plot(time_array, prices_plot[i, :], alpha=0.7, linewidth=1.5)
    ax1.axhline(y=K, color='r', linestyle='--', linewidth=2, label=f'Strike Price (K=${K:.0f})')
    ax1.set_xlabel('Time (Years)', fontsize=11)
    ax1.set_ylabel('Stock Price ($)', fontsize=11)
    ax1.set_title('First 10 Simulated GBM Price Paths (1-Year Horizon)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Monte Carlo Convergence
    iteration_counts = np.array([100, 250, 500, 1000, 2500, 5000, 10000])
    mc_prices = []
    mc_std = []

    np.random.seed(42)  # reproducible convergence behavior
    for n_iter in iteration_counts:
        # Averaging over repeats improves convergence display
        repeats = 20
        prices_repeat = []
        for _ in range(repeats):
            price, _ = monte_carlo_call_price(S_0, K, T, r, sigma, iterations=n_iter, method='exact')
            prices_repeat.append(price)
        avg_price = np.mean(prices_repeat)
        mc_prices.append(avg_price)
        mc_std.append(np.std(prices_repeat, ddof=1))

    ax2.errorbar(iteration_counts, mc_prices, yerr=np.array(mc_std)/np.sqrt(repeats),
                 fmt='b-o', linewidth=2, markersize=6, capsize=3, label='MC price (mean of repeats)')
    ax2.axhline(y=call_price_bs, color='r', linestyle='--', linewidth=2, label=f'Black-Scholes Price (${call_price_bs:.4f})')
    ax2.fill_between(iteration_counts, call_price_bs - 0.02, call_price_bs + 0.02,
                     alpha=0.2, color='red', label='BS +/- 0.02')
    ax2.set_xlabel('Number of Simulations', fontsize=11)
    ax2.set_ylabel('Call Option Price ($)', fontsize=11)
    ax2.set_title('Monte Carlo Convergence to Black-Scholes Price', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('option_pricing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =====================================================
    # Summary Table
    # =====================================================
    
    # Calculate absolute percentage error
    abs_pct_error = abs(call_price_bs - call_price_mc) / call_price_bs * 100
    
    # Create summary table using formatted f-strings
    print("\n" + "="*70)
    print("SUMMARY TABLE: European Call Option Pricing Analysis")
    print("="*70)
    
    summary_data = [
        ("Annualized Volatility", f"{sigma*100:.2f}%"),
        ("Black-Scholes Price", f"${call_price_bs:.4f}"),
        ("Monte Carlo Price (10,000 paths)", f"${call_price_mc:.4f}"),
        ("Absolute Percentage Error", f"{abs_pct_error:.4f}%"),
    ]
    
    # Print header
    print(f"{'Metric':<35} {'Value':>30}")
    print("-"*70)
    
    # Print rows
    for metric, value in summary_data:
        print(f"{metric:<35} {value:>30}")
    
    print("="*70)
    
   

