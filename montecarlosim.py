import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

data = yf.download("^NSEI", period="5y", progress=False)

closing_prices = data['Close']

log_returns = np.log(closing_prices / closing_prices.shift(1))
log_returns = log_returns.dropna()

daily_volatility = log_returns.std()
annualized_volatility = daily_volatility * np.sqrt(252)

print("\n" + "-"*50)
print(f"Annualized Volatility of Nifty 50 calculated through historical data (%): {annualized_volatility.item() * 100:.2f}%")
print("-"*50)



# Black-Scholes Call Option Pricing Function


def black_scholes_call(S, K, T, r, sigma):

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price



# Black-Scholes Delta Function

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta



# Monte Carlo Call Option Pricing Function


def monte_carlo_call_price(S0, K, T, r, sigma, iterations=10000):

    z = np.random.standard_normal(iterations)
    S_final = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoffs = np.maximum(S_final - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs, ddof=1) / np.sqrt(iterations)

    return call_price, std_error



# Monte Carlo Delta Hedging Simulation

def simulate_delta_hedging(S0, K, T, r, sigma, num_simulations=1000, num_steps=252):
    dt = T / num_steps


    stock_prices = np.zeros((num_simulations, num_steps + 1))
    stock_prices[:, 0] = S0

    cash_account = np.zeros(num_simulations)
    shares_held = np.zeros(num_simulations)
    initial_delta = black_scholes_delta(S0, K, T, r, sigma)
    shares_held[:] = initial_delta
    cash_account -= shares_held * S0
    z = np.random.standard_normal((num_simulations, num_steps))

    for j in range(num_steps):
        time_remaining = T - (j + 1) * dt
        drift = (r - 0.5 * sigma**2) * dt
        volatility_term = sigma * np.sqrt(dt) * z[:, j]
        stock_prices[:, j+1] = stock_prices[:, j] * np.exp(drift + volatility_term)
        if time_remaining > 0:

            new_delta = black_scholes_delta(stock_prices[:, j+1], K, time_remaining, r, sigma)
        else:
            new_delta = np.zeros(num_simulations)

        delta_change = new_delta - shares_held
        cash_account -= delta_change * stock_prices[:, j+1]
        shares_held = new_delta
        cash_account *= np.exp(r * dt)
    final_stock_prices = stock_prices[:, num_steps]
    option_payoff = np.maximum(final_stock_prices - K, 0)
    final_portfolio_value = cash_account + shares_held * final_stock_prices - option_payoff

    mean_hedging_pnl = np.mean(final_portfolio_value)
    std_hedging_pnl = np.std(final_portfolio_value)

    return mean_hedging_pnl, std_hedging_pnl, final_portfolio_value



S_0 = closing_prices.values[0].item()
K = S_0
T = 1
r = 0.07
sigma = annualized_volatility.item()

call_price_bs = black_scholes_call(S_0, K, T, r, sigma)
np.random.seed(42)
call_price_mc, std_error_mc = monte_carlo_call_price(
    S_0, K, T, r, sigma, iterations=10000
)

np.random.seed(42)
mean_hedging_pnl, std_hedging_pnl, final_pnl_values = simulate_delta_hedging(S_0, K, T, r, sigma, num_simulations=10000, num_steps=252 * 10) # Changed num_steps here

# Black-Scholes results
print("\n" + "-"*50)
print("Black-Scholes Call Option Pricing")
print("-"*50)
print(f"Initial Stock Price (S): INR {S_0:.2f}")
print(f"Strike Price (K): INR {K:.2f}")
print(f"Time to Expiration (T): {T} year(s)")
print(f"Risk-free Rate (r): {r*100:.1f}%")
print(f"Volatility (sigma): {sigma*100:.2f}%")
print("-"*50)
print(f"Call Option Price (BS): INR {call_price_bs:.4f}")
print("-"*50)

# Monte Carlo results
print("\n" + "-"*50)
print("Monte Carlo Call Option Pricing (10,000 simulations)")
print("-"*50)
print(f"Call Option Price (MC): INR {call_price_mc:.4f}")
print("-"*50)


print("\n" + "-"*50)
print("Comparison")
print("-"*50)
print(f"Black-Scholes Price: INR {call_price_bs:.4f}")
print(f"Monte Carlo Price:   INR {call_price_mc:.4f}")
print(f"Difference:          INR {abs(call_price_bs - call_price_mc):.4f}")
print(f"Difference (%):      {abs(call_price_bs - call_price_mc) / call_price_bs * 100:.2f}%")
print("-"*50)


# Plotting Section


iteration_counts = np.array([100, 500, 1000, 2500, 5000, 10000, 25000])
mc_prices_line = []
for n in iteration_counts:
    price, _ = monte_carlo_call_price(S_0, K, T, r, sigma, iterations=n)
    mc_prices_line.append(price)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#  10 Paths
dt = 1 / 252
steps = int(T * 252)
paths = np.zeros((10, steps + 1))
paths[:, 0] = S_0
for i in range(10):
    paths[i, 1:] = paths[i, 0] * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(steps)))

time_axis = np.linspace(0, T, steps + 1)
for i in range(10):
    ax1.plot(time_axis, paths[i, :], alpha=0.7, linewidth=1)

ax1.axhline(y=K, color='r', linestyle='--', label='Strike')
ax1.set_title('10 Simulated GBM Paths', fontsize=10, fontweight='bold')
ax1.set_xlabel('Years', fontsize=9)
ax1.set_ylabel('Price', fontsize=9)
ax1.tick_params(labelsize=8)
ax1.grid(True, alpha=0.2)

#  Convergence
ax2.plot(iteration_counts, mc_prices_line, marker='o', markersize=3, color='blue', label='MC')
ax2.axhline(y=call_price_bs, color='red', linestyle='-', label='BSM')
ax2.set_xscale('log')
ax2.set_title('MC Convergence to BSM', fontsize=10, fontweight='bold')
ax2.set_xlabel('Iterations (Log)', fontsize=9)
ax2.set_ylabel('Option Price', fontsize=9)
ax2.tick_params(labelsize=8)
ax2.grid(True, which="both", alpha=0.2)
y_min = int(call_price_bs - 100)
y_max = int(call_price_bs + 100)
ax2.set_yticks(np.arange(y_min, y_max + 1, 20))
ax2.minorticks_on()
ax2.grid(True, which='minor', axis='y', linestyle=':', alpha=0.2)
ax2.grid(True, which='major', axis='y', linestyle='-', alpha=0.4)
ax2.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('nifty_horizontal.png', dpi=150)
plt.show()

# Summary Table


abs_pct_error = abs(call_price_bs - call_price_mc) / call_price_bs * 100


print("\n" + "="*70)
print("SUMMARY TABLE: European Call Option Pricing Analysis")
print("="*70)

summary_data = [("Calculated Volatility (%)",f"{sigma*100:.2f}%"),
    ("Black-Scholes Price", f"INR {call_price_bs:.4f}"),
    ("Monte Carlo Price (10,000 paths)", f"INR {call_price_mc:.4f}"),
    ("Absolute Percentage Error", f"{abs_pct_error:.4f}%"),
    ("Delta P&L (Mean)", f"INR {mean_hedging_pnl:.4f}"),
    ("Delta P&L (Std Dev)", f"INR {std_hedging_pnl:.4f}")
]


print(f"{'Metric':<35} {'Value':>30}")
print("-"*70)

for metric, value in summary_data:
    print(f"{metric:<35} {value:>30}")

print("="*70)
