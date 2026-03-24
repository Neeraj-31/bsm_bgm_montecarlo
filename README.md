# Monte Carlo and Black-Scholes Option Pricing

This repository contains a Python script (`montecarlosim.py`) that:

- Downloads 5 years of daily Nifty 50 (^NSEI) closing prices using `yfinance`
- Computes daily log returns and annualized volatility
- Implements European call option pricing with Black-Scholes via `black_scholes_call(S, K, T, r, sigma)`
- Implements Monte Carlo pricing for a European call via `monte_carlo_call_price(S0, K, T, r, sigma, iterations=10000)`
- Visualizes:
  - first 10 simulated GBM paths (1-year horizon)
  - Monte Carlo convergence vs Black-Scholes (log-iteration scale)
- Saves plot as `nifty_horizontal.png`
- Prints a formatted summary table of results (INR labels) with percentage error and confidence interval

## Files

- `montecarlosim.py`: Main script with volatility estimation, pricing, plotting.
- `README.md`: This file.

## Installation

```bash
pip install yfinance numpy pandas scipy matplotlib tabulate
```

## Usage

```bash
python montecarlosim.py
```

The script produces:
- `option_pricing_analysis.png` with:
  - First 10 simulated GBM paths
  - Monte Carlo convergence plot (100 to 10,000 simulations) with Black-Scholes line
- Console output summary:
  - Annualized volatility
  - Black-Scholes price
  - Monte Carlo price
  - Absolute percentage error
  - Standard error and 95% CI

## Math and Theory

### Annualized Volatility

- Daily log returns: `ln(P_t / P_{t-1})`
- Daily standard deviation of log returns and then annualized using `sqrt(252)`.

### Black-Scholes Formula (European Call)

- $d_1 = \frac{\ln(S/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}$
- $d_2 = d_1 - \sigma \sqrt{T}$
- Call price: $C = S N(d_1) - K e^{-rT} N(d_2)$

where `N` is the standard normal cumulative distribution function.

### Monte Carlo Pricing

The Monte Carlo method simulates asset returns under Geometric Brownian Motion (GBM):

- $dS_t = r S_t dt + \sigma S_t dW_t$
- Terminal price: $S_T = S_0 \exp\left((r - 0.5\sigma^2)T + \sigma\sqrt{T}Z\right)$ where $Z \sim N(0,1)$
- Payoff: $\max(S_T - K, 0)$
- Price: $e^{-rT} \times \mathbb{E}[\text{payoff}]$
- Standard error: $\frac{\sigma_{\text{payoffs}}}{\sqrt{n}}$

## Convergence Behavior

Monte Carlo prices are noisy for low path counts. In tests, the average Monte Carlo estimate converges towards the Black-Scholes price within statistical error bounds. Repeated runs and higher iterations improve accuracy; the script produces a convergence plot (with error bars) to demonstrate this.

## Notes

- The script uses exact terminal distribution sampling for maximum accuracy
- Monte Carlo estimates converge to Black-Scholes as the number of simulations increases
