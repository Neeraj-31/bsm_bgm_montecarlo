# Monte Carlo and Black-Scholes Option Pricing

This repository contains a Python script (`montecarlosim.py`) that:

- Downloads 5 years of daily Nifty 50 (^NSEI) closing prices using `yfinance`
- Computes daily log returns and annualized volatility
- Implements European call option pricing with Black-Scholes
- Implements Monte Carlo pricing for a European call using GBM
- Visualizes GBM sample paths and Monte Carlo convergence
- Prints a formatted summary table of results

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
- Discretized increment: $S_{t+\Delta t} = S_t \exp\left((r - 0.5\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\right)$

The script supports two methods:
- `exact`: direct sample of terminal price $S_T$ via $\ln S_T \sim N(\ln S_0 + (r - 0.5\sigma^2)T,\, \sigma^2 T)$
- `vectorized`: simulate daily path in full using numpy vectorization

Payoff at maturity: $\max(S_T - K, 0)$, then discount with $e^{-rT}$.

## Convergence Behavior

Monte Carlo prices are noisy for low path counts. In tests, the average Monte Carlo estimate converges towards the Black-Scholes price within statistical error bounds. Repeated runs and higher iterations improve accuracy; the script produces a convergence plot (with error bars) to demonstrate this.

## Notes

- If you test the vectorized `monte_carlo_call_price(..., method='vectorized')` and `method='loop'`, they should produce similar results for high iterations.
- `method='exact'` gives the most accurate match to analytic Black-Scholes in expectation.
