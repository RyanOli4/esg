# ESG: Economic Scenario Generator

Welcome! This project is a lightweight, code-first implementation of an **Economic Scenario Generator (ESG)** based on correlated geometric Brownian motion. If you're interested in modeling asset prices, simulating market risk, or validating your understanding of Monte Carlo methods—this is for you.

---

## ?? What This Project Does

* Simulates realistic multi-asset price paths using **correlated log returns**
* Calibrates to historical data using estimated drift, volatility, and correlation
* Validates that simulations reproduce the empirical statistics accurately
* Fully vectorized and tested on millions of paths for accuracy

---

## ?? How It Works

### Core functions (in `src/`):

* `simulate_correlated_logreturns()` — simulates correlated log returns using Cholesky decomposition
* `build_prices()` — builds asset price paths from simulated log returns
* `get_ticker_data()` *(work in progress)* — fetches Yahoo Finance data and computes inputs

### Validation (in `test/`):

* Uses AAPL and COST historical returns
* Computes theoretical vs simulated drift, volatility, and correlation
* Reports % error in basis points (bps)
* Confirms the ESG engine is working as expected

---

## ?? Project Structure

```
ESG/
??? src/                        # Simulation engine
?   ??? simulate.py             # Main logic
??? test/
?   ??? _run_correlated_simulation.py  # Validation against real data
??? data/
?   ??? prices-split-adjusted.csv      # Input data (from Kaggle)
??? README.md                  # You're here
```

---

## ?? Running the Simulation Test

```bash
python test/_run_correlated_simulation.py
```

This script will:

* Load historical data
* Compute inputs (drift, vol, corr)
* Simulate log returns for multiple assets
* Compare simulated stats to actual historical ones

You’ll see a comparison printed to the console, with percentage differences in basis points.

---

## ? Sample Output

```
theoretical drift per step:
 [2.4e-04 1.7e-04]
simulated drift per step:
 [2.39e-04 1.71e-04]
% difference (mean):
 [0.12 0.58]

simulated ? per step:
 [0.0103 0.0075]
% difference (?):
 [-0.08 0.11]

simulated corr matrix:
 [[1.0, 0.29], [0.29, 1.0]]
```

These results show that the simulation accurately replicates the underlying statistical structure of the historical data.

---

## ?? To-Do

Here are some ideas for what’s next:

* [ ] Add stochastic interest rates (e.g. CIR model)
* [ ] Incorporate jumps (Merton or Kou)
* [ ] Support multi-frequency time steps (e.g. weekly/monthly)
* [ ] Allow regime-switching correlation structures
* [ ] Extend `get_ticker_data()` to handle more tickers and auto-cleaning
* [ ] Wrap simulations into a small web app for live demo or API

---

## ?? Author

Developed by **Ryan Olivier** — actuary, investment thinker, and builder of quant tools. For questions or suggestions, feel free to open an issue.

---

Thanks for checking it out! If you're working with markets, risk, or simulation—this project should feel like home.