# ESG: Economic Scenario Generator

Welcome! This project is my own lightweight attempt at an **Economic Scenario Generator (ESG)** based on correlated geometric Brownian motion.

![picture of sample paths](images/pic.png)

---

## Project Scope

### Current
* Simulates realistic multi-asset price paths using **correlated log returns**
* Calibrates to historical data using estimated drift, volatility, and correlation
* Validates that simulations reproduce the empirical statistics accurately
* Fully vectorized and tested on millions of paths for accuracy

### Future
* Add features that allow for time-varying volatility, such as GARCH-based models.
* Add other asset classes to the selection.

---

## How It Works

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

## Project Structure

```
ESG/
??? src/                               # Simulation engine
?   ??? simulate.py                    # Main logic
??? test/
?   ??? _run_correlated_simulation.py  # Validation against real data
??? data/
?   ??? prices-split-adjusted.csv      # Input data (from Kaggle)
??? README.md                          # You're here
```

---

## Running the Simulation Test

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

## Sample Output

```
theoretical drift per step:
 [2.45931239e-06 1.99025920e-06]
simulated drift per step:
 [2.33472987e-06 1.97560408e-06]
% difference (mean):
 [-5.06574612 -0.73634268]

theoretical ? per step:
 [0.00104121 0.00069824]
simulated ? per step:
 [0.00104171 0.00069822]
% difference (?):
 [ 0.04792867 -0.0025378 ]

theoretical corr matrix:
 [[1.         0.29400284]
 [0.29400284 1.        ]]
simulated corr matrix:
 [[1.        0.2940269]
 [0.2940269 1.       ]]
```

These results show that the simulation accurately replicates the underlying statistical structure of the historical data.

---

## To-Do

Here are some ideas for what’s next:

* [ ] Rewrite in OOP format
* [ ] Add support for dividends (currently only modelling price return)
* [ ] Add stochastic interest rates (e.g. CIR model)
* [ ] Incorporate jumps (Merton or Kou)
* [ ] Support multi-frequency time steps (e.g. weekly/monthly)
* [ ] Allow regime-switching correlation structures
* [ ] Extend `get_ticker_data()` to handle more tickers and auto-cleaning
* [ ] Wrap simulations into a small web app for live demo or API

---

## Author

Developed by **Ryan Olivier**. For questions or suggestions, feel free to open an issue.

---