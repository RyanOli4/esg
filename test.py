#%% 
# -*- coding: utf-8 -*-
"""
Validation of Correlated Log-Return Simulation

This script tests that the `simulate_correlated_logreturns` function correctly reproduces the empirical
first- and second-moment statistics (drift, volatility, correlation) observed in historical equity log-returns,
on a per-time-step basis.

Test 1: Historical data for AAPL and COST
-----------------------------------------
1. Load split-adjusted daily close prices (from: https://www.kaggle.com/datasets/dgawlik/nyse)
2. Compute daily log-returns, estimating mean (`mu`), standard deviation (`sigma`), and correlation matrix.
3. Define per-step theoretical drift and volatility:
   - drift_per_step      = (mu - 0.5 * sigma^2) * dt
   - volatility_per_step = sigma * sqrt(dt)
4. Run Monte Carlo simulation with `n_sims` paths and `time_steps` steps.
5. Extract empirical per-step mean, volatility, and correlation from simulated data.
6. Compare theoretical vs. simulated metrics, reporting percent deviations and matrix comparison.

Interpretation:
- Percent deviations for drift and volatility should be within a few basis points.
- Simulated correlation matrix should converge to the input correlation as sample size increases.
"""

import pandas as pd
import numpy as np
import src
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt


#%% 1. Data Preparation
raw_data = pd.read_csv(
    filepath_or_buffer= 'data/prices-split-adjusted.csv'
)

data = raw_data.pivot(index="date", columns="symbol", values="close")
data = data[['AAPL', 'MCD', 'GOOG']]

#%% 2. Historical Log-Return Estimation
log_return_data = np.delete(
    np.log(data / data.shift(1)),
    0,
    axis=0
)

#%% 3. Parameter Calculation
n_assets   = len(data.columns)
dt         = 1/252          
time_steps = 252*5
n_sims     = 10_000

E_X    = np.mean(log_return_data, axis=0)
SD_X = np.std(log_return_data, axis=0, ddof=1)

sigma = SD_X / np.sqrt(dt)
mu = E_X/dt + 0.5*sigma**2

# Empirical correlation matrix of daily returns
corr_mat = np.corrcoef(log_return_data[1:, :], rowvar=False)

#%% 4. Monte Carlo Simulation
model_1 = src.GeometricBrownianMotion(mu = mu, sigma = sigma, dt = 1/252, corr_mat =  corr_mat)

sim = model_1.simulate_correlated_logreturns(time_steps, n_sims)

#%% 5. Empirical Moment Extraction
# Flatten for correlation comparison
sim_flat = sim.reshape(n_assets, time_steps * n_sims)
sim_corr = np.corrcoef(sim_flat)

# Compute per-asset mean & standard deviation
sim_mean = sim.mean(axis=(1, 2))
sim_std  = np.array([np.std(sim[i], ddof=1) for i in range(n_assets)])

src.plot_returns(sim, n_plot = 10)

#%% 6. Validation Metrics and Output
print("Empirical mean of historical log-returns (E_X):\n", E_X)
print("Simulated mean of log-returns:\n", sim_mean)
print("% difference (mean):\n", (sim_mean / E_X - 1) * 100)

print("\nEmpirical std dev of historical log-returns (SD_X):\n", SD_X)
print("Simulated std dev of log-returns:\n", sim_std)
print("% difference (std dev):\n", (sim_std / SD_X - 1) * 100)

print("\nEmpirical correlation matrix:\n", corr_mat)
print("Simulated correlation matrix:\n", sim_corr)

asset_labels = list(data.columns)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Empirical correlation heatmap
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", xticklabels=asset_labels,
            yticklabels=asset_labels, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title("Empirical Correlation")

# Simulated correlation heatmap
sns.heatmap(sim_corr, annot=True, cmap="coolwarm", xticklabels=asset_labels,
            yticklabels=asset_labels, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("Simulated Correlation")

plt.tight_layout()
plt.show()