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

#%% 1. Data Preparation
raw_data = pd.read_csv(
    filepath_or_buffer= 'data/prices-split-adjusted.csv'
)

data = raw_data.pivot(index="date", columns="symbol", values="close")
data = data[['AAPL', 'COST']]

#%% 2. Historical Log-Return Estimation
log_return_data = np.delete(
    np.log(data / data.shift(1)),
    0,
    axis=0
)

#%% 3. Parameter Calculation
n_assets   = len(data.columns)
dt         = 1/252          
time_steps = 120     
n_sims     = 1_000

mu    = np.mean(log_return_data, axis=0)
sigma = np.std(log_return_data, axis=0, ddof=1)

# Theoretical per-step moments
drift_per_step      = (mu - 0.5 * sigma**2) * dt
volatility_per_step = sigma * np.sqrt(dt)

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

#%% 6. Validation Metrics and Output
print("theoretical drift per step:\n", drift_per_step)
print("simulated drift per step:\n", sim_mean)
print("% difference (drift):\n", (sim_mean / drift_per_step - 1) * 100)

print("\ntheoretical σ per step:\n", volatility_per_step)
print("simulated σ per step:\n", sim_std)
print("% difference (σ):\n", (sim_std / volatility_per_step - 1) * 100)

print("\ntheoretical corr matrix:\n", corr_mat)
print("simulated corr matrix:\n", sim_corr)

