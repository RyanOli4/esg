#%%
# Validation of Correlated Log-Return Simulation

"""
This script validates that `simulate_correlated_logreturns` correctly reproduces 
empirical drift, volatility, and correlation of historical equity returns.

Assets: AAPL, MCD, GOOG (daily data)
"""

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import src 

def test_1():
#%% 1. Load and Prepare Data
    raw_data = pd.read_csv('data/prices-split-adjusted.csv')
    data = raw_data.pivot(index="date", columns="symbol", values="close")[['AAPL', 'MCD', 'GOOG']]
    
#%% 2. Compute Log-Returns
    log_returns = np.log(data / data.shift(1)).dropna()
    
#%% 3. Parameter Estimation
    n_assets = len(data.columns)
    dt = 1 / 252
    n_sims = 10_000
    time_steps = 252 * 5
    
    E_X = log_returns.mean().values             # empirical daily mean
    SD_X = log_returns.std(ddof=1).values       # empirical daily std
    
    sigma = SD_X / np.sqrt(dt)                  # annualised volatility
    mu = E_X / dt + 0.5 * sigma**2              # drift to match mean
    corr_mat = log_returns.corr().values        # empirical correlation
    
#%% 4. Simulate Correlated Log-Returns
    model = src.GeometricBrownianMotion(mu=mu, sigma=sigma, dt=dt, corr_mat=corr_mat)
    sim = model.simulate_correlated_logreturns(time_steps, n_sims)  # shape: (assets, time_steps, sims)
    
#%% 5. Empirical Statistics from Simulation
    sim_flat = sim.reshape(n_assets, -1)
    sim_corr = np.corrcoef(sim_flat)
    sim_mean = sim.mean(axis=(1, 2))
    sim_std = sim.std(axis=(1, 2), ddof=1)
    
#%% 6. Visualise Results 
    model.plot_example()
    
#%% 7. Output Validation Metrics
    def percent_diff(sim, actual):
        return np.round((sim / actual - 1) * 100, 3)
    
    print("\n--- Drift Comparison ---")
    print("Empirical E[X]:", E_X)
    print("Simulated E[X]:", sim_mean)
    print("% Difference:", percent_diff(sim_mean, E_X))
    
    print("\n--- Volatility Comparison ---")
    print("Empirical SD[X]:", SD_X)
    print("Simulated SD[X]:", sim_std)
    print("% Difference:", percent_diff(sim_std, SD_X))
    
    print("\n--- Correlation Matrices ---")
    print("Empirical Correlation Matrix:\n", corr_mat)
    print("Simulated Correlation Matrix:\n", sim_corr)

if __name__ == "__main__":
    test_1()
    ...
    
    
    
    
    
    
    
    