# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 14:47:33 2025

@author: RyanPC
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import openpyxl as xl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def simulate_logreturns(mu, sigma, dt, dWt, time_steps, n_sims):
    """
    dWt should be shape (time_steps, n_sims)
    Returns log-returns array of same shape.
    """
    
    log_returns = np.zeros((time_steps, n_sims))
    drift = (mu - 0.5 * sigma**2) * dt
    log_returns = drift + sigma * dWt
    
    return log_returns

def correlated_dWt(n_asset, time_steps, n_sims, corr_mat, dt):
    """
    Generate correlated Wiener increments dW:
      - corr_mat is the n_asset × n_asset correlation matrix
      - dt is the time‐step size
    Returns array of shape (n_asset, time_steps, n_sims)
    """
    # cholesky decomposition
    L = np.linalg.cholesky(corr_mat)
    
    # generate uncorrelated normals and scaledby sqrt(dt)
    uncorrelated = np.random.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=(n_asset, time_steps, n_sims)
    )
    # correlate the uncorrelated normals (above): for each sim & time_step, mix across assets
    # result[i,t,s] = sum_j L[i,j] * uncorrelated[j,t,s]
    corr_dWt = np.einsum('ij,jts->its', L, uncorrelated)
    return corr_dWt

def simulate_correlated_logreturns(mu, sigma, dt, time_steps, n_sims, corr_mat):
    """
    Returns log_returns with shape (n_asset, time_steps, n_sims)
    """
    n_asset = len(mu)
    dWt_corr = correlated_dWt(n_asset, time_steps, n_sims, corr_mat, dt)
    log_returns = np.zeros((n_asset, time_steps, n_sims))

    for i in range(n_asset):
        # simulate per‐asset slice
        log_returns[i, :, :] = simulate_logreturns(
            mu[i], sigma[i], dt,
            dWt_corr[i, :, :],
            time_steps, n_sims
        )
    return log_returns

def build_prices(S0, log_returns):
    """
    From log_returns shape (n_asset, time_steps, n_sims),
    build price paths. Returns array of same shape.
    """
    n_asset, time_steps, n_sims = log_returns.shape
    prices = np.zeros_like(log_returns)
    # set initial prices (broadcast S0 over sims)
    prices[:, 0, :] = S0[:, None]
    # iterate forward
    for t in range(1, time_steps):
        prices[:, t, :] = prices[:, t - 1, :] * np.exp(log_returns[:, t - 1, :])
    return prices

def get_ticker_data(tickers, start_date, end_date, frequency):
    """
    Download Close prices, compute mu, sigma (sample std ddof=1) 
    and correlation matrix.
    Currently not working.
    """
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=frequency,
        group_by='ticker'
    )['Close']

    # compute log-returns (drop the first NaN)
    log_ret = np.log(df / df.shift(1)).dropna(how = 'all')

    # per-ticker drift and vol (use unbiased std, ddof=1)
    mu    = log_ret.mean(axis = 0)
    sigma = log_ret.std(axis = 0, ddof = 1)

    # corrcoef wants shape (n_obs, n_assets)
    corr_mat = log_ret.corr().values

    parameters = pd.DataFrame({
        "Mu": mu,
        "Sigma": sigma
    })

    return parameters, corr_mat
