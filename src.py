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

class GeometricBrownianMotion:
    
    def __init__(self, mu = None, sigma = None, dt = None, corr_mat = None):
        self.mu = mu
        self.sigma = sigma
        self.corr_mat = corr_mat
        self.dt = dt
        
    def __str__(self):
        return (f"Geometric Brownian Motion:\n"
                f"mu: {self.mu}\n"
                f"sigma: {self.sigma}\n"
                f"dt: {self.dt}\n"
                "corr_mat:\n"
                f"{self.corr_mat}")
    
    def __repr__(self):
        return (f"Geometric Brownian Motion:\n"
                f"mu: {self.mu}\n"
                f"sigma: {self.sigma}\n"
                f"dt: {self.dt}\n"
                "corr_mat:\n"
                f"{self.corr_mat}")
    
    def simulate_logreturns(self, time_steps, n_sims):
        """
        Returns are in shape (time_steps, n_sims) using iid N(0,1) as dWt
        Returns log-returns array of same shape.
        """
        if len(self.mu) > 1:
             raise ValueError("Multidimensional parameters given - solve issue by using: simulate_correlated_logreturns()")
        
        dWt = np.random.normal(loc = 0.0, scale = np.sqrt(self.dt), size = (time_steps, n_sims))
        log_returns = np.zeros((time_steps, n_sims))
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        log_returns = drift + self.sigma * dWt
        
        return log_returns
    
    def simulate_correlated_logreturns(self, time_steps, n_sims):
        """
        Returns are in shape (n_asset, time_steps, n_sims) using iid correlated N(0,1) as dWt (based on correlation matrix provided/calibrated)
        Returns log-returns array of same shape.
        """
        if len(self.mu) == 1:
            raise ValueError("Univariate parameters given - solve issue by using: simulate_logreturns()")
        
        n_asset = len(self.mu)
        
        mu = self.mu
        sigma = self.sigma
        dWt_corr = self._correlated_dWt(time_steps, n_sims)
        
        log_returns = np.zeros((n_asset, time_steps, n_sims))
        
        for i in range(n_asset):
            # simulate per‐asset slice
            log_returns[i, :, :] = self._multivariate_simulate_logreturns(
                mu = mu[i], 
                sigma = sigma[i], 
                dWt = dWt_corr[i, :, :],
                time_steps = time_steps,
                n_sims = n_sims
            )
        return log_returns
    
    def build_prices(self, S0, log_returns):
        """
        Does not depend on object - can really use seperately. To be integrated later.
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
    
    def calibrate(self, tickers, start_date, end_date, frequency):
        """
        Download Close prices, compute mu, sigma (sample std ddof=1) 
        and correlation matrix.
        Currently not working as desired.
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
    
        self.mu = mu
        self.sigma = sigma
        self.corr_mat = corr_mat
        self.start_date = start_date
        self.end_date = end_date
        
        if frequency == "1d":
            dt = 1/252
        elif frequency == "1mo":
            dt = 1/12
        elif frequency == "1y":
            dt = 1
        else:
            raise ValueError(f"Invalid frequency of {frequency} - please use 1d, 1mo, or 1y")
        self.dt = dt
        
    def _multivariate_simulate_logreturns(self, mu, sigma, dWt, time_steps, n_sims):
        """
        Returns are in shape (time_steps, n_sims) using iid N(0,1) as dWt
        Returns log-returns array of same shape.
        """
        
        log_returns = np.zeros((time_steps, n_sims))
        drift = (mu - 0.5 * sigma**2) * self.dt
        log_returns = drift + sigma * dWt
        
        return log_returns

    def _correlated_dWt(self, time_steps, n_sims):
        """
        Generate correlated Wiener increments dW:
          - corr_mat is the n_asset × n_asset correlation matrix
          - dt is the time‐step size
        Returns array of shape (n_asset, time_steps, n_sims)
        """

        n_asset = len(self.corr_mat)
        
        # cholesky decomposition
        L = np.linalg.cholesky(self.corr_mat)
        
        # generate uncorrelated normals and scaledby sqrt(dt)
        uncorrelated = np.random.normal(loc=0.0, scale=np.sqrt(self.dt), size=(n_asset, time_steps, n_sims))
        
        
       
        # correlate the uncorrelated normals (above): for each sim & time_step, mix across assets
        # result[i,t,s] = sum_j L[i,j] * uncorrelated[j,t,s]
        corr_dWt = np.einsum('ij,jts->its', L, uncorrelated)
        return corr_dWt
    


