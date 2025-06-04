import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

class PairSelector:
    def __init__(self, price_data):
        """
        price_data: pd.DataFrame, columns are tickers, index is datetime
        """
        self.price_data = price_data

    def rolling_correlation(self, window=60):
        """
        Calculate rolling correlation matrix for all pairs.
        Returns: dict of {(ticker1, ticker2): pd.Series}
        """
        correlations = {}
        tickers = self.price_data.columns
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                corr = self.price_data[t1].rolling(window).corr(self.price_data[t2])
                correlations[(t1, t2)] = corr
        return correlations

    def find_cointegrated_pairs(self, significance=0.05):
        """
        Test all pairs for cointegration using Engle-Granger test.
        Returns: list of tuples (ticker1, ticker2, p-value)
        """
        tickers = self.price_data.columns
        coint_pairs = []
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                score, pvalue, _ = coint(self.price_data[t1], self.price_data[t2])
                if pvalue < significance:
                    coint_pairs.append((t1, t2, pvalue))
        return coint_pairs

    def calculate_hedge_ratio(self, y, x):
        """
        Calculate hedge ratio using OLS regression (y ~ x).
        Returns: hedge ratio (float)
        """
        x = add_constant(x)
        model = OLS(y, x).fit()
        return model.params[1]

    def kalman_filter_hedge_ratio(self, y, x):
        """
        Dynamically estimate hedge ratio using a simple Kalman Filter.
        Returns: pd.Series of hedge ratios.
        """
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)  # State transition covariance
        obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]

        state_mean = np.zeros((2, 1))
        state_cov = np.eye(2)
        hedge_ratios = []

        for t in range(len(y)):
            if t == 0:
                pred_state_mean = state_mean
                pred_state_cov = state_cov + trans_cov
            else:
                pred_state_mean = state_mean
                pred_state_cov = state_cov + trans_cov

            obs = y.iloc[t]
            obs_matrix = obs_mat[t]
            pred_obs = np.dot(obs_matrix.T, pred_state_mean)[0, 0]
            innovation = obs - pred_obs
            innovation_cov = np.dot(np.dot(obs_matrix.T, pred_state_cov), obs_matrix) + 1.0

            kalman_gain = np.dot(pred_state_cov, obs_matrix) / innovation_cov
            state_mean = pred_state_mean + kalman_gain * innovation
            state_cov = pred_state_cov - np.dot(kalman_gain, obs_matrix.T) * pred_state_cov

            hedge_ratios.append(state_mean[0, 0])

        return pd.Series(hedge_ratios, index=y.index)