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

    def screen_universe(self, min_volume=1e6, min_volatility=0.01, sector_map=None):
        """
        Screen assets by average volume, volatility, and optionally sector.
        sector_map: dict {ticker: sector}
        Returns: list of tickers passing filters
        """
        avg_vol = self.price_data.pct_change().std()
        avg_volume = self.price_data.mean()
        tickers = [t for t in self.price_data.columns
                   if avg_vol[t] > min_volatility and avg_volume[t] > min_volume]
        if sector_map:
            # Example: filter by sector
            tickers = [t for t in tickers if sector_map.get(t, None) == "Technology"]
        return tickers

    def factor_neutral_spread(self, y, x, factors_df):
        """
        Regress y and x on factors, use residuals for cointegration.
        """
        from statsmodels.api import OLS, add_constant
        y_resid = OLS(y, add_constant(factors_df)).fit().resid
        x_resid = OLS(x, add_constant(factors_df)).fit().resid
        return y_resid, x_resid

    def adf_test(self, series):
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series)
        return result[1]  # p-value

    def pp_test(self, series):
        from statsmodels.tsa.stattools import PhillipsPerron
        result = PhillipsPerron(series).stat
        return result

    def kpss_test(self, series):
        from statsmodels.tsa.stattools import kpss
        result = kpss(series, regression='c')
        return result[1]  # p-value

    def johansen_test(self, df, det_order=0, k_ar_diff=1):
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        result = coint_johansen(df, det_order, k_ar_diff)
        return result.lr1, result.cvt  # eigenvalues, critical values

    def zivot_andrews_test(self, series):
        from statsmodels.tsa.stattools import zivot_andrews
        result = zivot_andrews(series)
        return result[1]  # p-value
    
    def rls_hedge_ratio(self, y, x, lambda_=0.99):
        """
        Recursive Least Squares for dynamic hedge ratio.
        """
        n = len(y)
        beta = np.zeros(n)
        P = 1e5
        theta = 0
        for t in range(n):
            x_t = x.iloc[t]
            y_t = y.iloc[t]
            K = P * x_t / (lambda_ + x_t * P * x_t)
            theta = theta + K * (y_t - x_t * theta)
            P = (P - K * x_t * P) / lambda_
            beta[t] = theta
        return pd.Series(beta, index=y.index)

    def cluster_assets(self, n_clusters=10, feature_df=None):
        """
        Cluster assets using KMeans on feature_df (e.g., vol, corr, etc.)
        """
        from sklearn.cluster import KMeans
        if feature_df is None:
            feature_df = self.price_data.pct_change().std().to_frame('vol')
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(feature_df)
        return dict(zip(feature_df.index, labels))

    def factor_exposure(self, tickers, factor_loadings):
        """
        Return factor exposures for given tickers.
        """
        return factor_loadings.loc[tickers]