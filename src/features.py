import numpy as np
import pandas as pd
from scipy.stats import zscore

class FeatureEngineer:
    def __init__(self, df):
        """
        df: pd.DataFrame with columns like ['spread', 'volume', ...]
        """
        self.df = df.copy()

    def add_lagged_features(self, column, lags=[1, 2, 3]):
        """
        Create lagged features for a given column.
        """
        for lag in lags:
            self.df[f"{column}_lag{lag}"] = self.df[column].shift(lag)
        return self

    def add_rolling_volatility(self, column, windows=[10, 20]):
        """
        Add rolling standard deviation (volatility) features.
        """
        for window in windows:
            self.df[f"{column}_vol_{window}"] = self.df[column].rolling(window).std()
        return self

    def add_zscore(self, column, window=20):
        """
        Add rolling z-score feature.
        """
        roll_mean = self.df[column].rolling(window).mean()
        roll_std = self.df[column].rolling(window).std()
        self.df[f"{column}_zscore"] = (self.df[column] - roll_mean) / roll_std
        return self

    def add_rsi(self, column, window=14):
        """
        Add Relative Strength Index (RSI) feature.
        """
        delta = self.df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        self.df[f"{column}_rsi"] = 100 - (100 / (1 + rs))
        return self

    def add_macd(self, column, span_short=12, span_long=26, span_signal=9):
        """
        Add MACD and MACD signal line features.
        """
        ema_short = self.df[column].ewm(span=span_short, adjust=False).mean()
        ema_long = self.df[column].ewm(span=span_long, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=span_signal, adjust=False).mean()
        self.df[f"{column}_macd"] = macd
        self.df[f"{column}_macd_signal"] = signal
        return self

    def add_volume_features(self, volume_col, windows=[10, 20]):
        """
        Add rolling mean and std of volume.
        """
        for window in windows:
            self.df[f"{volume_col}_mean_{window}"] = self.df[volume_col].rolling(window).mean()
            self.df[f"{volume_col}_std_{window}"] = self.df[volume_col].rolling(window).std()
        return self

    def add_cross_sectional_rank(self, column):
        """
        Add cross-sectional rank feature for the column at each timestamp.
        """
        self.df[f"{column}_cs_rank"] = self.df[column].rank(axis=1, pct=True)
        return self

    # --- High-Frequency & Microstructure Features ---
    def add_volume_imbalance(self, buy_col, sell_col, window=10):
        """
        Order flow imbalance: (Buy Volume - Sell Volume) / (Buy + Sell)
        """
        imbalance = (self.df[buy_col] - self.df[sell_col]) / (self.df[buy_col] + self.df[sell_col] + 1e-8)
        self.df[f"volume_imbalance_{window}"] = imbalance.rolling(window).mean()
        return self

    def add_bid_ask_spread(self, bid_col, ask_col):
        """
        Add bid-ask spread feature.
        """
        self.df["bid_ask_spread"] = self.df[ask_col] - self.df[bid_col]
        return self

    def add_amihud_illiquidity(self, price_col, volume_col, window=20):
        """
        Amihud's illiquidity: |Return| / Volume
        """
        returns = self.df[price_col].pct_change().abs()
        illiq = returns / (self.df[volume_col] + 1e-8)
        self.df[f"amihud_illiq_{window}"] = illiq.rolling(window).mean()
        return self

    def add_realized_volatility(self, price_col, window=20):
        """
        Realized volatility: sqrt(sum of squared returns)
        """
        returns = self.df[price_col].pct_change()
        realized_vol = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)
        self.df[f"realized_vol_{window}"] = realized_vol
        return self

    def add_parkinson_volatility(self, high_col, low_col, window=20):
        """
        Parkinson volatility estimator.
        """
        parkinson = (1.0 / (4 * np.log(2))) * ((np.log(self.df[high_col] / self.df[low_col])) ** 2)
        self.df[f"parkinson_vol_{window}"] = parkinson.rolling(window).mean()
        return self

    # --- Statistical Arbitrage Specific Features ---
    def add_spread_autocorrelation(self, spread_col, lags=[1, 2, 3]):
        """
        Add autocorrelation features for the spread.
        """
        for lag in lags:
            self.df[f"{spread_col}_autocorr_lag{lag}"] = self.df[spread_col].rolling(lag + 1).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False)
        return self

    def add_half_life(self, spread_col, window=100):
        """
        Calculate rolling half-life of mean reversion.
        """
        def calc_half_life(series):
            lagged = series.shift(1).dropna()
            delta = series.diff().dropna()
            if len(lagged) < 2:
                return np.nan
            beta = np.polyfit(lagged, delta, 1)[0]
            if beta == 0:
                return np.nan
            half_life = -np.log(2) / beta
            return half_life if half_life > 0 else np.nan
        self.df[f"{spread_col}_half_life"] = self.df[spread_col].rolling(window).apply(calc_half_life, raw=False)
        return self

    def add_hurst_exponent(self, col, window=100):
        """
        Rolling Hurst exponent.
        """
        def hurst(ts):
            lags = range(2, 20)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]*2.0
        self.df[f"{col}_hurst"] = self.df[col].rolling(window).apply(lambda x: hurst(x) if len(x) == window else np.nan, raw=False)
        return self

    def add_cointegration_residual(self, y_col, x_col, window=100):
        """
        Rolling OLS residuals as cointegration error.
        """
        def ols_resid(y, x):
            if len(y) < 2:
                return np.nan
            beta = np.polyfit(x, y, 1)[0]
            return y[-1] - beta * x[-1]
        self.df[f"{y_col}_{x_col}_coint_resid"] = self.df[[y_col, x_col]].rolling(window).apply(
            lambda x: ols_resid(x[:,0], x[:,1]), raw=False)
        return self

    # --- Market Regime Features ---
    def add_volatility_regime(self, price_col, n_states=2):
        """
        Use GaussianHMM to classify volatility regimes.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            returns = self.df[price_col].pct_change().dropna().values.reshape(-1, 1)
            model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
            model.fit(returns)
            hidden_states = model.predict(returns)
            self.df.loc[self.df.index[-len(hidden_states):], "vol_regime"] = hidden_states
        except ImportError:
            self.df["vol_regime"] = np.nan
        return self

    # --- Cross-Sectional Features ---
    def add_relative_strength(self, spread_col, group_df):
        """
        Spread z-score relative to other pairs at each timestamp.
        """
        self.df[f"{spread_col}_rel_strength"] = self.df[spread_col] - group_df[spread_col].mean(axis=1)
        return self

    def add_liquidity_ratio(self, volume_col, market_volume_col):
        """
        Pair's volume relative to market volume.
        """
        self.df[f"{volume_col}_liq_ratio"] = self.df[volume_col] / (self.df[market_volume_col] + 1e-8)
        return self

    # --- Feature Engineering Pipeline Example ---
    def pipeline(self, steps):
        """
        Apply a sequence of feature engineering steps.
        steps: list of tuples (method_name, kwargs)
        """
        for method, kwargs in steps:
            getattr(self, method)(**kwargs)
        return self

    def get_features(self):
        """
        Return the DataFrame with all engineered features.
        """
        return self.df