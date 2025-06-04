import numpy as np
import pandas as pd

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

    def get_features(self):
        """
        Return the DataFrame with all engineered features.
        """
        return self.df