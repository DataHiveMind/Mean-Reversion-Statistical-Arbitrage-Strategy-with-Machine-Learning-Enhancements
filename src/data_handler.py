import os
import pandas as pd
import numpy as np
import yfinance as yf

class DataHandler:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_data(self, tickers, start, end, interval="1d"):
        """
        Download historical price data for given tickers.
        """
        data = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=True)
        return data

    def clean_data(self, df):
        """
        Handle missing values and outliers.
        """
        # Forward fill, then backward fill for missing values
        df = df.ffill().bfill()
        # Remove extreme outliers (e.g., 5 standard deviations from mean)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = np.where(
                np.abs(df[col] - mean) > 5 * std,
                np.nan,
                df[col]
            )
        df = df.ffill().bfill()
        return df

    def align_data(self, dfs):
        """
        Align multiple DataFrames by their index (date/timestamp).
        """
        return pd.concat(dfs, axis=1, join='inner')

    def resample_data(self, df, rule="1D", agg_dict=None):
        """
        Resample high-frequency data to lower frequency.
        """
        if agg_dict is None:
            agg_dict = {col: 'last' for col in df.columns}
        return df.resample(rule).agg(agg_dict)

    def save_data(self, df, filename):
        """
        Save DataFrame to Parquet file.
        """
        filepath = os.path.join(self.data_dir, filename)
        df.to_parquet(filepath)

    def load_data(self, filename):
        """
        Load DataFrame from Parquet file.
        """
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_parquet(filepath)