import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timezone

class DataHandler:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_data(self, tickers, start, end, interval="1d", adjust_corporate_actions=True):
        """
        Download historical price data for given tickers.
        Simulates real-time feed by yielding rows if needed.
        """
        data = yf.download(
            tickers, start=start, end=end, interval=interval,
            group_by='ticker', auto_adjust=adjust_corporate_actions, progress=False
        )
        # Ensure UTC timestamps
        data.index = pd.to_datetime(data.index).tz_localize('UTC')
        return data

    def ingest_realtime(self, data_stream):
        """
        Simulate real-time ingestion from a data stream (generator or API).
        """
        for tick in data_stream:
            # Process each tick (dict or pd.Series)
            yield self._process_tick(tick)

    def _process_tick(self, tick):
        """
        Process a single tick: handle timestamp, deduplication, and mid-price.
        """
        # Example: tick = {'timestamp': ..., 'bid': ..., 'ask': ...}
        tick['timestamp'] = pd.to_datetime(tick['timestamp']).tz_convert('UTC')
        tick['mid'] = (tick['bid'] + tick['ask']) / 2
        return tick

    def clean_data(self, df, method="ffill_bfill", outlier_method="zscore", z_thresh=5):
        """
        Handle missing values and outliers.
        Advanced: Use Kalman filter or robust statistics for imputation.
        """
        # Forward/backward fill
        if method == "ffill_bfill":
            df = df.ffill().bfill()
        # Outlier detection
        if outlier_method == "zscore":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                df.loc[np.abs(z) > z_thresh, col] = np.nan
            df = df.ffill().bfill()
        return df

    def adjust_for_corporate_actions(self, df, actions_df):
        """
        Adjust prices for splits, dividends, etc.
        actions_df: DataFrame with corporate actions (splits, dividends)
        """
        # Placeholder: yfinance auto_adjust handles this, but for custom data, implement here.
        return df

    def align_data(self, dfs, tz="UTC"):
        """
        Align multiple DataFrames by their index (date/timestamp), ensure UTC.
        """
        aligned = [df.tz_convert(tz) if df.index.tz else df.tz_localize(tz) for df in dfs]
        return pd.concat(aligned, axis=1, join='inner')

    def resample_data(self, df, rule="1D", agg_dict=None):
        """
        Resample high-frequency data to lower frequency.
        """
        if agg_dict is None:
            agg_dict = {col: 'last' for col in df.columns}
        return df.resample(rule).agg(agg_dict)

    def save_data(self, df, filename, format="parquet"):
        """
        Save DataFrame to Parquet (default) or HDF5 for efficient storage.
        """
        filepath = os.path.join(self.data_dir, filename)
        if format == "parquet":
            df.to_parquet(filepath)
        elif format == "hdf":
            df.to_hdf(filepath, key='data', mode='w')
        else:
            raise ValueError("Unsupported format")

    def load_data(self, filename, format="parquet"):
        """
        Load DataFrame from Parquet or HDF5.
        """
        filepath = os.path.join(self.data_dir, filename)
        if format == "parquet":
            return pd.read_parquet(filepath)
        elif format == "hdf":
            return pd.read_hdf(filepath, key='data')
        else:
            raise ValueError("Unsupported format")

    # --- Advanced/Scalable Storage Awareness ---
    # For ultra-high-frequency or large-scale, consider:
    # - Integration with InfluxDB, kdb+, or ArcticDB for TSDB storage.
    # - Parallel loading/processing with joblib or multiprocessing.

    # --- Calendar & Timezone Management ---
    def get_trading_calendar(self, exchange="NYSE"):
        """
        Return a trading calendar (holidays, early closes).
        """
        try:
            import exchange_calendars as ecals
            cal = ecals.get_calendar(exchange)
            return cal
        except ImportError:
            return None  # Or fallback to pandas_market_calendars

    def localize_to_exchange(self, df, exchange="NYSE"):
        """
        Convert timestamps to exchange local time.
        """
        cal = self.get_trading_calendar(exchange)
        if cal:
            tz = cal.tz
            df.index = df.index.tz_convert(tz)
        return df