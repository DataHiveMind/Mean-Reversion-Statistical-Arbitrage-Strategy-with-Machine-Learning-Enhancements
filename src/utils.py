import logging
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_file="app.log", level=logging.INFO):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_config(config_path="config.json"):
    """
    Load configuration parameters from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def plot_time_series(df, columns=None, title="Time Series", figsize=(12, 6)):
    """
    Plot time series for specified columns.
    """
    plt.figure(figsize=figsize)
    if columns is None:
        columns = df.columns
    df[columns].plot(ax=plt.gca())
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_equity_curve(equity_curve, title="Equity Curve", figsize=(12, 6)):
    """
    Plot equity curve.
    """
    plt.figure(figsize=figsize)
    plt.plot(equity_curve)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

def plot_scatter(x, y, xlabel="X", ylabel="Y", title="Scatter Plot", figsize=(8, 6)):
    """
    Plot scatter plot of x vs y.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def safe_divide(a, b, default=0.0):
    """
    Safely divide two numbers, returning default if denominator is zero.
    """
    try:
        return a / b if b != 0 else default
    except Exception:
        return default