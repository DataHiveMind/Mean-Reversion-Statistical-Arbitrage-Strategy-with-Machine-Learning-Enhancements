import logging
import json
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import time

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# --- Robust Logging ---
def setup_logging(log_file="app.log", level=logging.INFO, console=True):
    """
    Set up structured logging configuration.
    """
    handlers = [logging.FileHandler(log_file, mode='a')]
    if console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )

# --- Configuration Management ---
def load_config(config_path="config.json", schema=None):
    """
    Load configuration parameters from a JSON file and validate with Pydantic if schema is provided.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    if schema and PYDANTIC_AVAILABLE:
        try:
            config = schema(**config)
        except ValidationError as e:
            raise ValueError(f"Config validation error: {e}")
    return config

# --- Performance Profiling ---
def profile(func):
    """
    Decorator for simple function timing.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"Function {func.__name__} executed in {elapsed:.4f} seconds.")
        return result
    return wrapper

# --- Error Handling & Retries ---
def retry(ExceptionToCheck, tries=3, delay=2, backoff=2):
    """
    Retry decorator with exponential backoff.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    logging.warning(f"{e}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

# --- Data Serialization/Deserialization ---
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_joblib(obj, filename):
    joblib.dump(obj, filename)

def load_joblib(filename):
    return joblib.load(filename)

# --- Mathematical & Statistical Helpers ---
def safe_divide(a, b, default=0.0):
    """
    Safely divide two numbers, returning default if denominator is zero.
    """
    try:
        return a / b if b != 0 else default
    except Exception:
        return default

def rolling_sharpe(returns, window=60):
    """
    Calculate rolling Sharpe ratio.
    """
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return mean / (std + 1e-8)

def rolling_drawdown(equity_curve):
    """
    Calculate rolling drawdown.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / (roll_max + 1e-8)
    return drawdown

# --- System Interactions ---
def ensure_dir(path):
    """
    Ensure a directory exists.
    """
    os.makedirs(path, exist_ok=True)

# --- Visualization Helpers ---
def plot_time_series(df, columns=None, title="Time Series", figsize=(12, 6)):
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
    plt.figure(figsize=figsize)
    plt.plot(equity_curve)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

def plot_scatter(x, y, xlabel="X", ylabel="Y", title="Scatter Plot", figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()