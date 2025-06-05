import numpy as np
import pandas as pd
from enum import Enum
import plotly.graph_objs as go

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"

class Event:
    def __init__(self, event_type):
        self.type = event_type

class MarketEvent(Event):
    def __init__(self, timestamp, data):
        super().__init__(EventType.MARKET)
        self.timestamp = timestamp
        self.data = data

class SignalEvent(Event):
    def __init__(self, timestamp, symbol, signal_type, strength):
        super().__init__(EventType.SIGNAL)
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength

class OrderEvent(Event):
    def __init__(self, timestamp, symbol, order_type, quantity, direction):
        super().__init__(EventType.ORDER)
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

class FillEvent(Event):
    def __init__(self, timestamp, symbol, quantity, direction, fill_cost, commission, slippage):
        super().__init__(EventType.FILL)
        self.timestamp = timestamp
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission
        self.slippage = slippage

class CommissionModel:
    def calculate(self, quantity, price):
        # Example: $0.005 per share, min $1
        return max(1.0, 0.005 * quantity)

class SlippageModel:
    def calculate(self, quantity, price, volatility, avg_daily_volume):
        # Example: Slippage increases with order size and volatility
        impact = 0.0001 * (quantity / avg_daily_volume) * volatility
        return price * impact

class Backtester:
    def __init__(self, strategy, features_df, prices_df, commission_model=None, slippage_model=None, transaction_cost=0.0, slippage=0.0):
        """
        strategy: instance of MeanReversionStrategy
        features_df: DataFrame of features (with index as datetime)
        prices_df: DataFrame with at least 'spread' column (with index as datetime)
        commission_model: instance of CommissionModel
        slippage_model: instance of SlippageModel
        transaction_cost: cost per trade (as fraction, e.g., 0.001 for 0.1%)
        slippage: slippage per trade (as fraction)
        """
        self.strategy = strategy
        self.features_df = features_df
        self.prices_df = prices_df
        self.commission_model = commission_model or CommissionModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.results = None

    def run(self):
        """
        Run the backtest simulation.
        """
        results = self.strategy.apply_strategy(self.features_df, self.prices_df)
        # Apply transaction costs and slippage
        trades = results['signal'].diff().fillna(0).abs()
        costs = trades * self.transaction_cost * self.prices_df['spread']
        slippage_costs = trades * self.slippage * self.prices_df['spread']
        results['pnl_net'] = results['pnl'] - costs - slippage_costs
        results['equity_curve_net'] = results['pnl_net'].cumsum()
        results['pnl_mean_reversion'] = ... # Calculate based on intraday signals
        results['pnl_overnight'] = ...      # Calculate based on overnight moves
        results['pnl_transaction_costs'] = costs + slippage_costs
        self.results = results
        return results

    def compute_performance_metrics(self, risk_free_rate=0.0, periods_per_year=252):
        """
        Compute key performance metrics.
        Returns: dict of metrics
        """
        if self.results is None:
            raise ValueError("Run the backtest first.")

        pnl = self.results['pnl_net']
        equity_curve = self.results['equity_curve_net']
        returns = pnl / (self.strategy.notional if self.strategy.notional != 0 else 1)
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

        # Annualized return
        total_return = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0
        annualized_return = (1 + total_return / self.strategy.notional) ** (periods_per_year / len(equity_curve)) - 1 if len(equity_curve) > 0 else 0

        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / (excess_returns.std() + 1e-8)

        # Sortino Ratio
        downside = excess_returns[excess_returns < 0]
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / (downside.std() + 1e-8)

        # Max Drawdown
        roll_max = equity_curve.cummax()
        drawdown = equity_curve - roll_max
        max_drawdown = drawdown.min()

        # Win rate
        wins = (pnl > 0).sum()
        total_trades = (self.results['signal'].diff().abs() > 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": total_trades
        }
        return metrics

    def export_results(self, equity_curve_path="equity_curve.csv", trade_log_path="trade_log.csv"):
        """
        Export equity curve and trade log to CSV files.
        """
        if self.results is None:
            raise ValueError("No results to export.")
        self.results[['equity_curve_net']].to_csv(equity_curve_path)
        trade_log = self.results[self.results['signal'].diff().abs() > 0]
        trade_log.to_csv(trade_log_path)

    def simulate_fill(self, order, price, volatility, avg_daily_volume, bid_ask_spread):
        commission = self.commission_model.calculate(order.quantity, price)
        slippage = self.slippage_model.calculate(order.quantity, price, volatility, avg_daily_volume)
        fill_price = price + (bid_ask_spread / 2 if order.direction == "BUY" else -bid_ask_spread / 2) + slippage
        return fill_price, commission, slippage

    def stress_test(self, scenarios):
        results = {}
        for scenario in scenarios:
            # Modify prices_df/features_df according to scenario
            # e.g., increase volatility, simulate crash, etc.
            scenario_results = self.run()
            results[scenario['name']] = self.compute_performance_metrics()
        return results

def bootstrap_sharpe(returns, n_bootstrap=1000):
    sharpe_dist = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        sharpe = np.mean(sample) / (np.std(sample) + 1e-8)
        sharpe_dist.append(sharpe)
    return np.percentile(sharpe_dist, [2.5, 97.5])  # 95% CI

def deflated_sharpe(sharpe, n_trials):
    # See Bailey & Lopez de Prado (2012)
    from scipy.stats import norm
    expected_max = norm.ppf(1 - 1.0 / n_trials)
    return sharpe - expected_max

def plot_equity_curve(equity_curve):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity Curve'))
    fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity')
    fig.show()