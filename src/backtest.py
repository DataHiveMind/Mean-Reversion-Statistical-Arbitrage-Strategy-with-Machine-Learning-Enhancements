import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, strategy, features_df, prices_df, transaction_cost=0.0, slippage=0.0):
        """
        strategy: instance of MeanReversionStrategy
        features_df: DataFrame of features (with index as datetime)
        prices_df: DataFrame with at least 'spread' column (with index as datetime)
        transaction_cost: cost per trade (as fraction, e.g., 0.001 for 0.1%)
        slippage: slippage per trade (as fraction)
        """
        self.strategy = strategy
        self.features_df = features_df
        self.prices_df = prices_df
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