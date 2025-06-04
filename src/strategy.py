import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(
        self,
        model,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        position_sizing="fixed",
        notional=10000,
        max_drawdown=0.2,
        stop_loss_pct=0.05
    ):
        """
        model: trained ML model with predict/predict_proba methods
        entry_z: z-score threshold for entry
        exit_z: z-score threshold for exit
        stop_z: z-score threshold for stop-loss
        position_sizing: 'fixed' or 'volatility'
        notional: fixed notional per trade
        max_drawdown: max allowed drawdown (fraction)
        stop_loss_pct: per-trade stop-loss (fraction)
        """
        self.model = model
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.position_sizing = position_sizing
        self.notional = notional
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.positions = []
        self.equity_curve = []

    def generate_signals(self, features_df):
        """
        Generate trading signals using ML model and z-score thresholds.
        Returns: pd.Series of signals (1=long, -1=short, 0=flat)
        """
        preds = self.model.predict(features_df)
        zscores = features_df['spread_zscore']
        signals = pd.Series(0, index=features_df.index)

        # ML prediction: 1=mean reversion expected, 0=not
        signals[(preds == 1) & (zscores > self.entry_z)] = -1  # Short spread
        signals[(preds == 1) & (zscores < -self.entry_z)] = 1  # Long spread
        # Exit signals
        signals[(zscores.abs() < self.exit_z)] = 0
        return signals

    def size_position(self, volatility=None):
        """
        Determine position size.
        """
        if self.position_sizing == "fixed":
            return self.notional
        elif self.position_sizing == "volatility" and volatility is not None:
            # Example: inverse volatility sizing
            return self.notional / (volatility + 1e-6)
        else:
            return self.notional

    def apply_strategy(self, features_df, prices_df):
        """
        Main loop: generate signals, size positions, manage entries/exits.
        Returns: DataFrame with signals, positions, PnL, equity curve.
        """
        signals = self.generate_signals(features_df)
        position = 0
        entry_price = 0
        peak_equity = 0
        equity = 0
        pnl_list = []
        equity_curve = []
        for i, (dt, row) in enumerate(features_df.iterrows()):
            price = prices_df.loc[dt, 'spread']
            zscore = row['spread_zscore']
            signal = signals.loc[dt]
            volatility = row.get('spread_vol_20', None)
            # Entry logic
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                size = self.size_position(volatility)
            # Exit logic
            elif position != 0:
                # Stop-loss
                if abs(price - entry_price) / abs(entry_price) > self.stop_loss_pct or abs(zscore) > self.stop_z:
                    pnl = position * (price - entry_price) * size
                    equity += pnl
                    position = 0
                    entry_price = 0
                # Normal exit
                elif signal == 0 or np.sign(signal) != np.sign(position):
                    pnl = position * (price - entry_price) * size
                    equity += pnl
                    position = 0
                    entry_price = 0
                else:
                    pnl = 0
            else:
                pnl = 0
            # Risk management: max drawdown
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0 and (peak_equity - equity) / peak_equity > self.max_drawdown:
                position = 0  # Flat all positions
            pnl_list.append(pnl)
            equity_curve.append(equity)
        results = features_df.copy()
        results['signal'] = signals
        results['pnl'] = pnl_list
        results['equity_curve'] = equity_curve
        return results