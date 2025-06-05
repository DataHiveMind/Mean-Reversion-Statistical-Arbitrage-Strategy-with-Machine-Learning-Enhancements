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
        stop_loss_pct=0.05,
        use_ml_confidence=False,
        ensemble_models=None,
        volatility_target=None,
        var_limit=None,
        cvar_limit=None,
        circuit_breaker_threshold=0.3,
        correlation_breakdown_threshold=0.3,
        close_on_friday=False
    ):
        """
        model: trained ML model with predict/predict_proba methods
        use_ml_confidence: use ML model's probability/confidence for sizing
        ensemble_models: list of (model, weight) tuples for ensemble signals
        volatility_target: target volatility for position sizing
        var_limit: Value-at-Risk limit (fraction of notional)
        cvar_limit: Conditional VaR limit (fraction of notional)
        circuit_breaker_threshold: max loss fraction to trigger shutdown
        correlation_breakdown_threshold: min rolling correlation to halt trading
        close_on_friday: close positions before weekend
        """
        self.model = model
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.position_sizing = position_sizing
        self.notional = notional
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.use_ml_confidence = use_ml_confidence
        self.ensemble_models = ensemble_models
        self.volatility_target = volatility_target
        self.var_limit = var_limit
        self.cvar_limit = cvar_limit
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.correlation_breakdown_threshold = correlation_breakdown_threshold
        self.close_on_friday = close_on_friday
        self.positions = []
        self.equity_curve = []
        self.state = {"open_position": 0, "entry_price": 0, "pnl": 0, "equity": 0}

    def combine_signals(self, features_df):
        """
        Combine signals from ML model and ensemble/traditional rules.
        """
        if self.ensemble_models:
            signals = np.zeros(len(features_df))
            for model, weight in self.ensemble_models:
                signals += weight * model.predict(features_df)
            signals = np.sign(signals)
        else:
            signals = self.model.predict(features_df)
        return pd.Series(signals, index=features_df.index)

    def generate_signals(self, features_df):
        """
        Generate trading signals using ML model, ensemble, and z-score thresholds.
        Returns: pd.Series of signals (1=long, -1=short, 0=flat)
        """
        signals = self.combine_signals(features_df)
        zscores = features_df['spread_zscore']
        # ML prediction: 1=mean reversion expected, 0=not
        final_signals = pd.Series(0, index=features_df.index)
        final_signals[(signals == 1) & (zscores > self.entry_z)] = -1  # Short spread
        final_signals[(signals == 1) & (zscores < -self.entry_z)] = 1  # Long spread
        final_signals[(zscores.abs() < self.exit_z)] = 0
        return final_signals

    def size_position(self, volatility=None, ml_confidence=None, price=None, pnl_history=None):
        """
        Determine position size using various methods.
        """
        size = self.notional
        # Volatility targeting
        if self.position_sizing == "volatility" and volatility is not None and self.volatility_target:
            size = self.volatility_target / (volatility + 1e-6) * self.notional
        # ML confidence scaling
        if self.use_ml_confidence and ml_confidence is not None:
            size *= ml_confidence
        # Kelly criterion (theoretical, often scaled down)
        if pnl_history is not None and len(pnl_history) > 20:
            mean_pnl = np.mean(pnl_history[-20:])
            std_pnl = np.std(pnl_history[-20:]) + 1e-8
            kelly_fraction = mean_pnl / (std_pnl ** 2)
            size *= max(0, min(kelly_fraction, 1))
        # VaR/CVaR limits (simplified)
        if self.var_limit and price is not None and volatility is not None:
            var = 1.65 * volatility * size  # 95% VaR
            if var > self.var_limit * self.notional:
                size = self.var_limit * self.notional / (1.65 * volatility + 1e-8)
        return size

    def check_correlation_breakdown(self, features_df, window=60, threshold=None):
        """
        Detect correlation breakdown and halt trading if needed.
        """
        if threshold is None:
            threshold = self.correlation_breakdown_threshold
        if 'spread' in features_df and 'hedge_asset' in features_df:
            corr = features_df['spread'].rolling(window).corr(features_df['hedge_asset'])
            if corr.iloc[-1] < threshold:
                return True
        return False

    def apply_strategy(self, features_df, prices_df):
        """
        Main loop: generate signals, size positions, manage entries/exits, risk controls.
        Returns: DataFrame with signals, positions, PnL, equity curve.
        """
        signals = self.generate_signals(features_df)
        position = 0
        entry_price = 0
        peak_equity = 0
        equity = 0
        pnl_list = []
        equity_curve = []
        ml_confidences = None
        if self.use_ml_confidence and hasattr(self.model, "predict_proba"):
            ml_confidences = self.model.predict_proba(features_df)[:, 1]
        pnl_history = []
        for i, (dt, row) in enumerate(features_df.iterrows()):
            price = prices_df.loc[dt, 'spread']
            zscore = row['spread_zscore']
            signal = signals.loc[dt]
            volatility = row.get('spread_vol_20', None)
            ml_conf = ml_confidences[i] if ml_confidences is not None else None

            # Circuit breaker: extreme drawdown
            if peak_equity > 0 and (peak_equity - equity) / peak_equity > self.circuit_breaker_threshold:
                position = 0
                entry_price = 0

            # Correlation breakdown
            if self.check_correlation_breakdown(features_df.iloc[:i+1]):
                position = 0
                entry_price = 0

            # Overnight/weekend exposure management
            if self.close_on_friday and dt.weekday() == 4:
                position = 0
                entry_price = 0

            # Entry logic
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                size = self.size_position(volatility, ml_conf, price, pnl_history)
            # Exit logic
            elif position != 0:
                # Stop-loss
                if abs(price - entry_price) / (abs(entry_price) + 1e-8) > self.stop_loss_pct or abs(zscore) > self.stop_z:
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
            pnl_history.append(pnl)
            equity_curve.append(equity)

        results = features_df.copy()
        results['signal'] = signals
        results['pnl'] = pnl_list
        results['equity_curve'] = equity_curve
        return results