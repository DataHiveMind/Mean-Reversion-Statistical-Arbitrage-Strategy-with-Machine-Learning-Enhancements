# Mean-Reversion Statistical Arbitrage Strategy with Machine Learning Enhancements

---

## Overview

This project implements a robust, production-grade statistical arbitrage framework focused on mean-reversion trading strategies. It leverages advanced machine learning techniques for pair selection, signal generation, and dynamic risk management. The system is modular, scalable, and designed for both research and real-world deployment, supporting tick, minute, and daily data.

---

## Contact

**Author:** [Your Name](Kenneth LeGare)
**Email:** [your.email@example.com](kennethlegare5@gmail.com)
**GitHub:** [github.com/yourusername/Mean-Reversion-Statistical-Arbitrage-Strategy-with-Machine-Learning-Enhancements](https://github.com/DataHiveMind/Mean-Reversion-Statistical-Arbitrage-Strategy-with-Machine-Learning-Enhancements)

---

## Problem Statement

Traditional statistical arbitrage strategies often rely on static rules and simple statistical tests, which can fail in dynamic, non-stationary markets. This project addresses the need for:

- **Robust pair selection** using advanced cointegration and clustering.
- **Dynamic signal generation** with machine learning models.
- **Adaptive risk management** that responds to changing market regimes.
- **Realistic backtesting** with event-driven simulation, slippage, and stress testing.

---

## Technology Stack

- **Python 3.8+**
- **Data Handling:** pandas, numpy, yfinance, pyarrow
- **Statistical Analysis:** statsmodels, scipy, mlfinlab, arch
- **Machine Learning:** scikit-learn, xgboost, lightgbm, catboost, optuna, shap, lime
- **Visualization:** matplotlib, seaborn, plotly
- **Backtesting & Simulation:** Custom event-driven engine, exchange_calendars
- **Utilities:** joblib, pydantic, tqdm, hmmlearn, pykalman

---

## Solution & Workflow

1. **Data Ingestion & Cleaning**
   - Load tick/minute/daily data using `src/data_handler.py`
   - Clean and preprocess data (missing values, outlier detection, corporate actions)

2. **Pair Selection & Feature Engineering**
   - Identify optimal pairs using cointegration, clustering, and factor models (`src/pair_selection.py`)
   - Engineer features capturing microstructure, mean-reversion, and market regime (`src/features.py`)

3. **Machine Learning Model Development**
   - Train and tune ensemble models (XGBoost, LightGBM, CatBoost) for signal generation (`src/ml_models.py`)
   - Use walk-forward cross-validation and leakage prevention
   - Interpret models with SHAP/LIME

4. **Strategy Backtesting & Analysis**
   - Run event-driven backtests with realistic slippage and order book simulation (`src/backtest.py`)
   - Analyze performance: equity curve, drawdown, rolling Sharpe, trade diagnostics, VaR/CVaR, scenario stress tests

5. **Deployment & Monitoring**
   - Save and version models for production
   - Outline monitoring and retraining pipelines

---

## Future Enhancements

- **Deep Learning Models:** Integrate LSTM/TCN for sequence modeling of spreads.
- **Reinforcement Learning:** Explore RL for optimal execution and dynamic position sizing.
- **Portfolio-Level Risk:** Expand to multi-pair and cross-asset portfolio management.
- **Live Trading Integration:** Connect to real-time data feeds and broker APIs.
- **Automated Monitoring:** Implement drift detection, performance monitoring, and auto-retraining.
- **Enhanced Visualization:** Add dashboards for live strategy monitoring and analytics.

---

## Getting Started

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/yourusername/Mean-Reversion-Statistical-Arbitrage-Strategy-with-Machine-Learning-Enhancements.git
   cd Mean-Reversion-Statistical-Arbitrage-Strategy-with-Machine-Learning-Enhancements
   pip install -r requirements.txt