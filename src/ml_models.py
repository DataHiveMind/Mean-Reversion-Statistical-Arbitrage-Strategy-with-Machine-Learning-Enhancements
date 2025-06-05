import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class MLModels:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None

    def train(self, X, y, model_type="logistic", params=None):
        """
        Train a supervised learning model.
        model_type: 'logistic', 'random_forest', 'xgboost', 'lightgbm'
        params: dict of model hyperparameters
        """
        if params is None:
            params = {}

        if model_type == "logistic":
            self.model = LogisticRegression(**params)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(**params)
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(**params)
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unsupported or unavailable model_type: {model_type}")

        self.model.fit(X, y)
        return self.model

    def hyperparameter_tuning(self, X, y, model_type="random_forest", param_grid=None, search_type="grid", cv=3, n_iter=10):
        """
        Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
        """
        if param_grid is None:
            param_grid = {}

        if model_type == "logistic":
            base_model = LogisticRegression()
        elif model_type == "random_forest":
            base_model = RandomForestClassifier()
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            base_model = xgb.XGBClassifier()
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            base_model = lgb.LGBMClassifier()
        else:
            raise ValueError(f"Unsupported or unavailable model_type: {model_type}")

        if search_type == "grid":
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        else:
            search = RandomizedSearchCV(base_model, param_grid, n_iter=n_iter, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)

        search.fit(X, y)
        self.model = search.best_estimator_
        return self.model, search.best_params_

    def predict(self, X):
        """
        Predict using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support probability prediction.")

    def save_model(self, filename):
        """
        Save the trained model to disk.
        """
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, filepath)

    def load_model(self, filename):
        """
        Load a trained model from disk.
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model = joblib.load(filepath)
        return self.model

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on given data.
        metrics: list of metrics to compute, e.g. ['accuracy', 'precision', 'recall', 'f1', 'rmse']
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        y_pred = self.predict(X)
        results = {}
        for metric in metrics:
            if metric == 'accuracy':
                results['accuracy'] = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                results['precision'] = precision_score(y_true, y_pred)
            elif metric == 'recall':
                results['recall'] = recall_score(y_true, y_pred)
            elif metric == 'f1':
                results['f1'] = f1_score(y_true, y_pred)
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        return results

    def walk_forward_cv(self, X, y, model_type="random_forest", params=None, n_splits=5):
        """
        Walk-forward cross-validation for time-series.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            self.train(X_train, y_train, model_type=model_type, params=params)
            score = self.evaluate(X_test, y_test)
            scores.append(score)
        return scores

    def bayesian_optimization(self, X, y, model_type="random_forest", n_trials=20):
        import optuna
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            self.train(X, y, model_type=model_type, params=params)
            score = self.evaluate(X, y, metrics=["accuracy"])["accuracy"]
            return score
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def explain_with_shap(self, X):
        import shap
        if hasattr(self.model, "predict_proba"):
            explainer = shap.Explainer(self.model, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X)
        else:
            print("SHAP not supported for this model.")