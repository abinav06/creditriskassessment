import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import re
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from ml.preprocess import load_and_clean

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "lending_club_loan.csv"

print("Loading and cleaning data for tuning...")
df = load_and_clean(CSV_PATH, nrows=50000)
X = df.drop(columns=["target"])
y = df["target"]

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')
X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in X.columns]

def objective(trial, X, y):
    params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 2000,
        'boosting_type': 'gbdt', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
        
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        imputer = SimpleImputer(strategy='median')
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, preds)
        scores.append(fold_score)

        trial.report(fold_score, fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

pruner = optuna.pruners.MedianPruner(n_warmup_steps=1) # Prune after 2nd fold (0-indexed)
study = optuna.create_study(direction='maximize', study_name='lgbm_credit_risk_robust_v2', pruner=pruner)

print("Starting robust hyperparameter tuning with Optuna (Corrected CV and Pruning)...")
study.optimize(lambda trial: objective(trial, X, y), n_trials=50, show_progress_bar=True)

print("\n--- Tuning Complete ---")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial value (Average CV AUC): {study.best_value:.4f}")
print("\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"    '{key}': {value},")