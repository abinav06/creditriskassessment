# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import optuna
# import re
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from sklearn.impute import SimpleImputer

# # Reuse the same preprocessing function from the main script
# # (In a real project, this would be in a shared module)
# def load_and_clean(path: str, nrows: int = None):
#     # This is the same function as in train_advanced.py
#     df = pd.read_csv(path, low_memory=False, nrows=nrows)
#     df['target'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
#     df.drop(columns=['loan_status'], inplace=True)
#     leakage_cols = [
#         'id', 'member_id', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 
#         'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'last_credit_pull_d', 
#         'last_pymnt_d', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv'
#     ]
#     df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)
#     df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
#     df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
#     df['credit_history_months'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 30).fillna(0)
#     df['term'] = df['term'].str.extract('(\d+)').astype(float)
#     df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
#     df['installment_to_income_ratio'] = df['installment'] / (df['annual_inc'] / 12 + 1)
#     df.drop(columns=['issue_d', 'earliest_cr_line'], inplace=True)
#     df = df.dropna(thresh=len(df) * 0.8, axis=1)
#     df = df.select_dtypes(include=['number', 'object'])
#     return df

# # --- 1. Load and Prepare Data for Tuning ---
# ROOT = Path(__file__).resolve().parents[1]
# CSV_PATH = ROOT / "data" / "raw" / "lending_club_loan.csv"

# # For tuning, we can use a smaller subset to speed things up
# df = load_and_clean(CSV_PATH, nrows=20000)

# X = df.drop(columns=["target"])
# y = df["target"]

# # Handle categoricals and column names
# for col in X.select_dtypes(include='object').columns:
#     X[col] = X[col].astype('category')
# X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in X.columns]

# # Use a single, stratified split for tuning
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.25, stratify=y, random_state=42
# )

# # Impute missing values
# numeric_cols = X_train.select_dtypes(include=np.number).columns
# imputer = SimpleImputer(strategy='median')
# X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
# X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])


# # --- 2. Define the Optuna Objective Function ---
# def objective(trial):
#     # Define the search space for hyperparameters
#     params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'n_estimators': 1000,
#         'boosting_type': 'gbdt',
#         'random_state': 42,
#         'n_jobs': -1,
#         'verbose': -1,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'max_depth': trial.suggest_int('max_depth', 5, 10),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#     }

#     model = lgb.LGBMClassifier(**params)
    
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='auc',
#         callbacks=[lgb.early_stopping(100, verbose=False)]
#     )
    
#     preds = model.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, preds)
    
#     return auc

# # --- 3. Run the Optimization ---
# print("Starting hyperparameter tuning with Optuna...")
# study = optuna.create_study(direction='maximize', study_name='lgbm_credit_risk')
# study.optimize(objective, n_trials=50) # Run 50 trials

# print("\n--- Tuning Complete ---")
# print(f"Number of finished trials: {len(study.trials)}")
# print("Best trial:")
# best_trial = study.best_trial

# print(f"  Value (AUC): {best_trial.value:.4f}")
# print("  Params: ")
# for key, value in best_trial.params.items():
#     print(f"    '{key}': {value},")

# File: ml/tune_hyperparams.py
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

# --- 1. Load and Prepare Data for Tuning ---
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "lending_club_loan.csv"

print("Loading and cleaning data for tuning...")
df = load_and_clean(CSV_PATH, nrows=50000)
X = df.drop(columns=["target"])
y = df["target"]

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')
X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in X.columns]

# --- 2. Define the Optuna Objective Function with Corrected Pruning ---
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
        # --- FIX 1: Use .copy() to prevent SettingWithCopyWarning ---
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
            # --- FIX 2: Remove the pruning callback from here ---
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, preds)
        scores.append(fold_score)

        # --- FIX 2: Manually report the score of each fold to the pruner ---
        trial.report(fold_score, fold)

        # Manually check if the trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

# --- 3. Run the Optimization ---
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