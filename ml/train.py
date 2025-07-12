# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import shap
# import re
# from pathlib import Path
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt

# # --- 1. Preprocessing Function with Feature Engineering ---

# def load_and_clean(path: str, nrows: int = None):
#     """
#     Loads the data, creates the target, engineers new features,
#     and performs cleaning, now with leakage removal.
#     """
#     df = pd.read_csv(path, low_memory=False, nrows=nrows)
#     df['target'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
#     df.drop(columns=['loan_status'], inplace=True)

#     # --- CHANGE: Added the leaky features identified from the SHAP plot ---
#     # These are columns with information recorded *after* a loan is issued
#     # and would not be available for a new loan applicant.
#     leakage_cols = [
#         'id', 'member_id', 'recoveries', 'collection_recovery_fee',
#         'last_pymnt_amnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
#         'last_credit_pull_d', 'last_pymnt_d', 'out_prncp', 'out_prncp_inv',
#         'total_pymnt', 'total_pymnt_inv',
#         'last_fico_range_high',  # This is the FICO score at the *end* of the loan. LEAKAGE.
#         'last_fico_range_low',   # This is the FICO score at the *end* of the loan. LEAKAGE.
#         'debt_settlement_flag' # This flag is set only after a charge-off. LEAKAGE.
#     ]
#     df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ml.preprocess import load_and_clean

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "lending_club_loan.csv"
ARTIFACTS_PATH = ROOT / "artifacts"
ARTIFACTS_PATH.mkdir(exist_ok=True)

print("Loading and cleaning data for final model training...")
df = load_and_clean(CSV_PATH, nrows=50000)

X = df.drop(columns=["target"])
y = df["target"]

# --- ADD THIS LINE ---
print("--- Pipeline requires the following columns: ---")
# --- AND THIS LINE ---
print(X.columns.tolist())


# --- 2. Define Preprocessing Steps for the Pipeline ---
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Define the Model with Best Hyperparameters ---
best_params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 300,
    'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'learning_rate': 0.01196, 'num_leaves': 91, 'max_depth': 9,
    'lambda_l1': 0.00181, 'lambda_l2': 0.00523, 'feature_fraction': 0.5216,
    'bagging_fraction': 0.5004, 'bagging_freq': 1, 'min_child_samples': 24,
}
model = lgb.LGBMClassifier(**best_params)

# --- 4. Create the Full Prediction Pipeline ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# --- 5. Train the Final Pipeline on ALL Data ---
print("Training the final pipeline on all available data...")

categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_features]

fit_params = {
    'classifier__categorical_feature': categorical_feature_indices
}

pipeline.fit(X, y, **fit_params)

# --- 6. Save the Pipeline Artifact ---
print("Saving pipeline artifact...")
joblib.dump(pipeline, ARTIFACTS_PATH / "credit_risk_pipeline.joblib")
print(f"Pipeline saved successfully to {ARTIFACTS_PATH / 'credit_risk_pipeline.joblib'}")