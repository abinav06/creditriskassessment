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
CSV_PATH = ROOT / "lending_club_loan.csv"
ARTIFACTS_PATH = ROOT / "artifacts"
ARTIFACTS_PATH.mkdir(exist_ok=True)

print("Loading and cleaning data for final model training...")
df = load_and_clean(CSV_PATH, nrows=50000)

X = df.drop(columns=["target"])
y = df["target"]

print("--- Pipeline requires the following columns: ---")
print(X.columns.tolist())


numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

best_params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 300,
    'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'learning_rate': 0.01196, 'num_leaves': 91, 'max_depth': 9,
    'lambda_l1': 0.00181, 'lambda_l2': 0.00523, 'feature_fraction': 0.5216,
    'bagging_fraction': 0.5004, 'bagging_freq': 1, 'min_child_samples': 24,
}
model = lgb.LGBMClassifier(**best_params)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("Training the final pipeline on all available data...")

categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_features]

fit_params = {
    'classifier__categorical_feature': categorical_feature_indices
}

pipeline.fit(X, y, **fit_params)

print("Saving pipeline artifact...")
joblib.dump(pipeline, ARTIFACTS_PATH / "credit_risk_pipeline.joblib")
print(f"Pipeline saved successfully to {ARTIFACTS_PATH / 'credit_risk_pipeline.joblib'}")