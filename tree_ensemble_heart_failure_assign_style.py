# Tree Ensembles on Heart Failure Prediction


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 1234
TARGET_COL = "HeartDisease"   # in heart.csv

# =========================
# 1) Load & prepare dataset
# =========================
df = pd.read_csv("heart.csv")  # assume file exists locally
assert TARGET_COL in df.columns, f"Expected '{TARGET_COL}' in heart.csv"

# Ensure any object/categorical are numeric (most versions are already numeric)
df = df.copy()
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = pd.factorize(df[c])[0]

X = df.drop(columns=[TARGET_COL]).values
y = df[TARGET_COL].values
feature_names = [c for c in df.columns if c != TARGET_COL]

# Train / Val / Test 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train: {X_train.shape},  Val: {X_val.shape},  Test: {X_test.shape}")
print(f"Positive rate (overall): {y.mean():.3f}")

# =================================
# 2) Baseline — Decision Tree (DT)
# =================================
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
yhat_val_dt = dt.predict(X_val)
acc_val_dt = accuracy_score(y_val, yhat_val_dt)
print(f"\nDecision Tree — Val accuracy: {acc_val_dt:.4f}")
print("DT (Val) classification report:\n", classification_report(y_val, yhat_val_dt, digits=4))

# =========================================
# 3) Random Forest — GridSearchCV (only 3 params)
# =========================================
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 6, 12],
    'min_samples_split': [2, 5, 10],
}
rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)
print("\nRF GridSearchCV — best params:", rf_grid.best_params_)
print(f"RF GridSearchCV — best CV score: {rf_grid.best_score_:.4f}")

rf_best = rf_grid.best_estimator_
acc_val_rf = accuracy_score(y_val, rf_best.predict(X_val))
print(f"RF (best) — Val accuracy: {acc_val_rf:.4f}")

# =======================================
# 6) XGBoost — GridSearchCV (only 3 params)
# =======================================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    n_jobs=-1
)
xgb_param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}
xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_train, y_train)
print("\nXGB GridSearchCV — best params:", xgb_grid.best_params_)
print(f"XGB GridSearchCV — best CV score: {xgb_grid.best_score_:.4f}")

xgb_best = xgb_grid.best_estimator_
acc_val_xgb = accuracy_score(y_val, xgb_best.predict(X_val))
print(f"XGB (best) — Val accuracy: {acc_val_xgb:.4f}")

# ===========================
# 7) Final evaluation (Test)
# ===========================
best_name = None
best_model = None
best_val = -1

if acc_val_rf >= acc_val_xgb:
    best_name, best_model, best_val = "RandomForest (best GS)", rf_best, acc_val_rf
else:
    best_name, best_model, best_val = "XGBoost (best GS)", xgb_best, acc_val_xgb

acc_test_best = accuracy_score(y_test, best_model.predict(X_test))
print(f"\nSelected model (Val): {best_name} — Val: {best_val:.4f} | Test: {acc_test_best:.4f}")

print("\nConfusion matrix on Test:")
print(confusion_matrix(y_test, best_model.predict(X_test)))
print("\nClassification report on Test:\n", classification_report(y_test, best_model.predict(X_test), digits=4))
