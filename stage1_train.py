"""
Shape: (2430, 16)
Target_24m
0    1883
1     547
Name: count, dtype: int64
/Users/sophiechoe/Health_AI/mci_ad/venv/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.py:1213: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis, out=out, keepdims=keepdims)

==============================
24-MONTH PROGRESSION MODEL (STAGE 1)
==============================
AUC            : 0.9071
Best Threshold : 0.1141
Accuracy       : 0.8189
Sensitivity    : 0.9083
Specificity    : 0.7931

Confusion Matrix:
[[299  78]
 [ 10  99]]
"""

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("adni_24m_progression_dataset.csv")

print("Shape:", df.shape)
print(df["Target_24m"].value_counts())

# =========================================================
# 2. FEATURES
# =========================================================

FEATURES = [
    "AGE",
    "PTEDUCAT",
    "PTGENDER",
    "APOE4",
    "MMSE",
    "MOCA",
    "EcogSPTotal",
    "EcogPtMem",
    "RAVLT_forgetting",
    "Hippocampus",
    "Entorhinal",
    "Ventricles"
]

TARGET = "Target_24m"

# =========================================================
# 3. CLEAN DATA
# =========================================================

model_df = df[FEATURES + [TARGET]].copy()

for col in FEATURES:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[col] = model_df[col].fillna(model_df[col].median())

model_df = model_df.dropna(subset=[TARGET])

X = model_df[FEATURES]
y = model_df[TARGET]

# =========================================================
# 4. SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 5. TRAIN MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# =========================================================
# 6. PREDICTIONS
# =========================================================

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

# =========================================================
# 7. THRESHOLD (SENSITIVITY TARGET = 0.90)
# =========================================================

target_sens = 0.90

thresholds = np.linspace(0.01, 0.99, 500)

best_threshold = 0.5
best_diff = 1e9
best_metrics = None

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)

    diff = abs(sens - target_sens)

    if diff < best_diff:
        best_diff = diff
        best_threshold = t
        best_metrics = (acc, sens, spec, tn, fp, fn, tp)

accuracy, sensitivity, specificity, tn, fp, fn, tp = best_metrics

# =========================================================
# 8. PRINT RESULTS
# =========================================================

print("\n==============================")
print("24-MONTH PROGRESSION MODEL (STAGE 1)")
print("==============================")

print(f"AUC            : {auc:.4f}")
print(f"Best Threshold : {best_threshold:.4f}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Sensitivity    : {sensitivity:.4f}")
print(f"Specificity    : {specificity:.4f}")

print("\nConfusion Matrix:")
print(np.array([[tn, fp],
                [fn, tp]]))

# =========================================================
# 9. SAVE MODEL ARTIFACTS
# =========================================================

joblib.dump(model, "stage1_progression_xgb.pkl")

with open("stage1_threshold.json", "w") as f:
    json.dump({
        "threshold": float(best_threshold),
        "target_sensitivity": target_sens,
        "auc": float(auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "features": FEATURES
    }, f, indent=2)

print("\nSaved: stage1_progression_xgb.pkl")
print("Saved: stage1_threshold.json")

# =========================================================
# 10. FEATURE IMPORTANCE PLOT
# =========================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

importance.to_csv("feature_importance_24m_xgb.csv", index=False)

plt.figure(figsize=(8, 6))
plt.barh(importance["feature"], importance["importance"])
plt.title("Feature Importance (Stage 1 MRI Progression Model)")
plt.tight_layout()

plt.savefig("feature_importance_24m_xgb.png", dpi=300)
plt.show()

print("Saved: feature_importance_24m_xgb.png")