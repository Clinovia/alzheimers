"""
====================
MRI GATE MODEL (CLINICAL VERSION)
====================
AUC         : 0.825
Threshold   : 0.2064
Accuracy    : 0.662
Sensitivity : 0.9005
Specificity : 0.2857

Classification Report:
              precision    recall  f1-score   support

           0       0.65      0.29      0.40       140
           1       0.67      0.90      0.77       221

    accuracy                           0.66       361
   macro avg       0.66      0.59      0.58       361
weighted avg       0.66      0.66      0.62       361
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD + CLEAN (BASELINE ONLY)
# =========================================================

df = pd.read_csv("ADNIMERGE.csv", low_memory=False)

df = df[df["VISCODE"] == "bl"].copy()
df = df.sort_values("RID").drop_duplicates("RID")

# =========================================================
# 2. TARGET (MRI GATE LABEL)
# =========================================================

df["MRI_GATE_TARGET"] = df["DX"].apply(
    lambda x: 1 if x in ["MCI", "Dementia"] else 0
)

# =========================================================
# 3. FEATURES (MRI-FIRST SET)
# =========================================================

FEATURES = [
    "Hippocampus",
    "Entorhinal",
    "Ventricles",
    "WholeBrain",
    "ICV",
    "AGE",
    "MMSE",
    "APOE4"
]

df = df.dropna(subset=FEATURES + ["MRI_GATE_TARGET"]).copy()

X = df[FEATURES].copy()
y = df["MRI_GATE_TARGET"]

for col in FEATURES:
    X.loc[:, col] = pd.to_numeric(X[col], errors="coerce")
    X.loc[:, col] = X[col].fillna(X[col].median())

# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 5. MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# 6. PREDICTIONS
# =========================================================

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

# =========================================================
# 7. THRESHOLD OPTIMIZATION (SENSITIVITY-FIRST)
# =========================================================

def find_threshold_for_sensitivity(y_true, y_prob, target_sensitivity=0.90):
    thresholds = np.linspace(0.01, 0.99, 500)

    best_threshold = 0.5
    best_diff = 1e9
    best_metrics = None

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if sensitivity >= target_sensitivity:
            diff = sensitivity - target_sensitivity

            if diff < best_diff:
                best_diff = diff
                best_threshold = t
                best_metrics = (accuracy, sensitivity, specificity, tn, fp, fn, tp)

    return best_threshold, best_metrics


TARGET_SENSITIVITY = 0.90

threshold, best_metrics = find_threshold_for_sensitivity(
    y_test,
    y_prob,
    target_sensitivity=TARGET_SENSITIVITY
)

accuracy, sensitivity, specificity, tn, fp, fn, tp = best_metrics

# =========================================================
# 8. RESULTS
# =========================================================

print("\n====================")
print("MRI GATE MODEL (CLINICAL VERSION)")
print("====================")

print("AUC         :", round(auc, 4))
print("Threshold   :", round(threshold, 4))
print("Accuracy    :", round(accuracy, 4))
print("Sensitivity :", round(sensitivity, 4))
print("Specificity :", round(specificity, 4))

print("\nClassification Report:")
print(classification_report(y_test, (y_prob >= threshold).astype(int)))

# =========================================================
# 9. SAVE MODEL ARTIFACTS (CRITICAL FIX)
# =========================================================

joblib.dump(model, "mri_model.pkl")

with open("mri_results.json", "w") as f:
    json.dump({
        "threshold": float(threshold),
        "target_sensitivity": TARGET_SENSITIVITY,
        "auc": float(auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "features": FEATURES
    }, f, indent=2)

print("Saved: mri_model.pkl")
print("Saved: mri_results.json")

# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance")

plt.figure(figsize=(8, 5))
plt.barh(importance["feature"], importance["importance"])
plt.title("MRI Gate Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance_mri.png", dpi=300)
plt.show()

print("Saved: feature_importance_mri.png")

# =========================================================
# 11. SHAP (SAFE VERSION)
# =========================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

# =========================================================
# 12. CLINICAL FUNCTION
# =========================================================

def mri_gate(prob, threshold):
    return (
        "PASS → PET eligible"
        if prob >= threshold
        else "FAIL → monitor"
    )