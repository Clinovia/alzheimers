import pandas as pd
import numpy as np
import json
import joblib
import numpy as np
np.int = int
np.bool = bool
import shap

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
# 3. FEATURES (YOUR STAGE 1 SET)
# =========================================================

FEATURES = [
    "AGE", "PTGENDER", "PTEDUCAT", "APOE4",
    "MMSE",
    "EcogSPTotal",
    "EcogMem_discrepancy",
    "RAVLT_forgetting",
    "RAVLT_immediate",
]

TARGET = "Target_24m"

# =========================================================
# 4. CLEAN DATA (WITH CATEGORICAL GENDER)
# =========================================================

model_df = df[FEATURES + [TARGET]].copy()

# ---- Handle gender explicitly (categorical) ----
# ADNI typically uses "Male"/"Female"
gender_map = {
    "Male": 1,
    "Female": 0
}

model_df["PTGENDER"] = model_df["PTGENDER"].map(gender_map)

# If already numeric (some versions), keep it
model_df["PTGENDER"] = pd.to_numeric(model_df["PTGENDER"], errors="coerce")

# ---- Handle remaining features ----
for col in FEATURES:
    if col == "PTGENDER":
        continue  # already handled

    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[col] = model_df[col].fillna(model_df[col].median())

# Optional: handle missing gender (rare)
model_df["PTGENDER_missing"] = model_df["PTGENDER"].isna().astype(int)
model_df["PTGENDER"] = model_df["PTGENDER"].fillna(model_df["PTGENDER"].median())

# Drop rows with missing target
model_df = model_df.dropna(subset=[TARGET])

X = model_df[FEATURES]
y = model_df[TARGET]

# =========================================================
# 5. SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 6. TRAIN MODEL
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
# 7. PREDICTIONS
# =========================================================

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

# =========================================================
# 8. THRESHOLD (SENSITIVITY TARGET = 0.90)
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
# 9. PRINT RESULTS
# =========================================================

print("\n==============================")
print("STAGE 1: CLINICAL PROGRESSION MODEL")
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
# 10. SAVE MODEL
# =========================================================

joblib.dump(model, "stage1_clinical_xgb.pkl")

with open("stage1_threshold.json", "w") as f:
    json.dump({
        "threshold": float(best_threshold),
        "auc": float(auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "features": FEATURES
    }, f, indent=2)

# =========================================================
# 11. SHAP ANALYSIS
# =========================================================
print("\nComputing SHAP values...")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Convert to numpy array for compatibility
shap_array = shap_values.values

# Save raw SHAP values
shap_df = pd.DataFrame(shap_array, columns=FEATURES)
shap_df.to_csv("stage1_shap_values.csv", index=False)

# Mean absolute SHAP importance
shap_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(shap_array).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=True)

shap_importance.to_csv("stage1_shap_importance.csv", index=False)


# =========================================================
# 12. SHAP BAR PLOT (PUBLICATION QUALITY)
# =========================================================

plt.figure(figsize=(8, 6))
plt.barh(shap_importance["feature"], shap_importance["mean_abs_shap"])
plt.xlabel("Mean |SHAP value|")
plt.title("Stage 1 Clinical Model — SHAP Feature Importance")
plt.tight_layout()

plt.savefig("stage1_shap_bar.png", dpi=300)
plt.show()

print("\nSaved:")
print(" - stage1_clinical_xgb.pkl")
print(" - stage1_threshold.json")
print(" - stage1_shap_values.csv")
print(" - stage1_shap_importance.csv")
print(" - stage1_shap_bar.png")