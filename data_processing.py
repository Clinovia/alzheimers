import pandas as pd
import numpy as np

# =========================================================
# 1. Load data
# =========================================================

DATA_PATH = "ADNIMERGE.csv"
df = pd.read_csv(DATA_PATH)

print("Raw shape:", df.shape)

# =========================================================
# 2. Sort longitudinally
# =========================================================

df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
df = df.sort_values(["RID", "EXAMDATE"])

# =========================================================
# 3. Define baseline dataset
# =========================================================

baseline = df[df["VISCODE"].isin(["bl", "m00"])].copy()

# Keep only first baseline per RID
baseline = baseline.sort_values("EXAMDATE").groupby("RID").first().reset_index()

print("Baseline shape:", baseline.shape)

# =========================================================
# 4. Define helper: 24-month progression label
# =========================================================

def compute_24m_target(rid, base_date, df, window_months=24):
    """
    Returns 1 if subject progresses to Dementia within 24 months.
    """

    subject = df[df["RID"] == rid].copy()

    # time difference in months
    subject["months_from_base"] = (
        (subject["EXAMDATE"] - base_date).dt.days / 30.44
    )

    within_window = subject[
        (subject["months_from_base"] > 0) &
        (subject["months_from_base"] <= window_months)
    ]

    if within_window.empty:
        return 0

    return int(any(within_window["DX"].astype(str).str.contains("Dementia", na=False)))


# =========================================================
# 5. Apply target creation
# =========================================================

targets = []

for _, row in baseline.iterrows():
    rid = row["RID"]
    base_date = row["EXAMDATE"]

    targets.append(
        compute_24m_target(rid, base_date, df, window_months=24)
    )

baseline["Target_24m"] = targets

print("Target distribution:")
print(baseline["Target_24m"].value_counts())

# =========================================================
# 6. Select modeling features (MVP set)
# =========================================================

features = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",

    "APOE4",

    "MMSE",
    "MOCA",
    "CDRSB",
    "FAQ",

    "EcogSPTotal",
    "EcogPtMem",

    "RAVLT_forgetting",

    "Hippocampus",
    "Entorhinal",
    "Ventricles"
]

# Keep only available columns
model_df = baseline[["RID"] + features + ["Target_24m"]].copy()

# =========================================================
# 7. Clean data
# =========================================================

for col in features:
    if col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df[col] = model_df[col].fillna(model_df[col].median())

model_df = model_df.dropna(subset=["Target_24m"])

print("Final dataset shape:", model_df.shape)

# =========================================================
# 8. Save structured dataset
# =========================================================

model_df.to_csv("adni_24m_progression_dataset.csv", index=False)

print("Saved: adni_24m_progression_dataset.csv")