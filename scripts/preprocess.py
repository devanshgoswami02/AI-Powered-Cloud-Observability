import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 2: Data Preprocessing
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PREPROCESSING STARTED")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. Select features
# ------------------------------------------------------------

features = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent"
]

X = df[features].copy()

# Preserve original dataset row numbers.
original_indices = np.arange(len(X))


# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------

X_train, X_test, train_indices, test_indices = train_test_split(
    X,
    original_indices,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ------------------------------------------------------------
# 4. Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 5. Convert back to DataFrames
# ------------------------------------------------------------

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=features,
    index=train_indices
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=features,
    index=test_indices
)


# ------------------------------------------------------------
# 6. Save scaler
# ------------------------------------------------------------

scaler_path = OUTPUT_DIR / "scaler.joblib"

joblib.dump(
    scaler,
    scaler_path
)


# ------------------------------------------------------------
# 7. Save processed datasets
# ------------------------------------------------------------

train_path = OUTPUT_DIR / "X_train_scaled.csv"
test_path = OUTPUT_DIR / "X_test_scaled.csv"

X_train_scaled_df.to_csv(train_path)
X_test_scaled_df.to_csv(test_path)


# ------------------------------------------------------------
# 8. Print verification information
# ------------------------------------------------------------

print("\nScaled training shape:")
print(X_train_scaled_df.shape)

print("\nScaled testing shape:")
print(X_test_scaled_df.shape)

print("\nTraining means:")
print(X_train_scaled_df[features].mean())

print("\nTraining standard deviations:")
print(X_train_scaled_df[features].std())

print("\nSaved files:")

print(scaler_path)
print(train_path)
print(test_path)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)