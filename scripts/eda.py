import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 1: Exploratory Data Analysis
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
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDescriptive statistics:")
print(df.describe())

# ------------------------------------------------------------
# Feature list
# ------------------------------------------------------------

features = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent"
]

# ------------------------------------------------------------
# 1. Feature vs Row Index plots
# ------------------------------------------------------------

for feature in features:

    plt.figure(figsize=(10, 5))

    plt.plot(
        df.index,
        df[feature],
        linewidth=1
    )

    plt.xlabel("Row Index")
    plt.ylabel(feature)
    plt.title(f"{feature} vs Row Index")

    plt.tight_layout()

    filename = OUTPUT_DIR / f"{feature}_vs_row_index.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# 2. Histograms
# ------------------------------------------------------------

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature],
        bins=30
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature}")

    plt.tight_layout()

    filename = OUTPUT_DIR / f"{feature}_histogram.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# 3. Correlation matrix
# ------------------------------------------------------------

correlation_matrix = df[features].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

filename = OUTPUT_DIR / "correlation_matrix.png"

plt.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n" + "=" * 60)
print("EDA VISUALIZATIONS COMPLETE")
print("=" * 60)

print(f"\nGraphs saved to:")
print(OUTPUT_DIR)