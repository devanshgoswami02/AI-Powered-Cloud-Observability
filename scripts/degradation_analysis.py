import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 6: Degradation / Trend Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_detection_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Load anomaly detection results
# ------------------------------------------------------------

df = pd.read_csv(RESULTS_PATH)

df = df.sort_values(
    "original_row_index"
).reset_index(drop=True)


print("=" * 60)
print("DEGRADATION / TREND ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. Calculate rolling reconstruction error
# ------------------------------------------------------------

window = 50

df["rolling_reconstruction_error"] = (
    df["reconstruction_error"]
    .rolling(
        window=window,
        min_periods=1
    )
    .mean()
)


# ------------------------------------------------------------
# 3. Calculate rolling anomaly rate
# ------------------------------------------------------------

df["rolling_anomaly_rate"] = (
    df["anomaly"]
    .rolling(
        window=window,
        min_periods=1
    )
    .mean()
)


# ------------------------------------------------------------
# 4. Compare first and last portions
# ------------------------------------------------------------

split_point = len(df) // 2

first_half = df.iloc[:split_point]
second_half = df.iloc[split_point:]


first_mean_error = (
    first_half["reconstruction_error"].mean()
)

second_mean_error = (
    second_half["reconstruction_error"].mean()
)

first_anomaly_rate = (
    first_half["anomaly"].mean() * 100
)

second_anomaly_rate = (
    second_half["anomaly"].mean() * 100
)


print("\nFirst half mean reconstruction error:")
print(first_mean_error)

print("\nSecond half mean reconstruction error:")
print(second_mean_error)

print("\nFirst half anomaly rate:")
print(f"{first_anomaly_rate:.2f}%")

print("\nSecond half anomaly rate:")
print(f"{second_anomaly_rate:.2f}%")


# ------------------------------------------------------------
# 5. Calculate overall trend using linear regression
# ------------------------------------------------------------

x = df["original_row_index"].values
y = df["reconstruction_error"].values

slope, intercept = np.polyfit(
    x,
    y,
    1
)

trend_values = (
    slope * x + intercept
)


print("\nReconstruction error trend slope:")
print(slope)

if slope > 0:
    trend_direction = "Increasing"
elif slope < 0:
    trend_direction = "Decreasing"
else:
    trend_direction = "Flat"

print("\nOverall trend direction:")
print(trend_direction)


# ------------------------------------------------------------
# 6. Highest-risk observations
# ------------------------------------------------------------

top_risk = (
    df[
        [
            "original_row_index",
            "reconstruction_error",
            "anomaly"
        ]
    ]
    .sort_values(
        "reconstruction_error",
        ascending=False
    )
    .head(20)
)


print("\nTop 20 highest-risk observations:")

print(top_risk.to_string(index=False))


# ------------------------------------------------------------
# 7. Save trend analysis data
# ------------------------------------------------------------

trend_path = (
    OUTPUT_DIR
    / "degradation_trend_results.csv"
)

df.to_csv(
    trend_path,
    index=False
)


# ------------------------------------------------------------
# 8. Reconstruction error trend plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    df["original_row_index"],
    df["reconstruction_error"],
    linewidth=0.8,
    alpha=0.5,
    label="Reconstruction Error"
)

plt.plot(
    df["original_row_index"],
    df["rolling_reconstruction_error"],
    linewidth=2,
    label=f"{window}-Observation Rolling Mean"
)

plt.plot(
    x,
    trend_values,
    linestyle="--",
    linewidth=2,
    label="Linear Trend"
)

plt.xlabel(
    "Original Row Index"
)

plt.ylabel(
    "Reconstruction Error"
)

plt.title(
    "Degradation-Risk Proxy: Reconstruction Error Trend"
)

plt.legend()

plt.tight_layout()

trend_plot_path = (
    OUTPUT_DIR
    / "degradation_risk_trend.png"
)

plt.savefig(
    trend_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 9. Rolling anomaly rate plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["original_row_index"],
    df["rolling_anomaly_rate"] * 100,
    linewidth=2
)

plt.xlabel(
    "Original Row Index"
)

plt.ylabel(
    "Rolling Anomaly Rate (%)"
)

plt.title(
    f"{window}-Observation Rolling Anomaly Rate"
)

plt.tight_layout()

anomaly_rate_path = (
    OUTPUT_DIR
    / "rolling_anomaly_rate.png"
)

plt.savefig(
    anomaly_rate_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. Completion
# ------------------------------------------------------------

print("\nSaved files:")

print(trend_path)
print(trend_plot_path)
print(anomaly_rate_path)

print("\nNOTE:")
print(
    "The dataset has no timestamp or failure label. "
    "Original row index is therefore used only as a "
    "sequence/time proxy."
)

print("\n" + "=" * 60)
print("DEGRADATION ANALYSIS COMPLETE")
print("=" * 60)