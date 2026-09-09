import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path


# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 4: Reconstruction Error & Anomaly Detection
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
TEST_PATH = PROJECT_ROOT / "outputs" / "X_test_scaled.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Load trained model and test data
# ------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

test_df = pd.read_csv(TEST_PATH, index_col=0)

X_test = test_df.values
original_indices = test_df.index.values

print("=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

print("\nTest data shape:")
print(X_test.shape)


# ------------------------------------------------------------
# 2. Reconstruct test observations
# ------------------------------------------------------------

X_test_reconstructed = model.predict(
    X_test,
    verbose=0
)


# ------------------------------------------------------------
# 3. Calculate reconstruction error
# ------------------------------------------------------------

reconstruction_errors = np.mean(
    np.square(
        X_test - X_test_reconstructed
    ),
    axis=1
)


# ------------------------------------------------------------
# 4. Calculate 95th percentile threshold
# ------------------------------------------------------------

threshold = np.percentile(
    reconstruction_errors,
    95
)


# ------------------------------------------------------------
# 5. Classify observations
# ------------------------------------------------------------

anomaly_flags = (
    reconstruction_errors > threshold
)


# ------------------------------------------------------------
# 6. Create results DataFrame
# ------------------------------------------------------------

results = test_df.copy()

results["reconstruction_error"] = reconstruction_errors

results["anomaly"] = anomaly_flags.astype(int)

results["original_row_index"] = original_indices


# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("\nReconstruction error statistics:")

print(
    pd.Series(reconstruction_errors).describe()
)

print("\n95th percentile threshold:")
print(threshold)

print("\nDetected anomalies:")
print(anomaly_flags.sum())

print(
    "\nAnomaly percentage:"
)
print(
    anomaly_flags.mean() * 100
)


# ------------------------------------------------------------
# 8. Show highest-error observations
# ------------------------------------------------------------

print("\nTop 10 highest reconstruction errors:")

top_anomalies = (
    results
    .sort_values(
        "reconstruction_error",
        ascending=False
    )
    .head(10)
)

print(
    top_anomalies[
        [
            "original_row_index",
            "reconstruction_error",
            "anomaly"
        ]
    ]
)


# ------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------

results_path = (
    OUTPUT_DIR
    / "anomaly_detection_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------------------
# 10. Plot reconstruction error distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    reconstruction_errors,
    bins=40
)

plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="95th Percentile Threshold"
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")

plt.title(
    "Reconstruction Error Distribution"
)

plt.legend()

plt.tight_layout()

error_distribution_path = (
    OUTPUT_DIR
    / "reconstruction_error_distribution.png"
)

plt.savefig(
    error_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 11. Plot reconstruction error by original row index
# ------------------------------------------------------------

plot_data = results.sort_values(
    "original_row_index"
)

plt.figure(figsize=(12, 5))

plt.plot(
    plot_data["original_row_index"],
    plot_data["reconstruction_error"],
    linewidth=1,
    label="Reconstruction Error"
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="95th Percentile Threshold"
)

plt.xlabel("Original Row Index")
plt.ylabel("Reconstruction Error")

plt.title(
    "Reconstruction Error vs Original Row Index"
)

plt.legend()

plt.tight_layout()

error_trend_path = (
    OUTPUT_DIR
    / "reconstruction_error_trend.png"
)

plt.savefig(
    error_trend_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 12. Completion
# ------------------------------------------------------------

print("\nSaved files:")

print(results_path)
print(error_distribution_path)
print(error_trend_path)

print("\n" + "=" * 60)
print("ANOMALY DETECTION COMPLETE")
print("=" * 60)