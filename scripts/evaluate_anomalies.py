import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 5: Synthetic Anomaly Evaluation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
TEST_PATH = PROJECT_ROOT / "outputs" / "X_test_scaled.csv"
RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_detection_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Load model and test data
# ------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

test_df = pd.read_csv(
    TEST_PATH,
    index_col=0
)

X_test = test_df.values.copy()

features = list(test_df.columns)

print("=" * 60)
print("SYNTHETIC ANOMALY EVALUATION")
print("=" * 60)

print("\nOriginal test shape:")
print(X_test.shape)


# ------------------------------------------------------------
# 2. Load existing anomaly threshold
# ------------------------------------------------------------

previous_results = pd.read_csv(
    RESULTS_PATH
)

threshold = np.percentile(
    previous_results["reconstruction_error"],
    95
)

print("\nExisting anomaly threshold:")
print(threshold)


# ------------------------------------------------------------
# 3. Create controlled synthetic anomalies
# ------------------------------------------------------------

rng = np.random.default_rng(42)

X_eval = X_test.copy()

y_true = np.zeros(
    len(X_eval),
    dtype=int
)


# We will modify 40 test observations.
n_anomalies = 40

anomaly_indices = rng.choice(
    len(X_eval),
    size=n_anomalies,
    replace=False
)


# Split anomalies across different resource types.
groups = np.array_split(
    anomaly_indices,
    4
)


# CPU anomalies
for idx in groups[0]:
    X_eval[idx, 0] += 4.0


# Memory anomalies
for idx in groups[1]:
    X_eval[idx, 1] += 4.0


# Network anomalies
for idx in groups[2]:
    X_eval[idx, 2] += 4.0
    X_eval[idx, 3] += 4.0


# Disk I/O anomalies
for idx in groups[3]:
    X_eval[idx, 4] += 5.0


y_true[anomaly_indices] = 1


print("\nSynthetic anomalies injected:")
print(n_anomalies)

print("\nNormal observations:")
print((y_true == 0).sum())

print("\nKnown anomalies:")
print((y_true == 1).sum())


# ------------------------------------------------------------
# 4. Run synthetic data through autoencoder
# ------------------------------------------------------------

X_reconstructed = model.predict(
    X_eval,
    verbose=0
)


# ------------------------------------------------------------
# 5. Calculate reconstruction errors
# ------------------------------------------------------------

reconstruction_errors = np.mean(
    np.square(
        X_eval - X_reconstructed
    ),
    axis=1
)


# ------------------------------------------------------------
# 6. Predict anomalies
# ------------------------------------------------------------

y_pred = (
    reconstruction_errors > threshold
).astype(int)


# ------------------------------------------------------------
# 7. Calculate metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ------------------------------------------------------------
# 8. Print evaluation results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 9. Save detailed evaluation results
# ------------------------------------------------------------

evaluation_results = test_df.copy()

evaluation_results[
    "reconstruction_error"
] = reconstruction_errors

evaluation_results[
    "true_label"
] = y_true

evaluation_results[
    "predicted_label"
] = y_pred

evaluation_results[
    "correct_prediction"
] = (
    y_true == y_pred
)

evaluation_path = (
    OUTPUT_DIR
    / "synthetic_anomaly_evaluation.csv"
)

evaluation_results.to_csv(
    evaluation_path,
    index=False
)


# ------------------------------------------------------------
# 10. Save metrics
# ------------------------------------------------------------

metrics_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_path = (
    OUTPUT_DIR
    / "evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ------------------------------------------------------------
# 11. Plot error comparison
# ------------------------------------------------------------

normal_errors = reconstruction_errors[
    y_true == 0
]

anomaly_errors = reconstruction_errors[
    y_true == 1
]

plt.figure(figsize=(9, 5))

plt.hist(
    normal_errors,
    bins=30,
    alpha=0.7,
    label="Normal"
)

plt.hist(
    anomaly_errors,
    bins=30,
    alpha=0.7,
    label="Synthetic Anomaly"
)

plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="95th Percentile Threshold"
)

plt.xlabel(
    "Reconstruction Error"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Normal vs Synthetic Anomaly Reconstruction Error"
)

plt.legend()

plt.tight_layout()

comparison_plot = (
    OUTPUT_DIR
    / "normal_vs_synthetic_anomaly.png"
)

plt.savefig(
    comparison_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 12. Completion
# ------------------------------------------------------------

print("\nSaved files:")

print(evaluation_path)
print(metrics_path)
print(comparison_plot)

print("\n" + "=" * 60)
print("SYNTHETIC ANOMALY EVALUATION COMPLETE")
print("=" * 60)