"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 10E: Explain Bitbrains LSTM Anomalies by Feature Contribution

This script performs feature-level reconstruction error attribution on anomalous
sequences detected in Phase 10D by the trained Bitbrains Seq2Seq LSTM-Autoencoder.

Operational Disclaimer:
  Feature contribution percentages are mathematically derived from feature-level
  reconstruction residuals (squared error between input and reconstructed sequences).
  They represent the relative magnitude of model reconstruction discrepancy across
  telemetry channels. They do NOT represent system failure probabilities, time-to-failure,
  or causal proof of infrastructure failure.

Process:
  1. Load trained LSTM model from outputs/bitbrains/lstm_model/lstm_autoencoder.keras.
  2. Load test sequences from outputs/bitbrains/lstm_sequences/X_test_lstm.npy.
  3. Load anomaly detection results from outputs/bitbrains/lstm_model/lstm_anomaly_results.csv.
  4. Generate reconstructions for all test sequences using the LSTM-Autoencoder.
  5. Calculate per-feature squared reconstruction error for each time step:
       error(t, j) = (x(t, j) - reconstruction(t, j))^2
  6. Average squared error across the 12 time steps for each feature:
       MSE_j = mean_{t=0..11} (error(t, j))
  7. Compute each feature's percentage contribution to the sequence error:
       contribution_j = (MSE_j / sum_{k} MSE_k) * 100%
  8. Filter to anomalous sequences (anomaly == 1) and determine the top-contributing feature.
  9. Save detailed explanations to outputs/bitbrains/lstm_model/lstm_anomaly_explanations.csv.
  10. Generate and save summary to outputs/bitbrains/lstm_model/lstm_explanation_summary.csv.
  11. Create a bar chart showing average feature contributions among anomalous sequences,
      saved to outputs/bitbrains/lstm_model/lstm_feature_contributions.png.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# Paths and Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
SEQUENCES_DIR = BITBRAINS_DIR / "lstm_sequences"
MODEL_DIR = BITBRAINS_DIR / "lstm_model"

MODEL_PATH = MODEL_DIR / "lstm_autoencoder.keras"
TEST_SEQUENCES_PATH = SEQUENCES_DIR / "X_test_lstm.npy"
ANOMALY_RESULTS_PATH = MODEL_DIR / "lstm_anomaly_results.csv"

EXPLANATIONS_OUTPUT_PATH = MODEL_DIR / "lstm_anomaly_explanations.csv"
SUMMARY_OUTPUT_PATH = MODEL_DIR / "lstm_explanation_summary.csv"
BAR_CHART_PATH = MODEL_DIR / "lstm_feature_contributions.png"

# Sequence features in order of columns in X_test_lstm.npy
FEATURE_NAMES = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent"  # disk_io in raw Bitbrains, formatted as disk_io_percent per project standard
]

FEATURE_DISPLAY_LABELS = {
    "cpu_util_percent": "CPU Utilization (%)",
    "mem_util_percent": "Memory Utilization (%)",
    "net_in": "Network In (KB/s)",
    "net_out": "Network Out (KB/s)",
    "disk_io_percent": "Disk I/O Throughput"
}

BATCH_SIZE = 128


# ============================================================
# Main Explanation Pipeline
# ============================================================

def main():
    start_time = time.time()
    print("=" * 78)
    print("BITBRAINS LSTM ANOMALY FEATURE ATTRIBUTION & EXPLANATION (PHASE 10E)")
    print("=" * 78)

    # 1. Verify existence of required artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained LSTM model not found at: {MODEL_PATH}")
    if not TEST_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Test sequences not found at: {TEST_SEQUENCES_PATH}")
    if not ANOMALY_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Anomaly results not found at: {ANOMALY_RESULTS_PATH}")

    # 2. Load trained LSTM Autoencoder
    print("Loading trained LSTM-Autoencoder model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # 3. Load test sequence tensor and Phase 10D anomaly results
    print("\nLoading test sequences and anomaly detection results...")
    X_test = np.load(TEST_SEQUENCES_PATH)
    anomaly_df = pd.read_csv(ANOMALY_RESULTS_PATH)

    n_test_sequences = len(X_test)
    print(f"Loaded X_test_lstm shape : {X_test.shape} (dtype: {X_test.dtype})")
    print(f"Loaded anomaly_results   : {len(anomaly_df):,} rows")

    assert len(X_test) == len(anomaly_df), (
        f"Length mismatch: {len(X_test)} sequences vs {len(anomaly_df)} anomaly results!"
    )

    # 4. Generate Autoencoder Reconstructions
    print(f"\nReconstructing {n_test_sequences:,} test sequences with LSTM-Autoencoder...")
    infer_start_time = time.time()
    X_test_reconstructed = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    infer_duration = time.time() - infer_start_time
    print(f"Inference completed in: {infer_duration:.2f} seconds ({n_test_sequences / infer_duration:.1f} seq/sec)")

    # 5. Compute Squared Reconstruction Error Separately for Each Feature & Time Step
    # error(t, j) = (x(t, j) - reconstruction(t, j))^2
    print("\nComputing per-feature squared reconstruction errors across 12 time steps...")
    squared_errors = np.square(X_test - X_test_reconstructed)  # Shape: (N, 12, 5)

    # Aggregate across the 12 time steps using the mean for each feature
    feature_mse = np.mean(squared_errors, axis=1)  # Shape: (N, 5)

    # Total sequence reconstruction error (mean across 12 time steps and 5 features)
    total_reconstruction_mse = np.mean(feature_mse, axis=1)  # Shape: (N,)

    # Total sum of feature MSEs (equals 5 * total_reconstruction_mse)
    sum_feature_mse = np.sum(feature_mse, axis=1, keepdims=True)  # Shape: (N, 1)

    # Feature percentage contribution to total reconstruction error:
    # contrib_j (%) = (MSE_j / sum_{k} MSE_k) * 100%
    # Handle zero division safely (though MSE > 0 for real float sequences)
    safe_sum = np.where(sum_feature_mse == 0, 1e-12, sum_feature_mse)
    feature_contributions = (feature_mse / safe_sum) * 100.0  # Shape: (N, 5)

    # 6. Verify Mathematical Invariants
    print("\nVerifying mathematical invariants...")
    # Invariant 1: Contributions must be non-negative
    min_contrib = float(np.min(feature_contributions))
    assert min_contrib >= -1e-6, f"Negative feature contribution detected: {min_contrib}"
    feature_contributions = np.clip(feature_contributions, 0.0, 100.0)

    # Invariant 2: Contributions must sum approximately to 100%
    row_sums = np.sum(feature_contributions, axis=1)
    max_sum_deviation = float(np.max(np.abs(row_sums - 100.0)))
    assert max_sum_deviation < 1e-3, f"Contribution sum deviation exceeds tolerance: {max_sum_deviation}"

    # Invariant 3: Consistency with Phase 10D reconstruction errors
    error_diff = np.max(np.abs(total_reconstruction_mse - anomaly_df["reconstruction_error"].values))
    assert error_diff < 1e-4, f"Reconstruction error discrepancy with Phase 10D: {error_diff}"

    print("  [PASSED] All feature contributions are non-negative (min >= 0.0%)")
    print(f"  [PASSED] Contribution percentages sum strictly to 100% (max deviation: {max_sum_deviation:.6f}%)")
    print(f"  [PASSED] Sequence MSE aligns exactly with Phase 10D results (max diff: {error_diff:.8f})")

    # 7. Identify Top Contributing Feature for Every Sequence
    top_feature_indices = np.argmax(feature_contributions, axis=1)
    top_feature_names = [FEATURE_NAMES[idx] for idx in top_feature_indices]

    # 8. Filter to Anomalous Sequences Only
    anomaly_mask = (anomaly_df["anomaly"].values == 1)
    n_anomalies = int(np.sum(anomaly_mask))
    print(f"\nFiltering detailed explanations to anomalous sequences: {n_anomalies:,} / {n_test_sequences:,} ({n_anomalies/n_test_sequences*100:.2f}%)")

    # Assemble Detailed Explanations DataFrame
    explanations_df = pd.DataFrame({
        "sequence_index": anomaly_df.loc[anomaly_mask, "sequence_index"].values,
        "vm_id": anomaly_df.loc[anomaly_mask, "vm_id"].values,
        "start_timestamp": anomaly_df.loc[anomaly_mask, "start_timestamp"].values,
        "end_timestamp": anomaly_df.loc[anomaly_mask, "end_timestamp"].values,
        "reconstruction_error": total_reconstruction_mse[anomaly_mask],
        "cpu_contribution": feature_contributions[anomaly_mask, 0],
        "mem_contribution": feature_contributions[anomaly_mask, 1],
        "net_in_contribution": feature_contributions[anomaly_mask, 2],
        "net_out_contribution": feature_contributions[anomaly_mask, 3],
        "disk_io_contribution": feature_contributions[anomaly_mask, 4],
        "top_contributing_feature": [top_feature_names[i] for i, is_anom in enumerate(anomaly_mask) if is_anom]
    })

    # Save detailed explanations CSV
    explanations_df.to_csv(EXPLANATIONS_OUTPUT_PATH, index=False)
    print(f"Saved detailed anomaly explanations to: {EXPLANATIONS_OUTPUT_PATH}")

    # 9. Compute Explanation Summary Statistics Among Anomalies
    print("\n" + "=" * 78)
    print("ANOMALY FEATURE ATTRIBUTION SUMMARY (ANOMALOUS SEQUENCES ONLY)")
    print("=" * 78)

    top_feature_counts = explanations_df["top_contributing_feature"].value_counts()
    most_frequent_top_feature = top_feature_counts.index[0]
    most_frequent_top_count = int(top_feature_counts.iloc[0])
    most_frequent_top_pct = (most_frequent_top_count / n_anomalies) * 100.0

    print(f"Total Anomalous Sequences Evaluated : {n_anomalies:,}")
    print(f"Most Frequently Top-Contributing   : {most_frequent_top_feature} ({most_frequent_top_count:,} anomalies, {most_frequent_top_pct:.2f}%)")
    print("\nTop Contributor Breakdown by Feature:")

    top_contributor_stats = []
    for feat in FEATURE_NAMES:
        count = int(top_feature_counts.get(feat, 0))
        pct = (count / n_anomalies) * 100.0 if n_anomalies > 0 else 0.0
        top_contributor_stats.append((feat, count, pct))
        print(f"  - {feat:18s}: {count:4d} anomalies ({pct:6.2f}%)")

    # Overall Average Feature Contribution Among Anomalous Sequences
    avg_contributions = {
        "cpu_util_percent": float(explanations_df["cpu_contribution"].mean()),
        "mem_util_percent": float(explanations_df["mem_contribution"].mean()),
        "net_in": float(explanations_df["net_in_contribution"].mean()),
        "net_out": float(explanations_df["net_out_contribution"].mean()),
        "disk_io_percent": float(explanations_df["disk_io_contribution"].mean())
    }

    print("\nOverall Average Feature Contribution Among Anomalies:")
    for feat in FEATURE_NAMES:
        avg_pct = avg_contributions[feat]
        print(f"  - {feat:18s}: {avg_pct:6.2f}% average contribution")

    # Save summary CSV
    summary_records = [
        {"metric": "total_anomalous_sequences", "value": n_anomalies},
        {"metric": "most_frequently_top_contributing_feature", "value": most_frequent_top_feature},
        {"metric": "most_frequent_top_feature_count", "value": most_frequent_top_count},
        {"metric": "most_frequent_top_feature_percent", "value": f"{most_frequent_top_pct:.2f}"},
    ]

    for feat, count, pct in top_contributor_stats:
        summary_records.append({"metric": f"{feat}_top_contributor_count", "value": count})
        summary_records.append({"metric": f"{feat}_top_contributor_percent", "value": f"{pct:.2f}"})

    for feat, avg_pct in avg_contributions.items():
        summary_records.append({"metric": f"{feat}_average_contribution_percent", "value": f"{avg_pct:.2f}"})

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"\nSaved explanation summary to: {SUMMARY_OUTPUT_PATH}")

    # 10. Breakdown by Selected VM
    print("\nTop Contributing Feature by VM (Anomalous Sequences):")
    vm_breakdown = explanations_df.groupby(["vm_id", "top_contributing_feature"]).size().unstack(fill_value=0)
    print(vm_breakdown.to_string())

    # 11. Create Publication-Quality Feature Contribution Bar Chart
    print(f"\nGenerating feature contribution bar chart to: {BAR_CHART_PATH}...")
    fig, ax = plt.subplots(figsize=(10, 6))

    features_ordered = FEATURE_NAMES
    labels_ordered = [FEATURE_DISPLAY_LABELS[f] for f in features_ordered]
    contributions_ordered = [avg_contributions[f] for f in features_ordered]
    top_pcts_ordered = [dict([(f, p) for f, _, p in top_contributor_stats])[f] for f in features_ordered]

    # Harmonious corporate color palette
    colors = ["#2b5c8f", "#3a86c8", "#5c9ecc", "#d97736", "#c0392b"]

    bars = ax.bar(labels_ordered, contributions_ordered, color=colors, edgecolor="#1f2937", linewidth=1.2, width=0.55)

    # Add data labels on top of bars
    for bar, pct, top_pct in zip(bars, contributions_ordered, top_pcts_ordered):
        height = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%\n(Top: {top_pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1f2937"
        )

    ax.set_ylim(0, max(contributions_ordered) * 1.25)
    ax.set_ylabel("Average Contribution to Reconstruction Error (%)", fontsize=11, fontweight="semibold")
    # Title with cleanly spaced subtitle
    ax.set_title(
        "Bitbrains LSTM-Autoencoder: Feature Attribution Among Anomalous Sequences\n"
        f"Evaluated on {n_anomalies:,} anomalous test sequences (95th percentile threshold = 0.297136)",
        fontsize=12,
        fontweight="bold",
        pad=16,
        linespacing=1.4
    )

    # Operational disclaimer footer
    fig.text(
        0.5, 0.01,
        "Operational Note: Feature contribution derived from temporal squared reconstruction error residuals.\n"
        "Does not represent system failure probability, time-to-failure, or causal root cause.",
        ha="center", va="bottom", fontsize=8.5, color="#6b7280", style="italic"
    )

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(BAR_CHART_PATH, dpi=300)
    plt.close()
    print(f"Saved feature contribution bar chart to: {BAR_CHART_PATH}")

    # 12. Final Report
    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print("PHASE 10E EXECUTION COMPLETE")
    print("=" * 78)
    print("Generated Artifacts:")
    print(f"  1. Detailed Explanations : {EXPLANATIONS_OUTPUT_PATH} ({len(explanations_df):,} rows)")
    print(f"  2. Summary Statistics    : {SUMMARY_OUTPUT_PATH} ({len(summary_df)} metrics)")
    print(f"  3. Visual Bar Chart      : {BAR_CHART_PATH}")
    print(f"Total Execution Runtime    : {elapsed:.2f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
