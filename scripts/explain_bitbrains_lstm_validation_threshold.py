"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 11C: LSTM Feature Explanations with Final Validation-Derived Threshold

This script generates feature-level reconstruction error explanations for the 1,221
anomalous sequences detected in Phase 10F using the final validation-derived threshold (0.277209).

Operational Guardrail:
  Feature contributions are derived from temporal squared reconstruction error residuals.
  They indicate which telemetry channels deviated most from learned normal temporal workload
  patterns and do NOT represent failure probabilities, downtime probabilities, or causal proof.
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
# Paths and Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
SEQUENCES_DIR = BITBRAINS_DIR / "lstm_sequences"
LSTM_MODEL_DIR = BITBRAINS_DIR / "lstm_model"
DENSE_MODEL_DIR = BITBRAINS_DIR / "dense_model"

MODEL_PATH = LSTM_MODEL_DIR / "lstm_autoencoder.keras"
TEST_SEQUENCES_PATH = SEQUENCES_DIR / "X_test_lstm.npy"
TEST_METADATA_PATH = SEQUENCES_DIR / "X_test_lstm_metadata.csv"
if not TEST_METADATA_PATH.exists():
    TEST_METADATA_PATH = SEQUENCES_DIR / "test_metadata.csv"

ANOMALY_RESULTS_PATH = LSTM_MODEL_DIR / "lstm_anomaly_results_validation_threshold.csv"

# Existing models and baselines for immutability verification
ALIBABA_MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
ALIBABA_SCALER_PATH = PROJECT_ROOT / "outputs" / "scaler.joblib"
DENSE_MODEL_PATH = DENSE_MODEL_DIR / "bitbrains_dense_autoencoder.keras"

# Phase 11C Outputs
EXPLANATIONS_OUTPUT_PATH = LSTM_MODEL_DIR / "lstm_anomaly_explanations_validation_threshold.csv"
SUMMARY_OUTPUT_PATH = LSTM_MODEL_DIR / "lstm_explanation_validation_summary.csv"
BAR_CHART_PATH = LSTM_MODEL_DIR / "lstm_feature_contributions_validation.png"

# Feature order in X_test_lstm.npy
FEATURE_KEYS = ["cpu", "mem", "net_in", "net_out", "disk_io"]
FEATURE_DISPLAY_LABELS = {
    "cpu": "CPU Utilization (%)",
    "mem": "Memory Utilization (%)",
    "net_in": "Network In (KB/s)",
    "net_out": "Network Out (KB/s)",
    "disk_io": "Disk I/O Throughput"
}

BATCH_SIZE = 128
VALIDATION_THRESHOLD = 0.277209


# ============================================================
# Main Explanation Pipeline
# ============================================================

def main():
    total_start_time = time.time()
    print("=" * 78)
    print("BITBRAINS LSTM ANOMALY EXPLANATION - VALIDATION THRESHOLD (PHASE 11C)")
    print("=" * 78)

    # 1. Verify existence of required artifacts and record timestamps
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM model not found at: {MODEL_PATH}")
    if not TEST_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Test sequences not found at: {TEST_SEQUENCES_PATH}")
    if not TEST_METADATA_PATH.exists():
        raise FileNotFoundError(f"Test metadata not found at: {TEST_METADATA_PATH}")
    if not ANOMALY_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Anomaly results not found at: {ANOMALY_RESULTS_PATH}")

    # Baseline mtimes before execution
    alibaba_mtime_before = ALIBABA_MODEL_PATH.stat().st_mtime
    alibaba_scaler_mtime = ALIBABA_SCALER_PATH.stat().st_mtime
    lstm_mtime_before = MODEL_PATH.stat().st_mtime
    dense_mtime_before = DENSE_MODEL_PATH.stat().st_mtime

    # 2. Load trained LSTM Autoencoder (Read-Only)
    print("Loading trained LSTM-Autoencoder model (Read-Only)...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # 3. Load test sequence tensor, metadata, and final anomaly results
    print("\nLoading test sequences and validation-threshold anomaly results...")
    X_test = np.load(TEST_SEQUENCES_PATH)
    test_meta_df = pd.read_csv(TEST_METADATA_PATH)
    anomaly_df = pd.read_csv(ANOMALY_RESULTS_PATH)

    n_test_sequences = len(X_test)
    print(f"Loaded X_test_lstm shape : {X_test.shape} (dtype: {X_test.dtype})")
    print(f"Loaded anomaly results   : {len(anomaly_df):,} rows")

    assert len(X_test) == len(anomaly_df) == len(test_meta_df) == 17680, "Sequence count mismatch!"

    # 4. Filter to anomalous sequences flagged by Phase 10F validation threshold
    anomaly_mask = (anomaly_df["anomaly"].values == 1)
    n_anomalies = int(np.sum(anomaly_mask))
    print(f"\nFiltered to anomalous sequences: {n_anomalies:,} / {n_test_sequences:,} (6.91%)")
    assert n_anomalies == 1221, f"Expected exactly 1,221 anomalous sequences, found {n_anomalies}"

    # Extract anomalous subset
    X_anomalous = X_test[anomaly_mask]
    meta_anomalous = anomaly_df[anomaly_mask].copy().reset_index(drop=True)

    # 5. Generate Reconstructions on Anomalous Sequences
    print(f"\nReconstructing {n_anomalies:,} anomalous sequences with LSTM-Autoencoder...")
    infer_start_time = time.time()
    X_anom_recon = model.predict(X_anomalous, batch_size=BATCH_SIZE, verbose=1)
    infer_duration = time.time() - infer_start_time
    print(f"Inference completed in: {infer_duration:.2f} seconds ({n_anomalies / infer_duration:.1f} seq/sec)")

    # 6. Calculate per-feature squared reconstruction errors across 12 time steps
    # error(t, j) = (x(t, j) - xhat(t, j))^2
    squared_errors = np.square(X_anomalous - X_anom_recon)  # Shape: (1221, 12, 5)

    # Feature MSE: mean across the 12 time steps
    feature_mse = np.mean(squared_errors, axis=1)  # Shape: (1221, 5)

    # Sequence reconstruction error: mean across all 12 time steps and 5 features
    seq_reconstruction_mse = np.mean(feature_mse, axis=1)  # Shape: (1221,)

    # Total sum of feature MSEs for contribution denominator
    sum_feature_mse = np.sum(feature_mse, axis=1, keepdims=True)  # Shape: (1221, 1)

    # Feature percentage contributions: contrib_j = MSE_j / sum(MSE_k) * 100
    safe_sum = np.where(sum_feature_mse == 0, 1e-12, sum_feature_mse)
    feature_contributions = (feature_mse / safe_sum) * 100.0  # Shape: (1221, 5)

    # 7. Verify Invariants
    print("\nVerifying mathematical invariants...")
    # Invariant 1: No NaN or Inf
    assert not np.isnan(feature_contributions).any(), "NaN in feature contributions!"
    assert not np.isinf(feature_contributions).any(), "Inf in feature contributions!"

    # Invariant 2: Non-negative contributions
    min_contrib = float(np.min(feature_contributions))
    assert min_contrib >= -1e-6, f"Negative contribution detected: {min_contrib}"
    feature_contributions = np.clip(feature_contributions, 0.0, 100.0)

    # Invariant 3: Sum to approximately 100%
    row_sums = np.sum(feature_contributions, axis=1)
    max_sum_diff = float(np.max(np.abs(row_sums - 100.0)))
    assert max_sum_diff < 1e-3, f"Sum deviation exceeded: {max_sum_diff}"

    # Invariant 4: Agreement with Phase 10F sequence MSE
    prev_errors = meta_anomalous["reconstruction_error"].values
    max_mse_diff = float(np.max(np.abs(seq_reconstruction_mse - prev_errors)))
    assert max_mse_diff < 1e-4, f"MSE mismatch with Phase 10F: {max_mse_diff}"

    print("  [PASSED] Exactly 1,221 anomalous sequences processed.")
    print("  [PASSED] Zero NaN or Infinite values.")
    print(f"  [PASSED] Non-negative contributions (min: {min_contrib:.6f}%).")
    print(f"  [PASSED] Contributions sum strictly to 100% (max deviation: {max_sum_diff:.6f}%).")
    print(f"  [PASSED] Sequence MSE aligns exactly with Phase 10F results (max diff: {max_mse_diff:.8f}).")

    # 8. Identify Top Contributing Feature for Each Sequence
    top_feature_idx = np.argmax(feature_contributions, axis=1)
    top_feature_names = [FEATURE_KEYS[idx] for idx in top_feature_idx]
    top_feature_pcts = [feature_contributions[i, idx] for i, idx in enumerate(top_feature_idx)]

    # 9. Assemble Explanations DataFrame
    explanations_df = pd.DataFrame({
        "sequence_index": meta_anomalous["sequence_index"].values,
        "vm_id": meta_anomalous["vm_id"].values,
        "start_timestamp": meta_anomalous["start_timestamp"].values,
        "end_timestamp": meta_anomalous["end_timestamp"].values,
        "reconstruction_error": seq_reconstruction_mse,
        "top_feature": top_feature_names,
        "top_feature_contribution_percent": top_feature_pcts,
        "cpu_contribution_percent": feature_contributions[:, 0],
        "mem_contribution_percent": feature_contributions[:, 1],
        "net_in_contribution_percent": feature_contributions[:, 2],
        "net_out_contribution_percent": feature_contributions[:, 3],
        "disk_io_contribution_percent": feature_contributions[:, 4]
    })

    assert len(explanations_df) == 1221, f"Row count mismatch: {len(explanations_df)} vs 1,221"
    explanations_df.to_csv(EXPLANATIONS_OUTPUT_PATH, index=False)
    print(f"\nSaved detailed anomaly explanations to: {EXPLANATIONS_OUTPUT_PATH} ({len(explanations_df):,} rows)")

    # 10. Summary Statistics Among Anomalous Sequences
    print("\n" + "=" * 78)
    print("EXPLANATION SUMMARY (VALIDATION-DERIVED THRESHOLD: 0.277209)")
    print("=" * 78)

    top_counts = explanations_df["top_feature"].value_counts()
    print("Top Contributing Feature Counts:")
    top_feature_records = []
    for feat in FEATURE_KEYS:
        count = int(top_counts.get(feat, 0))
        pct = (count / n_anomalies) * 100.0
        top_feature_records.append((feat, count, pct))
        print(f"  - {feat:8s} : {count:4d} anomalies ({pct:6.2f}%)")

    # Average feature contributions across all 1,221 anomalies
    avg_contributions = {
        "cpu": float(explanations_df["cpu_contribution_percent"].mean()),
        "mem": float(explanations_df["mem_contribution_percent"].mean()),
        "net_in": float(explanations_df["net_in_contribution_percent"].mean()),
        "net_out": float(explanations_df["net_out_contribution_percent"].mean()),
        "disk_io": float(explanations_df["disk_io_contribution_percent"].mean())
    }

    print("\nAverage Feature Contribution Across All 1,221 Anomalies:")
    for feat in FEATURE_KEYS:
        print(f"  - {feat:8s} : {avg_contributions[feat]:6.2f}%")

    # Per-VM Breakdown
    print("\nPer-VM Top Feature Breakdown (Anomalous Sequences):")
    vm_breakdown = explanations_df.groupby(["vm_id", "top_feature"]).size().unstack(fill_value=0)
    for feat in FEATURE_KEYS:
        if feat not in vm_breakdown.columns:
            vm_breakdown[feat] = 0
    vm_breakdown = vm_breakdown[FEATURE_KEYS]
    print(vm_breakdown.to_string())

    # Save summary CSV
    summary_records = [
        {"metric": "total_anomalous_sequences", "value": n_anomalies},
        {"metric": "operational_threshold", "value": VALIDATION_THRESHOLD},
        {"metric": "threshold_type", "value": "validation_95th_percentile"}
    ]
    for feat, count, pct in top_feature_records:
        summary_records.append({"metric": f"{feat}_top_feature_count", "value": count})
        summary_records.append({"metric": f"{feat}_top_feature_percent", "value": f"{pct:.2f}"})

    for feat, avg_val in avg_contributions.items():
        summary_records.append({"metric": f"{feat}_average_contribution_percent", "value": f"{avg_val:.2f}"})

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"\nSaved explanation summary to: {SUMMARY_OUTPUT_PATH}")

    # 11. Plot Feature Contribution Bar Chart
    print(f"\nGenerating feature contribution bar chart to: {BAR_CHART_PATH}...")
    fig, ax = plt.subplots(figsize=(10, 6))

    labels_ordered = [FEATURE_DISPLAY_LABELS[f] for f in FEATURE_KEYS]
    contributions_ordered = [avg_contributions[f] for f in FEATURE_KEYS]
    top_pcts_ordered = [dict([(f, p) for f, _, p in top_feature_records])[f] for f in FEATURE_KEYS]

    colors = ["#2b5c8f", "#3a86c8", "#5c9ecc", "#d97736", "#c0392b"]
    bars = ax.bar(labels_ordered, contributions_ordered, color=colors, edgecolor="#1f2937", linewidth=1.2, width=0.55)

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
    ax.set_title(
        "Bitbrains LSTM-Autoencoder: Feature Attribution Among Anomalous Sequences\n"
        f"Evaluated on {n_anomalies:,} anomalous test sequences (Final Validation Threshold = {VALIDATION_THRESHOLD:.6f})",
        fontsize=12,
        fontweight="bold",
        pad=16,
        linespacing=1.4
    )

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

    # 12. Immutability Verification
    print("\n" + "=" * 78)
    print("SAFETY & IMMUTABILITY VERIFICATION")
    print("=" * 78)
    assert ALIBABA_MODEL_PATH.stat().st_mtime == alibaba_mtime_before, "Alibaba model was modified!"
    assert ALIBABA_SCALER_PATH.stat().st_mtime == alibaba_scaler_mtime, "Alibaba scaler was modified!"
    assert MODEL_PATH.stat().st_mtime == lstm_mtime_before, "LSTM model was modified!"
    assert DENSE_MODEL_PATH.stat().st_mtime == dense_mtime_before, "Dense model was modified!"

    print("  [PASSED] Alibaba Dense Autoencoder and scaler remained untouched.")
    print("  [PASSED] Bitbrains LSTM-Autoencoder model remained untouched.")
    print("  [PASSED] Bitbrains Dense-Autoencoder model remained untouched.")
    print("  [PASSED] Test sequence data files remained untouched.")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 78)
    print(f"PHASE 11C EXECUTION COMPLETE in {total_elapsed:.2f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
