"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 11B: Dense Autoencoder Anomaly Detection with Validation-Derived Threshold

This script performs anomaly detection using the trained Bitbrains Dense Autoencoder
(outputs/bitbrains/dense_model/bitbrains_dense_autoencoder.keras).

Key Methodological Guardrails:
  1. Threshold is derived STRICTLY from the 14,212 validation sequences (95th percentile).
  2. Test data (17,680 sequences) is NOT used to calculate the anomaly threshold (zero leakage).
  3. Sequences are flattened from (12, 5) to (60,) before feedforward processing.
  4. The threshold is an empirical operational anomaly indicator, NOT a failure probability.
  5. Immutability: Alibaba baseline, LSTM model, and prior Phase 10/11A artifacts remain untouched.
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
DENSE_MODEL_DIR = BITBRAINS_DIR / "dense_model"

DENSE_MODEL_PATH = DENSE_MODEL_DIR / "bitbrains_dense_autoencoder.keras"
TRAIN_SEQUENCES_PATH = SEQUENCES_DIR / "X_train_lstm.npy"
TRAIN_METADATA_PATH = SEQUENCES_DIR / "train_metadata.csv"
TEST_SEQUENCES_PATH = SEQUENCES_DIR / "X_test_lstm.npy"
TEST_METADATA_PATH = SEQUENCES_DIR / "test_metadata.csv"

# Existing Baseline / Prior Artifacts for Immutability Checking
ALIBABA_MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
LSTM_MODEL_PATH = BITBRAINS_DIR / "lstm_model" / "lstm_autoencoder.keras"
PHASE_10D_RESULTS_PATH = BITBRAINS_DIR / "lstm_model" / "lstm_anomaly_results.csv"
PHASE_10F_RESULTS_PATH = BITBRAINS_DIR / "lstm_model" / "lstm_anomaly_results_validation_threshold.csv"

# Phase 11B Outputs
VAL_SUMMARY_OUTPUT_PATH = DENSE_MODEL_DIR / "dense_validation_threshold_summary.csv"
TEST_RESULTS_OUTPUT_PATH = DENSE_MODEL_DIR / "dense_anomaly_results_validation_threshold.csv"
VAL_DIST_PLOT_PATH = DENSE_MODEL_DIR / "dense_validation_threshold.png"

BATCH_SIZE = 128
TRAIN_VAL_SPLIT_RATIO = 0.80
INPUT_DIM = 60


# ============================================================
# Replicate Exact Phase 10C/11A Validation Split
# ============================================================

def create_vm_aware_train_val_split(train_meta_df: pd.DataFrame, split_ratio: float = 0.80):
    """
    Replicate the exact VM-aware chronological split used in Phase 10C and 11A:
    Takes first 80% sequences per VM for training, final 20% for validation.
    """
    train_indices = []
    val_indices = []

    for vm_id, group in train_meta_df.groupby("vm_id", sort=False):
        indices = group["sequence_index"].values
        n_seqs = len(indices)
        n_train_sub = int(n_seqs * split_ratio)

        train_indices.extend(indices[:n_train_sub])
        val_indices.extend(indices[n_train_sub:])

    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


# ============================================================
# Main Detection Pipeline
# ============================================================

def main():
    total_start_time = time.time()
    print("=" * 78)
    print("BITBRAINS DENSE-AE ANOMALY DETECTION (PHASE 11B)")
    print("=" * 78)

    # 1. Verify existence of required artifacts and record mtimes
    if not DENSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Dense model not found at: {DENSE_MODEL_PATH}")
    if not TRAIN_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Train sequences not found at: {TRAIN_SEQUENCES_PATH}")
    if not TRAIN_METADATA_PATH.exists():
        raise FileNotFoundError(f"Train metadata not found at: {TRAIN_METADATA_PATH}")
    if not TEST_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Test sequences not found at: {TEST_SEQUENCES_PATH}")
    if not TEST_METADATA_PATH.exists():
        raise FileNotFoundError(f"Test metadata not found at: {TEST_METADATA_PATH}")

    dense_mtime_before = DENSE_MODEL_PATH.stat().st_mtime
    dense_size_before = DENSE_MODEL_PATH.stat().st_size
    alibaba_mtime_before = ALIBABA_MODEL_PATH.stat().st_mtime
    lstm_mtime_before = LSTM_MODEL_PATH.stat().st_mtime
    p10d_mtime_before = PHASE_10D_RESULTS_PATH.stat().st_mtime
    p10f_mtime_before = PHASE_10F_RESULTS_PATH.stat().st_mtime

    # 2. Load trained Dense Autoencoder (Read-Only)
    print("Loading trained Bitbrains Dense Autoencoder (Read-Only)...")
    model = tf.keras.models.load_model(DENSE_MODEL_PATH)
    print(f"Loaded model from: {DENSE_MODEL_PATH}")
    assert model.input_shape == (None, INPUT_DIM), f"Unexpected input shape: {model.input_shape}"

    # 3. Load training sequence data and reconstruct Phase 10C/11A validation partition
    print("\nLoading training sequences and reconstructing validation partition...")
    X_train_full = np.load(TRAIN_SEQUENCES_PATH)
    train_meta_df = pd.read_csv(TRAIN_METADATA_PATH)

    train_indices, val_indices = create_vm_aware_train_val_split(train_meta_df, split_ratio=TRAIN_VAL_SPLIT_RATIO)
    X_val_seqs = X_train_full[val_indices]

    n_val_sequences = len(X_val_seqs)
    print(f"Reconstructed validation sequence count: {n_val_sequences:,} (Expected: 14,212)")
    assert n_val_sequences == 14212, f"Validation count mismatch: {n_val_sequences} vs 14,212"

    # Flatten validation sequences to 60 dimensions: (14212, 12, 5) -> (14212, 60)
    X_val_flat = X_val_seqs.reshape(n_val_sequences, INPUT_DIM)
    print(f"Flattened validation tensor shape: {X_val_flat.shape}")

    # 4. Infer Reconstructions on Validation Set
    print(f"\nGenerating autoencoder reconstructions for {n_val_sequences:,} validation sequences...")
    val_infer_start = time.time()
    X_val_recon = model.predict(X_val_flat, batch_size=BATCH_SIZE, verbose=1)
    val_infer_duration = time.time() - val_infer_start
    print(f"Validation inference completed in: {val_infer_duration:.2f} seconds ({n_val_sequences / val_infer_duration:.1f} seq/sec)")

    # 5. Compute Validation Reconstruction Errors (MSE across all 60 flattened values)
    val_squared_diff = np.square(X_val_flat - X_val_recon)
    val_reconstruction_errors = np.mean(val_squared_diff, axis=1).astype(np.float64)

    assert len(val_reconstruction_errors) == n_val_sequences
    assert not np.isnan(val_reconstruction_errors).any(), "NaN found in validation reconstruction errors"
    assert not np.isinf(val_reconstruction_errors).any(), "Inf found in validation reconstruction errors"

    # 6. Calculate Validation Distribution Statistics
    val_count = len(val_reconstruction_errors)
    val_mean = float(np.mean(val_reconstruction_errors))
    val_median = float(np.median(val_reconstruction_errors))
    val_std = float(np.std(val_reconstruction_errors))
    val_min = float(np.min(val_reconstruction_errors))
    val_max = float(np.max(val_reconstruction_errors))
    val_p90 = float(np.percentile(val_reconstruction_errors, 90))
    val_p95 = float(np.percentile(val_reconstruction_errors, 95))
    val_p99 = float(np.percentile(val_reconstruction_errors, 99))

    # Selected Threshold: 95th Percentile of Validation Errors
    selected_threshold = val_p95

    print("\n" + "=" * 78)
    print("DENSE-AE VALIDATION RECONSTRUCTION ERROR DISTRIBUTION STATISTICS")
    print("=" * 78)
    print(f"  Validation Sequences : {val_count:,}")
    print(f"  Mean                 : {val_mean:.6f}")
    print(f"  Median (50th %)      : {val_median:.6f}")
    print(f"  Standard Deviation   : {val_std:.6f}")
    print(f"  Minimum              : {val_min:.6f}")
    print(f"  Maximum              : {val_max:.6f}")
    print(f"  90th Percentile      : {val_p90:.6f}")
    print(f"  95th Percentile (TH) : {val_p95:.6f}  <-- PRIMARY OPERATIONAL THRESHOLD")
    print(f"  99th Percentile      : {val_p99:.6f}")
    print("=" * 78)

    # 7. Load Untouched Test Sequences
    print("\nLoading untouched test sequences (17,680 sequences)...")
    X_test_seqs = np.load(TEST_SEQUENCES_PATH)
    test_meta_df = pd.read_csv(TEST_METADATA_PATH)

    n_test_sequences = len(X_test_seqs)
    assert n_test_sequences == 17680, f"Test count mismatch: {n_test_sequences} vs 17,680"
    assert len(test_meta_df) == 17680, f"Metadata count mismatch: {len(test_meta_df)} vs 17,680"

    # Flatten test sequences to 60 dimensions: (17680, 12, 5) -> (17680, 60)
    X_test_flat = X_test_seqs.reshape(n_test_sequences, INPUT_DIM)
    print(f"Flattened test tensor shape: {X_test_flat.shape}")

    # 8. Infer Reconstructions on Test Set
    print(f"Generating reconstructions for {n_test_sequences:,} test sequences...")
    test_infer_start = time.time()
    X_test_recon = model.predict(X_test_flat, batch_size=BATCH_SIZE, verbose=1)
    test_infer_duration = time.time() - test_infer_start
    print(f"Test inference completed in: {test_infer_duration:.2f} seconds ({n_test_sequences / test_infer_duration:.1f} seq/sec)")

    # 9. Compute Test Reconstruction Errors across all 60 values
    test_squared_diff = np.square(X_test_flat - X_test_recon)
    test_reconstruction_errors = np.mean(test_squared_diff, axis=1).astype(np.float64)

    assert not np.isnan(test_reconstruction_errors).any(), "NaN found in test reconstruction errors"
    assert not np.isinf(test_reconstruction_errors).any(), "Inf found in test reconstruction errors"

    # Test Distribution Statistics
    test_count = len(test_reconstruction_errors)
    test_mean = float(np.mean(test_reconstruction_errors))
    test_median = float(np.median(test_reconstruction_errors))
    test_std = float(np.std(test_reconstruction_errors))
    test_min = float(np.min(test_reconstruction_errors))
    test_max = float(np.max(test_reconstruction_errors))
    test_p90 = float(np.percentile(test_reconstruction_errors, 90))
    test_p95 = float(np.percentile(test_reconstruction_errors, 95))
    test_p99 = float(np.percentile(test_reconstruction_errors, 99))

    print("\n" + "=" * 78)
    print("DENSE-AE TEST RECONSTRUCTION ERROR DISTRIBUTION STATISTICS")
    print("=" * 78)
    print(f"  Test Sequences       : {test_count:,}")
    print(f"  Mean                 : {test_mean:.6f}")
    print(f"  Median (50th %)      : {test_median:.6f}")
    print(f"  Standard Deviation   : {test_std:.6f}")
    print(f"  Minimum              : {test_min:.6f}")
    print(f"  Maximum              : {test_max:.6f}")
    print(f"  90th Percentile      : {test_p90:.6f}")
    print(f"  95th Percentile      : {test_p95:.6f}")
    print(f"  99th Percentile      : {test_p99:.6f}")
    print("=" * 78)

    # 10. Apply Fixed Validation Threshold to Test Sequences
    anomaly_flags = (test_reconstruction_errors > selected_threshold).astype(int)

    n_anomalies = int(np.sum(anomaly_flags == 1))
    n_normal = int(np.sum(anomaly_flags == 0))
    pct_anomalies = (n_anomalies / n_test_sequences) * 100.0
    pct_normal = (n_normal / n_test_sequences) * 100.0

    print("\n" + "=" * 78)
    print("DENSE-AE TEST CLASSIFICATION RESULTS (VALIDATION THRESHOLD)")
    print("=" * 78)
    print(f"  Fixed Validation Threshold (95th %) : {selected_threshold:.6f}")
    print(f"  Total Test Sequences Evaluated     : {n_test_sequences:,} (100.00%)")
    print(f"  Normal Sequences                   : {n_normal:,} ({pct_normal:.2f}%)")
    print(f"  Anomalous Sequences                : {n_anomalies:,} ({pct_anomalies:.2f}%)")
    print("  Zero Test Leakage Verified         : Threshold calculated strictly on validation set.")
    print("=" * 78)

    # 11. Save Validation Threshold Summary CSV
    summary_data = [
        {"metric": "validation_count", "value": val_count},
        {"metric": "validation_mean", "value": f"{val_mean:.6f}"},
        {"metric": "validation_median", "value": f"{val_median:.6f}"},
        {"metric": "validation_standard_deviation", "value": f"{val_std:.6f}"},
        {"metric": "validation_minimum", "value": f"{val_min:.6f}"},
        {"metric": "validation_maximum", "value": f"{val_max:.6f}"},
        {"metric": "validation_90th_percentile", "value": f"{val_p90:.6f}"},
        {"metric": "validation_95th_percentile", "value": f"{val_p95:.6f}"},
        {"metric": "validation_99th_percentile", "value": f"{val_p99:.6f}"},
        {"metric": "selected_threshold", "value": f"{selected_threshold:.6f}"},
        {"metric": "threshold_methodology", "value": "95th_percentile_validation_reconstruction_error"},
        {"metric": "test_count", "value": test_count},
        {"metric": "test_mean", "value": f"{test_mean:.6f}"},
        {"metric": "test_median", "value": f"{test_median:.6f}"},
        {"metric": "test_standard_deviation", "value": f"{test_std:.6f}"},
        {"metric": "test_minimum", "value": f"{test_min:.6f}"},
        {"metric": "test_maximum", "value": f"{test_max:.6f}"},
        {"metric": "test_90th_percentile", "value": f"{test_p90:.6f}"},
        {"metric": "test_95th_percentile", "value": f"{test_p95:.6f}"},
        {"metric": "test_99th_percentile", "value": f"{test_p99:.6f}"},
        {"metric": "test_normal_count", "value": n_normal},
        {"metric": "test_normal_percent", "value": f"{pct_normal:.2f}"},
        {"metric": "test_anomaly_count", "value": n_anomalies},
        {"metric": "test_anomaly_percent", "value": f"{pct_anomalies:.2f}"}
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(VAL_SUMMARY_OUTPUT_PATH, index=False)
    print(f"\nSaved validation threshold summary to: {VAL_SUMMARY_OUTPUT_PATH}")

    # 12. Save Test Anomaly Results CSV
    results_df = pd.DataFrame({
        "sequence_index": test_meta_df["sequence_index"].values,
        "vm_id": test_meta_df["vm_id"].values,
        "start_timestamp": test_meta_df["start_timestamp"].values,
        "end_timestamp": test_meta_df["end_timestamp"].values,
        "reconstruction_error": test_reconstruction_errors,
        "anomaly": anomaly_flags
    })

    assert len(results_df) == 17680, f"Row count mismatch: {len(results_df)} vs 17,680"
    results_df.to_csv(TEST_RESULTS_OUTPUT_PATH, index=False)
    print(f"Saved test anomaly results to: {TEST_RESULTS_OUTPUT_PATH} ({len(results_df):,} rows)")

    # 13. Per-VM Anomaly Breakdown
    print("\nPer-VM Anomaly Breakdown (Dense AE Test Set under Validation Threshold):")
    vm_breakdown = results_df.groupby("vm_id").agg(
        total_test_seqs=("anomaly", "count"),
        anomalies=("anomaly", "sum"),
        mean_error=("reconstruction_error", "mean"),
        max_error=("reconstruction_error", "max")
    ).reset_index()
    vm_breakdown["anomaly_rate_pct"] = (vm_breakdown["anomalies"] / vm_breakdown["total_test_seqs"]) * 100.0

    for _, r in vm_breakdown.iterrows():
        print(f"  VM {int(r['vm_id']):4d} | Total: {int(r['total_test_seqs']):,d} | Anomalies: {int(r['anomalies']):4d} ({r['anomaly_rate_pct']:5.2f}%) | Mean Error: {r['mean_error']:.4f} | Max Error: {r['max_error']:.4f}")

    # 14. Create Validation Threshold Distribution Plot
    print(f"\nGenerating validation reconstruction error distribution plot to: {VAL_DIST_PLOT_PATH}...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(
        val_reconstruction_errors,
        bins=60,
        alpha=0.75,
        color="#3a86c8",
        edgecolor="black",
        linewidth=0.5,
        density=True,
        label="Dense-AE Validation Reconstruction Error Distribution"
    )
    ax.axvline(
        selected_threshold,
        color="#d9534f",
        linestyle="--",
        linewidth=2.2,
        label=f"Selected Threshold: 95th Percentile ({selected_threshold:.4f})"
    )
    ax.axvline(
        val_p99,
        color="#8e44ad",
        linestyle=":",
        linewidth=1.8,
        label=f"99th Percentile ({val_p99:.4f})"
    )

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Density", fontsize=11, fontweight="semibold")
    ax.set_title(
        "Bitbrains Dense-Autoencoder: Validation Set Reconstruction Error Distribution\n"
        f"Validation Partition: {val_count:,} sequences (Chronological 20% split per VM | Input: 60D)",
        fontsize=12,
        fontweight="bold",
        pad=16,
        linespacing=1.4
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.text(
        0.5, 0.01,
        "Operational Note: Anomaly threshold derived strictly from validation distribution to prevent test leakage.\n"
        "Represents empirical operational anomaly deviation, NOT system failure or downtime probability.",
        ha="center", va="bottom", fontsize=8.5, color="#6b7280", style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(VAL_DIST_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved validation distribution plot to: {VAL_DIST_PLOT_PATH}")

    # 15. Safety & Immutability Verification
    print("\n" + "=" * 78)
    print("SAFETY & IMMUTABILITY VERIFICATION")
    print("=" * 78)
    assert DENSE_MODEL_PATH.stat().st_mtime == dense_mtime_before, "Dense model was modified!"
    assert DENSE_MODEL_PATH.stat().st_size == dense_size_before, "Dense model size changed!"
    assert ALIBABA_MODEL_PATH.stat().st_mtime == alibaba_mtime_before, "Alibaba model was modified!"
    assert LSTM_MODEL_PATH.stat().st_mtime == lstm_mtime_before, "Bitbrains LSTM model was modified!"
    assert PHASE_10D_RESULTS_PATH.stat().st_mtime == p10d_mtime_before, "Phase 10D results were modified!"
    assert PHASE_10F_RESULTS_PATH.stat().st_mtime == p10f_mtime_before, "Phase 10F results were modified!"

    print("  [PASSED] Dense Autoencoder model remained unchanged (strictly read-only).")
    print("  [PASSED] Bitbrains LSTM-Autoencoder remained untouched.")
    print("  [PASSED] Alibaba Dense-AE baseline remained untouched.")
    print("  [PASSED] Phase 10 outputs remained strictly preserved.")
    print("  [PASSED] Exactly 17,680 rows generated in test results (0 NaN / 0 Inf).")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 78)
    print(f"PHASE 11B EXECUTION COMPLETE in {total_elapsed:.2f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
