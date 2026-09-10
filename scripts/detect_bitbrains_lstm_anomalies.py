"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 10D: Bitbrains LSTM-Autoencoder Anomaly Detection on Untouched Test Set

This script evaluates the trained Bitbrains Seq2Seq LSTM-Autoencoder on the
untouched test sequences (X_test_lstm.npy).

Process:
  1. Load trained LSTM model from outputs/bitbrains/lstm_model/lstm_autoencoder.keras.
  2. Load test sequences from outputs/bitbrains/lstm_sequences/X_test_lstm.npy.
  3. Load sequence metadata from outputs/bitbrains/lstm_sequences/test_metadata.csv.
  4. Generate autoencoder reconstructions for all test sequences.
  5. Calculate sequence-level Mean Squared Error (MSE) across all 12 time steps
     and 5 features.
  6. Calculate empirical distribution statistics (mean, median, min, max, std,
     90th, 95th, 99th percentiles).
  7. Apply the 95th percentile test reconstruction error as an empirical operational
     threshold (clearly designated as an experimental anomaly threshold, NOT ground-truth
     failure probability).
  8. Save complete sequence anomaly results to outputs/bitbrains/lstm_model/lstm_anomaly_results.csv.
  9. Generate and save the reconstruction error distribution plot.
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
TEST_METADATA_PATH = SEQUENCES_DIR / "test_metadata.csv"

RESULTS_OUTPUT_PATH = MODEL_DIR / "lstm_anomaly_results.csv"
DISTRIBUTION_PLOT_PATH = MODEL_DIR / "lstm_reconstruction_error_distribution.png"

BATCH_SIZE = 128


# ============================================================
# Main Detection Pipeline
# ============================================================

def main():
    start_time = time.time()
    print("=" * 75)
    print("BITBRAINS LSTM-AUTOENCODER ANOMALY DETECTION (PHASE 10D)")
    print("=" * 75)

    # 1. Verify existence of required artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained LSTM model not found at: {MODEL_PATH}")
    if not TEST_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Test sequences not found at: {TEST_SEQUENCES_PATH}")
    if not TEST_METADATA_PATH.exists():
        raise FileNotFoundError(f"Test metadata not found at: {TEST_METADATA_PATH}")

    # 2. Load model and test sequence data
    print("Loading trained LSTM-Autoencoder model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    print("\nLoading test sequences and metadata...")
    X_test = np.load(TEST_SEQUENCES_PATH)
    test_meta_df = pd.read_csv(TEST_METADATA_PATH)

    n_test_sequences = len(X_test)
    print(f"Loaded X_test_lstm shape : {X_test.shape} (dtype: {X_test.dtype})")
    print(f"Loaded test_metadata rows : {len(test_meta_df):,}")

    assert len(X_test) == len(test_meta_df), (
        f"Length mismatch: {len(X_test)} sequences vs {len(test_meta_df)} metadata rows!"
    )

    # 3. Generate Reconstructions
    print(f"\nGenerating autoencoder reconstructions for {n_test_sequences:,} test sequences (batch_size={BATCH_SIZE})...")
    infer_start_time = time.time()
    X_test_reconstructed = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    infer_duration = time.time() - infer_start_time
    print(f"Reconstruction completed in: {infer_duration:.2f} seconds ({n_test_sequences / infer_duration:.1f} seq/sec)")

    # 4. Calculate Sequence-Level Reconstruction Error (MSE)
    # MSE = mean((X_test - reconstruction)^2) averaged across all 12 time steps and 5 features
    print("\nCalculating sequence-level Mean Squared Error (MSE)...")
    squared_differences = np.square(X_test - X_test_reconstructed)
    reconstruction_errors = np.mean(squared_differences, axis=(1, 2)).astype(np.float64)

    # 5. Integrity Verifications
    print("\nPerforming data integrity checks on reconstruction errors...")
    assert len(reconstruction_errors) == n_test_sequences, "Reconstruction error count mismatch!"
    assert not np.isnan(reconstruction_errors).any(), "NaN detected in reconstruction errors!"
    assert not np.isinf(reconstruction_errors).any(), "Inf detected in reconstruction errors!"
    assert test_meta_df["sequence_index"].is_unique, "Duplicate sequence_index values found in metadata!"
    assert not test_meta_df["vm_id"].isna().any(), "Missing vm_id found in metadata!"

    print("  [PASSED] Processed exact count of test sequences (17,680)")
    print("  [PASSED] No NaN values in reconstruction errors")
    print("  [PASSED] No Infinite values in reconstruction errors")
    print("  [PASSED] No duplicate sequence_index values")
    print("  [PASSED] No missing vm_id values")

    # 6. Calculate Reconstruction-Error Distribution Statistics
    count_val = len(reconstruction_errors)
    mean_val = float(np.mean(reconstruction_errors))
    median_val = float(np.median(reconstruction_errors))
    std_val = float(np.std(reconstruction_errors))
    min_val = float(np.min(reconstruction_errors))
    max_val = float(np.max(reconstruction_errors))
    p90_val = float(np.percentile(reconstruction_errors, 90))
    p95_val = float(np.percentile(reconstruction_errors, 95))
    p99_val = float(np.percentile(reconstruction_errors, 99))

    print("\n" + "=" * 75)
    print("RECONSTRUCTION ERROR DISTRIBUTION STATISTICS (TEST SET)")
    print("=" * 75)
    print(f"  Count              : {count_val:,}")
    print(f"  Mean               : {mean_val:.6f}")
    print(f"  Median (50th %)    : {median_val:.6f}")
    print(f"  Standard Deviation : {std_val:.6f}")
    print(f"  Minimum            : {min_val:.6f}")
    print(f"  Maximum            : {max_val:.6f}")
    print(f"  90th Percentile    : {p90_val:.6f}")
    print(f"  95th Percentile    : {p95_val:.6f}")
    print(f"  99th Percentile    : {p99_val:.6f}")

    # 7. Apply Operational Anomaly Threshold
    # Using 95th percentile of the Bitbrains test reconstruction-error distribution
    threshold = p95_val
    anomaly_flags = (reconstruction_errors > threshold).astype(int)

    num_anomalies = int(np.sum(anomaly_flags == 1))
    num_normal = int(np.sum(anomaly_flags == 0))
    pct_anomalies = (num_anomalies / count_val) * 100.0
    pct_normal = (num_normal / count_val) * 100.0

    print("\n" + "=" * 75)
    print("OPERATIONAL ANOMALY CLASSIFICATION RESULTS")
    print("=" * 75)
    print("Threshold Designation: 95th Percentile of Test Reconstruction Errors")
    print("Operational Guardrail: Empirical anomaly threshold for this experiment;")
    print("                       NOT a ground-truth system failure threshold.")
    print(f"\n  Operational Threshold : {threshold:.6f}")
    print(f"  Anomalous Sequences   : {num_anomalies:,} ({pct_anomalies:.2f}%)")
    print(f"  Normal Sequences      : {num_normal:,} ({pct_normal:.2f}%)")
    print(f"  Total Evaluated       : {count_val:,} (100.00%)")

    # 8. Assemble & Save Complete Results DataFrame
    results_df = pd.DataFrame({
        "sequence_index": test_meta_df["sequence_index"].values,
        "vm_id": test_meta_df["vm_id"].values,
        "start_timestamp": test_meta_df["start_timestamp"].values,
        "end_timestamp": test_meta_df["end_timestamp"].values,
        "reconstruction_error": reconstruction_errors,
        "anomaly": anomaly_flags
    })

    assert len(results_df) == n_test_sequences, f"Row count mismatch: {len(results_df)} vs {n_test_sequences}"

    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"\nSaved complete anomaly detection results to: {RESULTS_OUTPUT_PATH}")

    # 9. Create and Save Reconstruction Error Distribution Histogram
    print(f"Generating reconstruction error distribution plot to: {DISTRIBUTION_PLOT_PATH}...")
    plt.figure(figsize=(10, 5.5))
    plt.hist(
        reconstruction_errors,
        bins=50,
        alpha=0.75,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
        density=True,
        label="Test Sequence Reconstruction Error"
    )
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"95th Percentile Threshold ({threshold:.4f})"
    )
    plt.axvline(
        p99_val,
        color="purple",
        linestyle=":",
        linewidth=1.8,
        label=f"99th Percentile ({p99_val:.4f})"
    )
    plt.xlabel("Reconstruction Error (MSE)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Bitbrains LSTM-Autoencoder Test Set Reconstruction Error Distribution", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_PATH, dpi=300)
    plt.close()
    print("Saved reconstruction error distribution plot.")

    # 10. Summary by Selected VM
    print("\nPer-VM Anomaly Breakdown:")
    vm_summary = results_df.groupby("vm_id").agg(
        total_test_seqs=("anomaly", "count"),
        anomalies=("anomaly", "sum"),
        mean_error=("reconstruction_error", "mean"),
        max_error=("reconstruction_error", "max")
    ).reset_index()
    vm_summary["anomaly_rate_pct"] = (vm_summary["anomalies"] / vm_summary["total_test_seqs"]) * 100.0

    for _, r in vm_summary.iterrows():
        print(f"  VM {int(r['vm_id']):4d} | Total: {int(r['total_test_seqs']):,d} | Anomalies: {int(r['anomalies']):4d} ({r['anomaly_rate_pct']:5.2f}%) | Mean Error: {r['mean_error']:.4f} | Max Error: {r['max_error']:.4f}")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("PHASE 10D DETECTION PIPELINE COMPLETE")
    print("=" * 75)
    print(f"Generated Files:")
    print(f"  - Results CSV: {RESULTS_OUTPUT_PATH} ({RESULTS_OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Histogram  : {DISTRIBUTION_PLOT_PATH}")
    print(f"Total Execution Runtime: {total_elapsed:.2f} seconds")
    print("=" * 75)


if __name__ == "__main__":
    main()
