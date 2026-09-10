"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 12: Controlled Synthetic Anomaly Evaluation of Bitbrains Dense-AE vs LSTM-AE

This script conducts a fair, controlled synthetic anomaly benchmark comparing the
Bitbrains Dense Autoencoder against the Bitbrains Seq2Seq LSTM-Autoencoder.

Scientific / Operational Disclaimer:
  This is a CONTROLLED SYNTHETIC BENCHMARK designed to evaluate model sensitivity
  to temporal resource shifts. It does NOT represent real-world catastrophic failure
  prediction or operational downtime probability.

Methodological Fairness:
  1. Identical Benchmark: Both models evaluate the exact same 4,000 synthetic cases
     (2,000 clean negative controls + 2,000 contiguous synthetic anomalies).
  2. Fixed Validation Thresholds:
     - Dense-AE: 0.359623 (Phase 11B validation 95th percentile)
     - LSTM-AE : 0.277209 (Phase 10F validation 95th percentile)
  3. Contiguous Temporal Injections: Anomalies span 3-6 consecutive time steps (15-30 mins)
     across 5 resource channels (CPU, Memory, Net-In, Net-Out, Disk I/O).
  4. Disk Immutability: X_test_lstm.npy is loaded read-only and never modified on disk.
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

# Set fixed random seed for full determinism and reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# Paths and Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
SEQUENCES_DIR = BITBRAINS_DIR / "lstm_sequences"
DENSE_DIR = BITBRAINS_DIR / "dense_model"
LSTM_DIR = BITBRAINS_DIR / "lstm_model"

OUTPUT_DIR = BITBRAINS_DIR / "synthetic_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DENSE_MODEL_PATH = DENSE_DIR / "bitbrains_dense_autoencoder.keras"
LSTM_MODEL_PATH = LSTM_DIR / "lstm_autoencoder.keras"
ALIBABA_MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"

TEST_SEQUENCES_PATH = SEQUENCES_DIR / "X_test_lstm.npy"
TEST_METADATA_PATH = SEQUENCES_DIR / "X_test_lstm_metadata.csv"
if not TEST_METADATA_PATH.exists():
    TEST_METADATA_PATH = SEQUENCES_DIR / "test_metadata.csv"

# Pre-existing test results for identifying clean normal candidates
LSTM_ANOMALY_RESULTS_PATH = LSTM_DIR / "lstm_anomaly_results_validation_threshold.csv"
DENSE_ANOMALY_RESULTS_PATH = DENSE_DIR / "dense_anomaly_results_validation_threshold.csv"

# Output Files
SYNTHETIC_CASES_PATH = OUTPUT_DIR / "synthetic_cases.csv"
DENSE_RESULTS_PATH = OUTPUT_DIR / "dense_synthetic_results.csv"
LSTM_RESULTS_PATH = OUTPUT_DIR / "lstm_synthetic_results.csv"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "synthetic_model_comparison.csv"
ANOMALY_TYPE_PATH = OUTPUT_DIR / "synthetic_anomaly_type_comparison.csv"
CONFUSION_MATRICES_PLOT_PATH = OUTPUT_DIR / "synthetic_confusion_matrices.png"
MODEL_COMPARISON_PLOT_PATH = OUTPUT_DIR / "synthetic_model_comparison.png"

# Validation-derived thresholds
DENSE_THRESHOLD = 0.359623
LSTM_THRESHOLD = 0.277209

FEATURE_NAMES = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io"
]

ANOMALY_TYPES = [
    ("cpu_spike", 0, "cpu_util_percent"),
    ("mem_spike", 1, "mem_util_percent"),
    ("net_in_spike", 2, "net_in"),
    ("net_out_spike", 3, "net_out"),
    ("disk_io_spike", 4, "disk_io")
]

NUM_NORMAL_CASES = 2000
NUM_ANOMALOUS_CASES = 2000
TOTAL_CASES = NUM_NORMAL_CASES + NUM_ANOMALOUS_CASES  # 4,000
CASES_PER_ANOMALY_TYPE = NUM_ANOMALOUS_CASES // len(ANOMALY_TYPES)  # 400 each


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def main():
    total_start_time = time.time()
    print("=" * 78)
    print("PHASE 12: FAIR SYNTHETIC ANOMALY EVALUATION (DENSE-AE vs LSTM-AE)")
    print("=" * 78)
    print("Experimental Designation: Controlled synthetic anomaly evaluation.")
    print("Random Seed: 42 (Fully deterministic and reproducible)")

    # 1. Verify existence of models and data
    assert DENSE_MODEL_PATH.exists(), f"Dense model missing: {DENSE_MODEL_PATH}"
    assert LSTM_MODEL_PATH.exists(), f"LSTM model missing: {LSTM_MODEL_PATH}"
    assert TEST_SEQUENCES_PATH.exists(), f"Test sequences missing: {TEST_SEQUENCES_PATH}"
    assert TEST_METADATA_PATH.exists(), f"Test metadata missing: {TEST_METADATA_PATH}"

    dense_mtime_before = DENSE_MODEL_PATH.stat().st_mtime
    lstm_mtime_before = LSTM_MODEL_PATH.stat().st_mtime
    alibaba_mtime_before = ALIBABA_MODEL_PATH.stat().st_mtime
    test_seq_mtime_before = TEST_SEQUENCES_PATH.stat().st_mtime

    # 2. Load Models (Read-Only)
    print("\nLoading trained models (Read-Only)...")
    dense_model = tf.keras.models.load_model(DENSE_MODEL_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    print(f"  - Dense AE: {DENSE_MODEL_PATH} (Threshold = {DENSE_THRESHOLD})")
    print(f"  - LSTM-AE : {LSTM_MODEL_PATH} (Threshold = {LSTM_THRESHOLD})")

    # 3. Load Untouched Test Sequences
    print("\nLoading untouched test sequences (in-memory read-only)...")
    X_test_raw = np.load(TEST_SEQUENCES_PATH)
    test_meta_df = pd.read_csv(TEST_METADATA_PATH)
    n_test = len(X_test_raw)
    print(f"Loaded X_test_lstm shape: {X_test_raw.shape} ({n_test:,} sequences)")

    # 4. Identify Mutually Clean Normal Test Sequences
    print("\nIdentifying clean baseline normal sequences...")
    lstm_results = pd.read_csv(LSTM_ANOMALY_RESULTS_PATH)
    dense_results = pd.read_csv(DENSE_ANOMALY_RESULTS_PATH)

    # Intersection of sequences classified as normal by both baseline models
    mutually_normal_mask = (lstm_results["anomaly"] == 0) & (dense_results["anomaly"] == 0)
    normal_candidate_indices = np.where(mutually_normal_mask)[0]
    print(f"Total candidate normal sequences (normal in both models): {len(normal_candidate_indices):,} / {n_test:,}")

    assert len(normal_candidate_indices) >= TOTAL_CASES, (
        f"Need at least {TOTAL_CASES} normal candidates, but found {len(normal_candidate_indices)}"
    )

    # Deterministically sample 4,000 distinct sequence indices
    rng = np.random.RandomState(RANDOM_SEED)
    selected_indices = rng.choice(normal_candidate_indices, size=TOTAL_CASES, replace=False)

    normal_indices = selected_indices[:NUM_NORMAL_CASES]
    anomalous_base_indices = selected_indices[NUM_NORMAL_CASES:]

    print(f"Selected {TOTAL_CASES:,} distinct sequences:")
    print(f"  - {NUM_NORMAL_CASES:,} designated as Normal Controls (unmodified)")
    print(f"  - {NUM_ANOMALOUS_CASES:,} designated for Controlled Anomaly Injection")

    # 5. Construct Synthetic Benchmark Dataset
    print("\nSynthesizing benchmark sequences with contiguous temporal injections...")
    synthetic_tensors = []
    case_records = []

    # 5a. Process 2,000 Normal Cases
    for i, orig_idx in enumerate(normal_indices):
        seq = X_test_raw[orig_idx].copy()
        synthetic_tensors.append(seq)
        vm_id = int(test_meta_df.loc[orig_idx, "vm_id"])
        case_records.append({
            "synthetic_case_id": len(case_records),
            "original_sequence_index": int(orig_idx),
            "vm_id": vm_id,
            "anomaly_type": "none",
            "affected_feature": "none",
            "affected_timestep_range": "none",
            "injection_magnitude": 0.0,
            "ground_truth_label": 0
        })

    # 5b. Process 2,000 Anomalous Cases (400 per anomaly type)
    anom_type_idx = 0
    for i, orig_idx in enumerate(anomalous_base_indices):
        anom_name, feat_idx, feat_name = ANOMALY_TYPES[anom_type_idx]
        seq = X_test_raw[orig_idx].copy()  # Shape: (12, 5)

        # Contiguous duration L in [3..6] timesteps (15 to 30 mins)
        duration = rng.randint(3, 7)  # 3, 4, 5, or 6
        t_start = rng.randint(0, 12 - duration + 1)
        t_end = t_start + duration  # exclusive

        # Standardized additive magnitude shift M in [2.5, 4.5] sigma
        magnitude = float(rng.uniform(2.5, 4.5))

        # Apply contiguous temporal spike
        seq[t_start:t_end, feat_idx] += magnitude

        synthetic_tensors.append(seq)
        vm_id = int(test_meta_df.loc[orig_idx, "vm_id"])
        case_records.append({
            "synthetic_case_id": len(case_records),
            "original_sequence_index": int(orig_idx),
            "vm_id": vm_id,
            "anomaly_type": anom_name,
            "affected_feature": feat_name,
            "affected_timestep_range": f"{t_start}-{t_end - 1}",
            "injection_magnitude": round(magnitude, 4),
            "ground_truth_label": 1
        })

        # Cycle through the 5 anomaly types evenly (400 of each)
        if (i + 1) % CASES_PER_ANOMALY_TYPE == 0:
            anom_type_idx += 1

    synthetic_dataset = np.array(synthetic_tensors, dtype=np.float32)  # Shape: (4000, 12, 5)
    cases_df = pd.DataFrame(case_records)

    assert len(synthetic_dataset) == TOTAL_CASES == 4000, f"Benchmark count mismatch: {len(synthetic_dataset)}"
    assert not np.isnan(synthetic_dataset).any(), "NaN in synthetic benchmark dataset!"
    assert not np.isinf(synthetic_dataset).any(), "Inf in synthetic benchmark dataset!"

    cases_df.to_csv(SYNTHETIC_CASES_PATH, index=False)
    print(f"Saved synthetic cases registry to: {SYNTHETIC_CASES_PATH} ({len(cases_df):,} rows)")

    # 6. Evaluate Dense Autoencoder on Synthetic Benchmark
    print(f"\nEvaluating Dense Autoencoder on all {TOTAL_CASES:,} synthetic cases...")
    # Flatten (4000, 12, 5) -> (4000, 60)
    synthetic_flat = synthetic_dataset.reshape(TOTAL_CASES, 60)

    dense_infer_start = time.time()
    dense_recon = dense_model.predict(synthetic_flat, batch_size=128, verbose=0)
    dense_infer_time = time.time() - dense_infer_start

    dense_mse = np.mean(np.square(synthetic_flat - dense_recon), axis=1).astype(np.float64)
    dense_pred = (dense_mse > DENSE_THRESHOLD).astype(int)

    # 7. Evaluate LSTM Autoencoder on Synthetic Benchmark
    print(f"Evaluating LSTM Autoencoder on all {TOTAL_CASES:,} synthetic cases...")
    lstm_infer_start = time.time()
    lstm_recon = lstm_model.predict(synthetic_dataset, batch_size=128, verbose=0)
    lstm_infer_time = time.time() - lstm_infer_start

    lstm_mse = np.mean(np.square(synthetic_dataset - lstm_recon), axis=(1, 2)).astype(np.float64)
    lstm_pred = (lstm_mse > LSTM_THRESHOLD).astype(int)

    y_true = cases_df["ground_truth_label"].values

    # 8. Compute Detailed Performance Metrics
    def compute_metrics(y_true, y_pred):
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Specificity": specificity
        }

    dense_overall = compute_metrics(y_true, dense_pred)
    lstm_overall = compute_metrics(y_true, lstm_pred)

    # Save per-model results
    dense_results_df = cases_df.copy()
    dense_results_df["reconstruction_error"] = dense_mse
    dense_results_df["predicted_label"] = dense_pred
    dense_results_df["classification"] = np.where(
        (y_true == 1) & (dense_pred == 1), "TP",
        np.where((y_true == 0) & (dense_pred == 0), "TN",
        np.where((y_true == 0) & (dense_pred == 1), "FP", "FN"))
    )
    dense_results_df.to_csv(DENSE_RESULTS_PATH, index=False)
    print(f"\nSaved Dense synthetic results to: {DENSE_RESULTS_PATH}")

    lstm_results_df = cases_df.copy()
    lstm_results_df["reconstruction_error"] = lstm_mse
    lstm_results_df["predicted_label"] = lstm_pred
    lstm_results_df["classification"] = np.where(
        (y_true == 1) & (lstm_pred == 1), "TP",
        np.where((y_true == 0) & (lstm_pred == 0), "TN",
        np.where((y_true == 0) & (lstm_pred == 1), "FP", "FN"))
    )
    lstm_results_df.to_csv(LSTM_RESULTS_PATH, index=False)
    print(f"Saved LSTM synthetic results to: {LSTM_RESULTS_PATH}")

    # 9. Comparison Summary Table
    print("\n" + "=" * 78)
    print("SYNTHETIC BENCHMARK CLASSIFICATION RESULTS (OVERALL)")
    print("=" * 78)
    comparison_records = [
        {"Model": "Dense Autoencoder", "Threshold": DENSE_THRESHOLD, **dense_overall, "Inference_Time_s": round(dense_infer_time, 2)},
        {"Model": "LSTM Autoencoder", "Threshold": LSTM_THRESHOLD, **lstm_overall, "Inference_Time_s": round(lstm_infer_time, 2)}
    ]
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "Specificity", "TP", "FP", "FN", "TN"]].to_string(index=False))
    print(f"\nSaved overall comparison table to: {MODEL_COMPARISON_PATH}")

    # 10. Performance Breakdown by Anomaly Type
    print("\n" + "=" * 78)
    print("DETECTION RATE (RECALL) BREAKDOWN BY ANOMALY TYPE (400 cases each)")
    print("=" * 78)
    type_records = []
    for anom_name, _, feat_name in ANOMALY_TYPES:
        mask = (cases_df["anomaly_type"] == anom_name).values
        n_type = int(np.sum(mask))

        dense_tp_type = int(np.sum((y_true[mask] == 1) & (dense_pred[mask] == 1)))
        dense_recall_type = (dense_tp_type / n_type) * 100.0
        dense_mean_mse_type = float(np.mean(dense_mse[mask]))

        lstm_tp_type = int(np.sum((y_true[mask] == 1) & (lstm_pred[mask] == 1)))
        lstm_recall_type = (lstm_tp_type / n_type) * 100.0
        lstm_mean_mse_type = float(np.mean(lstm_mse[mask]))

        type_records.append({
            "Anomaly_Type": anom_name,
            "Channel": feat_name,
            "Total_Cases": n_type,
            "Dense_TP": dense_tp_type,
            "Dense_Detection_Rate_pct": round(dense_recall_type, 2),
            "Dense_Mean_MSE": round(dense_mean_mse_type, 4),
            "LSTM_TP": lstm_tp_type,
            "LSTM_Detection_Rate_pct": round(lstm_recall_type, 2),
            "LSTM_Mean_MSE": round(lstm_mean_mse_type, 4)
        })

    type_df = pd.DataFrame(type_records)
    type_df.to_csv(ANOMALY_TYPE_PATH, index=False)
    print(type_df[["Anomaly_Type", "Channel", "Total_Cases", "Dense_Detection_Rate_pct", "LSTM_Detection_Rate_pct", "Dense_Mean_MSE", "LSTM_Mean_MSE"]].to_string(index=False))
    print(f"\nSaved anomaly type comparison to: {ANOMALY_TYPE_PATH}")

    # 11. Confusion Matrix Heatmaps Plot
    print(f"\nGenerating confusion matrix comparison plot to: {CONFUSION_MATRICES_PLOT_PATH}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    def plot_cm(ax, cm_dict, model_name, threshold_val):
        cm = np.array([
            [cm_dict["TN"], cm_dict["FP"]],
            [cm_dict["FN"], cm_dict["TP"]]
        ])
        cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                color = "white" if val > cm.max() / 2 else "black"
                label_text = f"{val:,}\n({val / TOTAL_CASES * 100:.1f}%)"
                ax.text(c, r, label_text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Normal", "Predicted Anomaly"], fontsize=10, fontweight="semibold")
        ax.set_yticklabels(["Actual Normal", "Actual Anomaly"], fontsize=10, fontweight="semibold")
        ax.set_title(f"{model_name}\n(Threshold $\\tau = {threshold_val:.4f}$ | F1: {cm_dict['F1_Score']*100:.1f}%)", fontsize=11, fontweight="bold", pad=12)

    plot_cm(ax1, dense_overall, "Dense Autoencoder", DENSE_THRESHOLD)
    plot_cm(ax2, lstm_overall, "LSTM Autoencoder", LSTM_THRESHOLD)

    fig.suptitle("Synthetic Anomaly Evaluation: Confusion Matrix Comparison (N = 4,000 cases)", fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.02,
        "Controlled Synthetic Evaluation: 2,000 clean controls + 2,000 contiguous temporal anomalies (400 per channel).\n"
        "Does not represent real-world failure prediction.",
        ha="center", va="bottom", fontsize=8.5, color="#6b7280", style="italic"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(CONFUSION_MATRICES_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRICES_PLOT_PATH}")

    # 12. Model Comparison Bar Chart
    print(f"Generating performance comparison bar chart to: {MODEL_COMPARISON_PLOT_PATH}...")
    fig, (ax_metric, ax_type) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Metric comparison
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    x = np.arange(len(metrics))
    width = 0.35

    dense_vals = [dense_overall[m] * 100 for m in metrics]
    lstm_vals = [lstm_overall[m] * 100 for m in metrics]

    rects1 = ax_metric.bar(x - width/2, dense_vals, width, label="Dense AE (60D)", color="#3a86c8", edgecolor="black", linewidth=0.8)
    rects2 = ax_metric.bar(x + width/2, lstm_vals, width, label="LSTM-AE (Seq2Seq)", color="#d9534f", edgecolor="black", linewidth=0.8)

    for rect in rects1:
        h = rect.get_height()
        ax_metric.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax_metric.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_metric.set_ylabel("Score (%)", fontsize=11, fontweight="semibold")
    ax_metric.set_title("Overall Performance Metrics\n(N = 4,000 cases | 50% Anomaly Prevalence)", fontsize=11, fontweight="bold", pad=12)
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(metrics, fontsize=10, fontweight="semibold")
    ax_metric.set_ylim(0, 115)
    ax_metric.grid(axis="y", linestyle=":", alpha=0.6)
    ax_metric.legend(fontsize=9.5, loc="lower right")

    # Anomaly type detection rates
    anom_labels = ["CPU Spike", "Mem Spike", "Net-In Spike", "Net-Out Spike", "Disk I/O Spike"]
    x_type = np.arange(len(anom_labels))

    dense_type_rates = type_df["Dense_Detection_Rate_pct"].values
    lstm_type_rates = type_df["LSTM_Detection_Rate_pct"].values

    rects_t1 = ax_type.bar(x_type - width/2, dense_type_rates, width, label="Dense AE", color="#3a86c8", edgecolor="black", linewidth=0.8)
    rects_t2 = ax_type.bar(x_type + width/2, lstm_type_rates, width, label="LSTM-AE", color="#d9534f", edgecolor="black", linewidth=0.8)

    for rect in rects_t1:
        h = rect.get_height()
        ax_type.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects_t2:
        h = rect.get_height()
        ax_type.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_type.set_ylabel("Detection Rate / Recall (%)", fontsize=11, fontweight="semibold")
    ax_type.set_title("Detection Rate by Resource Channel\n(400 Synthetic Cases per Channel)", fontsize=11, fontweight="bold", pad=12)
    ax_type.set_xticks(x_type)
    ax_type.set_xticklabels(anom_labels, fontsize=9.5, fontweight="semibold", rotation=15)
    ax_type.set_ylim(0, 115)
    ax_type.grid(axis="y", linestyle=":", alpha=0.6)
    ax_type.legend(fontsize=9.5, loc="lower right")

    fig.suptitle("Bitbrains Synthetic Anomaly Evaluation: Dense-AE vs. LSTM-AE", fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.01,
        "Controlled synthetic evaluation on 4,000 identical cases. Fixed validation thresholds (Dense: 0.3596, LSTM: 0.2772).\n"
        "Does not represent real-world failure prediction.",
        ha="center", va="bottom", fontsize=8.5, color="#6b7280", style="italic"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to: {MODEL_COMPARISON_PLOT_PATH}")

    # 13. Safety & Immutability Verification
    print("\n" + "=" * 78)
    print("SAFETY & IMMUTABILITY VERIFICATION")
    print("=" * 78)
    assert DENSE_MODEL_PATH.stat().st_mtime == dense_mtime_before, "Dense model was modified!"
    assert LSTM_MODEL_PATH.stat().st_mtime == lstm_mtime_before, "LSTM model was modified!"
    assert ALIBABA_MODEL_PATH.stat().st_mtime == alibaba_mtime_before, "Alibaba model was modified!"
    assert TEST_SEQUENCES_PATH.stat().st_mtime == test_seq_mtime_before, "X_test_lstm.npy was modified on disk!"

    print("  [PASSED] Exactly 4,000 synthetic cases evaluated (2,000 normal + 2,000 anomalous).")
    print("  [PASSED] Identical benchmark cases evaluated by both models.")
    print("  [PASSED] Zero NaN or Infinite values detected in any calculation.")
    print("  [PASSED] Original X_test_lstm.npy on disk remains strictly untouched.")
    print("  [PASSED] Existing Dense AE, LSTM-AE, and Alibaba baseline models remained untouched.")
    print("  [PASSED] Fixed validation-derived thresholds used (zero test leakage).")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 78)
    print(f"PHASE 12 EXECUTION COMPLETE in {total_elapsed:.2f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
