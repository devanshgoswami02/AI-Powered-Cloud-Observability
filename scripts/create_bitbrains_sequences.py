"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 10A: Bitbrains LSTM Sequence Generation Pipeline

This script extracts and validates fixed-length temporal sequences (T=12, stride=1)
for an LSTM-Autoencoder from the preprocessed Bitbrains VM telemetry.

Key Design & Guardrails:
1. Deterministic Stratified VM Selection: Exactly 10 representative VMs (>= 6,800 train rows)
   spanning High-CPU, High-Memory, High-Network, High-Disk, and Typical/Balanced workloads.
2. Boundary Isolation: Sequences are generated strictly within individual VM traces;
   never crossing VM boundaries.
3. Train/Test Separation: Train sequences come exclusively from X_train_bitbrains.csv;
   test sequences come exclusively from X_test_bitbrains.csv.
4. Output Format: NumPy .npy arrays (float32) of shape (N, 12, 5) with corresponding
   metadata CSVs tracking vm_id and end_timestamp for anomaly traceability.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Paths and Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
TRAIN_INPUT_PATH = BITBRAINS_OUTPUT_DIR / "X_train_bitbrains.csv"
TEST_INPUT_PATH = BITBRAINS_OUTPUT_DIR / "X_test_bitbrains.csv"

SELECTED_VMS_PATH = BITBRAINS_OUTPUT_DIR / "selected_vms.csv"

SEQUENCE_OUTPUT_DIR = BITBRAINS_OUTPUT_DIR / "lstm_sequences"
SEQUENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEQUENCES_PATH = SEQUENCE_OUTPUT_DIR / "X_train_lstm.npy"
TEST_SEQUENCES_PATH = SEQUENCE_OUTPUT_DIR / "X_test_lstm.npy"
TRAIN_META_PATH = SEQUENCE_OUTPUT_DIR / "train_metadata.csv"
TEST_META_PATH = SEQUENCE_OUTPUT_DIR / "test_metadata.csv"
SUMMARY_PATH = SEQUENCE_OUTPUT_DIR / "sequence_summary.csv"

SEQUENCE_LENGTH = 12  # T = 12 time steps (1.0 hour at 5-minute sampling)
STRIDE = 1
MIN_TRAIN_OBSERVATIONS = 6800  # Strict filter ensuring full 1-month traces

FEATURE_COLUMNS = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io"
]


# ============================================================
# Deterministic VM Selection
# ============================================================

def select_representative_vms(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Select exactly 10 representative VMs using a reproducible, deterministic rule:
      - 2 High CPU
      - 2 High Memory
      - 2 High Network
      - 2 High Disk I/O
      - 2 Typical / Balanced (closest to baseline origin in scaled space)
    Filtered strictly to VMs having >= 6,800 training observations.
    """
    print("\n[Step 1/4] Calculating fleet resource profiles for VM selection...")

    agg = train_df.groupby("vm_id").agg(
        train_rows=("timestamp", "count"),
        cpu_mean=("cpu_util_percent", "mean"),
        mem_mean=("mem_util_percent", "mean"),
        net_in_mean=("net_in", "mean"),
        net_out_mean=("net_out", "mean"),
        disk_mean=("disk_io", "mean")
    ).reset_index()

    agg["net_combined_mean"] = agg["net_in_mean"] + agg["net_out_mean"]

    # Filter to VMs with >= 6,800 train rows
    eligible = agg[agg["train_rows"] >= MIN_TRAIN_OBSERVATIONS].copy()
    print(f"Total eligible VMs with >= {MIN_TRAIN_OBSERVATIONS} train rows: {len(eligible):,}")

    test_row_counts = test_df.groupby("vm_id")["timestamp"].count().to_dict()
    eligible["test_rows"] = eligible["vm_id"].map(test_row_counts)

    selected_vms = []
    category_assignments = {}

    def pick_top_vms(candidate_df, sort_col, category_name, k=2):
        picked = []
        sorted_candidates = candidate_df.sort_values(sort_col, ascending=False)
        for vid in sorted_candidates["vm_id"].tolist():
            if vid not in selected_vms:
                picked.append(vid)
                selected_vms.append(vid)
                category_assignments[vid] = category_name
                if len(picked) == k:
                    break
        return picked

    # 1. High CPU
    pick_top_vms(eligible, "cpu_mean", "high_cpu", k=2)

    # 2. High Memory
    pick_top_vms(eligible[~eligible["vm_id"].isin(selected_vms)], "mem_mean", "high_memory", k=2)

    # 3. High Network
    pick_top_vms(eligible[~eligible["vm_id"].isin(selected_vms)], "net_combined_mean", "high_network", k=2)

    # 4. High Disk I/O
    pick_top_vms(eligible[~eligible["vm_id"].isin(selected_vms)], "disk_mean", "high_disk", k=2)

    # 5. Typical / Balanced (closest Euclidean distance to origin in scaled space)
    rem = eligible[~eligible["vm_id"].isin(selected_vms)].copy()
    rem["dist_from_mean"] = np.sqrt(
        rem["cpu_mean"]**2 +
        rem["mem_mean"]**2 +
        rem["net_in_mean"]**2 +
        rem["net_out_mean"]**2 +
        rem["disk_mean"]**2
    )
    balanced_candidates = rem.sort_values("dist_from_mean", ascending=True)
    for vid in balanced_candidates["vm_id"].tolist():
        if vid not in selected_vms:
            selected_vms.append(vid)
            category_assignments[vid] = "typical_balanced"
            if len([v for v, c in category_assignments.items() if c == "typical_balanced"]) == 2:
                break

    # Build summary record for selected VMs
    selected_records = []
    for vid in selected_vms:
        r = eligible[eligible["vm_id"] == vid].iloc[0]
        selected_records.append({
            "vm_id": vid,
            "category": category_assignments[vid],
            "train_rows": int(r["train_rows"]),
            "test_rows": int(r["test_rows"]),
            "cpu_mean_scaled": round(float(r["cpu_mean"]), 4),
            "mem_mean_scaled": round(float(r["mem_mean"]), 4),
            "net_in_mean_scaled": round(float(r["net_in_mean"]), 4),
            "net_out_mean_scaled": round(float(r["net_out_mean"]), 4),
            "disk_mean_scaled": round(float(r["disk_mean"]), 4)
        })

    selected_df = pd.DataFrame(selected_records)
    selected_df.to_csv(SELECTED_VMS_PATH, index=False)
    print(f"Saved selected VMs record to: {SELECTED_VMS_PATH}")

    return selected_df, selected_vms


# ============================================================
# Sequence Building with VM Boundary Isolation
# ============================================================

def build_vm_isolated_sequences(df: pd.DataFrame, selected_vms: list, seq_length: int = 12, stride: int = 1):
    """
    Generate sliding window sequences strictly isolated within each VM.
    Guarantees:
      1. Every sequence is drawn from a single VM trace.
      2. No sequence ever spans or crosses across a VM boundary.
      3. Preserves vm_id and end_timestamp for each sequence.
    """
    sequences = []
    metadata = []
    sequences_per_vm = {}

    # Filter to selected VMs
    filtered_df = df[df["vm_id"].isin(selected_vms)].copy()

    # Iterate VM by VM to guarantee boundary isolation
    for vm_id in selected_vms:
        vm_group = filtered_df[filtered_df["vm_id"] == vm_id].sort_values("timestamp")
        n_obs = len(vm_group)

        if n_obs < seq_length:
            raise ValueError(f"VM {vm_id} has only {n_obs} observations, which is less than sequence length {seq_length}")

        feature_matrix = vm_group[FEATURE_COLUMNS].values.astype(np.float32)
        timestamps = vm_group["timestamp"].values.astype(np.int64)

        vm_seq_count = 0
        for i in range(0, n_obs - seq_length + 1, stride):
            window = feature_matrix[i : i + seq_length]
            sequences.append(window)
            metadata.append({
                "sequence_index": len(sequences) - 1,
                "vm_id": vm_id,
                "start_timestamp": timestamps[i],
                "end_timestamp": timestamps[i + seq_length - 1]
            })
            vm_seq_count += 1

        sequences_per_vm[vm_id] = vm_seq_count

    sequences_array = np.array(sequences, dtype=np.float32)
    metadata_df = pd.DataFrame(metadata)

    return sequences_array, metadata_df, sequences_per_vm


# ============================================================
# Main Pipeline
# ============================================================

def main():
    start_time = time.time()
    print("=" * 75)
    print("BITBRAINS LSTM SEQUENCE GENERATION PIPELINE (PHASE 10A)")
    print("=" * 75)

    if not TRAIN_INPUT_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_INPUT_PATH}")
    if not TEST_INPUT_PATH.exists():
        raise FileNotFoundError(f"Testing dataset not found: {TEST_INPUT_PATH}")

    # Load preprocessed datasets
    print("Loading preprocessed training and testing data...")
    train_df = pd.read_csv(TRAIN_INPUT_PATH)
    test_df = pd.read_csv(TEST_INPUT_PATH)
    print(f"Loaded X_train_bitbrains.csv: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    print(f"Loaded X_test_bitbrains.csv : {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")

    # Step 1: Select exactly 10 representative VMs
    selected_df, selected_vms = select_representative_vms(train_df, test_df)

    print("\nSelected 10 Representative VMs:")
    for _, r in selected_df.iterrows():
        print(f"  VM {r['vm_id']:4d} | Category: {r['category']:16s} | Train Obs: {r['train_rows']:,} | Test Obs: {r['test_rows']:,}")

    # Step 2: Build Training Sequences (strictly from X_train_bitbrains.csv)
    print("\n[Step 2/4] Building training sequences with strict VM boundary isolation...")
    X_train_lstm, train_meta_df, train_seqs_per_vm = build_vm_isolated_sequences(
        train_df, selected_vms, seq_length=SEQUENCE_LENGTH, stride=STRIDE
    )
    print(f"Generated X_train_lstm shape: {X_train_lstm.shape} (dtype: {X_train_lstm.dtype})")

    # Step 3: Build Testing Sequences (strictly from X_test_bitbrains.csv)
    print("\n[Step 3/4] Building testing sequences with strict VM boundary isolation...")
    X_test_lstm, test_meta_df, test_seqs_per_vm = build_vm_isolated_sequences(
        test_df, selected_vms, seq_length=SEQUENCE_LENGTH, stride=STRIDE
    )
    print(f"Generated X_test_lstm shape: {X_test_lstm.shape} (dtype: {X_test_lstm.dtype})")

    # Step 4: Verification before saving
    print("\n[Step 4/4] Verifying sequence integrity and boundary isolation...")

    # Assertions
    assert X_train_lstm.shape[1:] == (SEQUENCE_LENGTH, 5), f"Unexpected train shape: {X_train_lstm.shape}"
    assert X_test_lstm.shape[1:] == (SEQUENCE_LENGTH, 5), f"Unexpected test shape: {X_test_lstm.shape}"
    assert X_train_lstm.dtype == np.float32, f"Train dtype is {X_train_lstm.dtype}, expected float32"
    assert X_test_lstm.dtype == np.float32, f"Test dtype is {X_test_lstm.dtype}, expected float32"

    train_nan_count = int(np.isnan(X_train_lstm).sum())
    test_nan_count = int(np.isnan(X_test_lstm).sum())
    train_inf_count = int(np.isinf(X_train_lstm).sum())
    test_inf_count = int(np.isinf(X_test_lstm).sum())

    assert train_nan_count == 0, f"Found {train_nan_count} NaNs in train sequences!"
    assert test_nan_count == 0, f"Found {test_nan_count} NaNs in test sequences!"
    assert train_inf_count == 0, f"Found {train_inf_count} Infs in train sequences!"
    assert test_inf_count == 0, f"Found {test_inf_count} Infs in test sequences!"

    # Verify boundary isolation mathematically:
    # Each VM with N_vm observations must yield exactly (N_vm - T + 1) sequences
    for vid in selected_vms:
        expected_train_seq = selected_df[selected_df["vm_id"] == vid]["train_rows"].iloc[0] - SEQUENCE_LENGTH + 1
        expected_test_seq = selected_df[selected_df["vm_id"] == vid]["test_rows"].iloc[0] - SEQUENCE_LENGTH + 1
        assert train_seqs_per_vm[vid] == expected_train_seq, f"VM {vid} train sequence count mismatch!"
        assert test_seqs_per_vm[vid] == expected_test_seq, f"VM {vid} test sequence count mismatch!"

    # Verify partition isolation: train end_timestamp <= test start_timestamp for every VM
    for vid in selected_vms:
        max_train_ts = train_meta_df[train_meta_df["vm_id"] == vid]["end_timestamp"].max()
        min_test_ts = test_meta_df[test_meta_df["vm_id"] == vid]["start_timestamp"].min()
        assert max_train_ts <= min_test_ts, f"Temporal inversion detected in VM {vid}: train max ts {max_train_ts} > test min ts {min_test_ts}"

    print("Integrity Verifications Passed:")
    print("  [PASSED] No NaN values found")
    print("  [PASSED] No Infinite values found")
    print("  [PASSED] Tensor dtype is float32")
    print("  [PASSED] Sequence shape is exactly (12, 5)")
    print("  [PASSED] Cross-VM boundary isolation confirmed for all 10 VMs")
    print("  [PASSED] Strict temporal separation (train timestamps strictly precede test timestamps)")

    # Save outputs
    print("\nSaving sequences and metadata artifacts...")
    np.save(TRAIN_SEQUENCES_PATH, X_train_lstm)
    np.save(TEST_SEQUENCES_PATH, X_test_lstm)
    train_meta_df.to_csv(TRAIN_META_PATH, index=False)
    test_meta_df.to_csv(TEST_META_PATH, index=False)

    summary_records = [
        {"metric": "selected_vms_count", "value": len(selected_vms)},
        {"metric": "selected_vm_ids", "value": str(selected_vms)},
        {"metric": "sequence_length_T", "value": SEQUENCE_LENGTH},
        {"metric": "stride", "value": STRIDE},
        {"metric": "features_count", "value": len(FEATURE_COLUMNS)},
        {"metric": "features_list", "value": str(FEATURE_COLUMNS)},
        {"metric": "train_sequences_count", "value": X_train_lstm.shape[0]},
        {"metric": "test_sequences_count", "value": X_test_lstm.shape[0]},
        {"metric": "total_sequences_count", "value": X_train_lstm.shape[0] + X_test_lstm.shape[0]},
        {"metric": "min_train_sequences_per_vm", "value": min(train_seqs_per_vm.values())},
        {"metric": "max_train_sequences_per_vm", "value": max(train_seqs_per_vm.values())},
        {"metric": "min_test_sequences_per_vm", "value": min(test_seqs_per_vm.values())},
        {"metric": "max_test_sequences_per_vm", "value": max(test_seqs_per_vm.values())},
        {"metric": "train_tensor_shape", "value": str(X_train_lstm.shape)},
        {"metric": "test_tensor_shape", "value": str(X_test_lstm.shape)},
        {"metric": "tensor_dtype", "value": str(X_train_lstm.dtype)},
        {"metric": "train_tensor_size_mb", "value": round(TRAIN_SEQUENCES_PATH.stat().st_size / (1024*1024), 2)},
        {"metric": "test_tensor_size_mb", "value": round(TEST_SEQUENCES_PATH.stat().st_size / (1024*1024), 2)},
        {"metric": "boundary_crossing_violations", "value": 0},
        {"metric": "temporal_leakage_violations", "value": 0},
    ]

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved sequence summary to: {SUMMARY_PATH}")

    # Final Execution Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("PHASE 10A EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Selected VM IDs                   : {selected_vms}")
    print(f"Number of Train Sequences         : {X_train_lstm.shape[0]:,}")
    print(f"Number of Test Sequences          : {X_test_lstm.shape[0]:,}")
    print(f"Total Sequences                   : {X_train_lstm.shape[0] + X_test_lstm.shape[0]:,}")
    print(f"Sequence Shape                    : {X_train_lstm.shape[1:]} (T={SEQUENCE_LENGTH}, features=5)")
    print(f"Train Sequences Per VM (min / max): {min(train_seqs_per_vm.values()):,} / {max(train_seqs_per_vm.values()):,}")
    print(f"Test Sequences Per VM (min / max) : {min(test_seqs_per_vm.values()):,} / {max(test_seqs_per_vm.values()):,}")
    print(f"Boundary Crossing Guarantee       : VERIFIED (0 sequences cross VM boundaries)")
    print(f"Train / Test Separation Guarantee : VERIFIED (100% chronological partition isolation)")
    print("\nGenerated Artifacts:")
    print(f"  - Selected VMs Table : {SELECTED_VMS_PATH}")
    print(f"  - X_train_lstm.npy   : {TRAIN_SEQUENCES_PATH} ({TRAIN_SEQUENCES_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - X_test_lstm.npy    : {TEST_SEQUENCES_PATH} ({TEST_SEQUENCES_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - train_metadata.csv : {TRAIN_META_PATH}")
    print(f"  - test_metadata.csv  : {TEST_META_PATH}")
    print(f"  - sequence_summary.csv: {SUMMARY_PATH}")
    print(f"\nExecution Runtime: {elapsed:.2f} seconds")
    print("=" * 75)


if __name__ == "__main__":
    main()
