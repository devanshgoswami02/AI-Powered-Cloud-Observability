"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 9B: Bitbrains fastStorage Preprocessing Pipeline

This script prepares the Bitbrains GWA-T-12 fastStorage VM telemetry for model training
and evaluation. It implements VM-level telemetry extraction, safe handling of zero-capacity
memory edge cases, log1p transformation of heavy-tailed I/O metrics, chronological 80/20
train/test splitting within each VM trace, and standardized scaling on training data only.

Outputs are isolated in `outputs/bitbrains/` without modifying any existing baseline assets.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ============================================================
# Paths and Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_RAW_DIR = Path(
    r"C:\Users\goswa\OneDrive\Desktop\PROJECTS\DOWNLOADS\gwa_t_12_fastStorage\fastStorage\2013-8"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT_PATH = OUTPUT_DIR / "X_train_bitbrains.csv"
TEST_OUTPUT_PATH = OUTPUT_DIR / "X_test_bitbrains.csv"
SCALER_OUTPUT_PATH = OUTPUT_DIR / "bitbrains_scaler.joblib"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "bitbrains_preprocessing_summary.csv"

RAW_COLUMNS = [
    "Timestamp [ms]",
    "CPU cores",
    "CPU capacity provisioned [MHZ]",
    "CPU usage [MHZ]",
    "CPU usage [%]",
    "Memory capacity provisioned [KB]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]"
]

FEATURE_COLUMNS = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io"
]

METADATA_COLUMNS = ["vm_id", "timestamp"]
ALL_OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

MIN_OBSERVATIONS_PER_VM = 100
TRAIN_SPLIT_RATIO = 0.80


# ============================================================
# Feature Extraction & Transformation Helpers
# ============================================================

def transform_vm_dataframe(df: pd.DataFrame, vm_id: str):
    """
    Extract and transform features for a single VM trace:
      1. Sort chronologically by timestamp.
      2. cpu_util_percent = CPU usage [%]
      3. mem_util_percent = (Memory usage / Memory capacity) * 100 with zero-capacity handling and 100.0 cap.
      4. net_in = log1p(Network received throughput [KB/s])
      5. net_out = log1p(Network transmitted throughput [KB/s])
      6. disk_io = log1p(Disk read throughput [KB/s] + Disk write throughput [KB/s])
    """
    # Sort chronologically
    df = df.sort_values("Timestamp [ms]").reset_index(drop=True)

    timestamps = df["Timestamp [ms]"].values.astype(np.int64)

    # 1. CPU utilization percent
    cpu_util = df["CPU usage [%]"].values.astype(np.float64)

    # 2. Memory utilization percent (handle capacity == 0 safely and cap at 100.0)
    mem_cap = df["Memory capacity provisioned [KB]"].values.astype(np.float64)
    mem_use = df["Memory usage [KB]"].values.astype(np.float64)

    zero_cap_mask = (mem_cap <= 0)
    zero_cap_count = int(np.sum(zero_cap_mask))

    mem_util = np.zeros_like(mem_cap, dtype=np.float64)
    valid_cap_mask = ~zero_cap_mask
    mem_util[valid_cap_mask] = (mem_use[valid_cap_mask] / mem_cap[valid_cap_mask]) * 100.0

    # Cap at 100.0%
    mem_util = np.clip(mem_util, 0.0, 100.0)

    # 3. Network received throughput (log1p)
    net_in_raw = df["Network received throughput [KB/s]"].values.astype(np.float64)
    net_in = np.log1p(np.maximum(net_in_raw, 0.0))

    # 4. Network transmitted throughput (log1p)
    net_out_raw = df["Network transmitted throughput [KB/s]"].values.astype(np.float64)
    net_out = np.log1p(np.maximum(net_out_raw, 0.0))

    # 5. Disk I/O throughput (read + write, log1p)
    disk_read = df["Disk read throughput [KB/s]"].values.astype(np.float64)
    disk_write = df["Disk write throughput [KB/s]"].values.astype(np.float64)
    disk_total = np.maximum(disk_read + disk_write, 0.0)
    disk_io = np.log1p(disk_total)

    # Assemble transformed feature array
    features_unscaled = np.column_stack([
        cpu_util,
        mem_util,
        net_in,
        net_out,
        disk_io
    ])

    return timestamps, features_unscaled, zero_cap_count


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("BITBRAINS GWA-T-12 PREPROCESSING PIPELINE (PHASE 9B)")
    print("=" * 70)

    if not BITBRAINS_RAW_DIR.exists():
        raise FileNotFoundError(f"Bitbrains directory not found at: {BITBRAINS_RAW_DIR}")

    csv_files = sorted(list(BITBRAINS_RAW_DIR.glob("*.csv")))
    num_files = len(csv_files)
    print(f"Total VM CSV files discovered: {num_files}")

    # ------------------------------------------------------------
    # Pass 1: Trace validation, chronological split, & scaler fit
    # ------------------------------------------------------------
    print("\n[Pass 1/2] Filtering short traces and fitting StandardScaler on training data...")

    scaler = StandardScaler()
    excluded_vms = []
    used_vms = []

    total_train_rows = 0
    total_test_rows = 0
    total_zero_cap_handled = 0

    vm_trace_info = []  # Store (fpath, vm_id, n_train, n_total) for Pass 2

    for idx, fpath in enumerate(csv_files, start=1):
        vm_id = fpath.stem

        # Read CSV with semicolon-tab delimiter
        try:
            df = pd.read_csv(
                fpath,
                sep=r";\s*",
                engine="python",
                header=0
            )
        except Exception as e:
            print(f"Warning: Failed to read {fpath.name}: {e}")
            excluded_vms.append((vm_id, f"read_error: {e}"))
            continue

        n_rows = len(df)
        if n_rows < MIN_OBSERVATIONS_PER_VM:
            excluded_vms.append((vm_id, f"too_short_{n_rows}_rows"))
            continue

        # Valid VM
        n_train = int(n_rows * TRAIN_SPLIT_RATIO)
        n_test = n_rows - n_train

        timestamps, features_unscaled, zero_cap_count = transform_vm_dataframe(df, vm_id)
        total_zero_cap_handled += zero_cap_count

        # Split features into train (first 80%)
        train_features = features_unscaled[:n_train]
        scaler.partial_fit(train_features)

        total_train_rows += n_train
        total_test_rows += n_test
        used_vms.append(vm_id)

        vm_trace_info.append((fpath, vm_id, n_train, n_rows))

        if idx % 250 == 0 or idx == num_files:
            print(f"  Processed {idx}/{num_files} VM files (Current train rows: {total_train_rows:,})...")

    num_used = len(used_vms)
    num_excluded = len(excluded_vms)
    print(f"\nPass 1 Complete:")
    print(f"  VMs used in modeling dataset : {num_used:,}")
    print(f"  VMs excluded (< {MIN_OBSERVATIONS_PER_VM} rows)    : {num_excluded:,}")
    print(f"  Total training rows (80%)    : {total_train_rows:,}")
    print(f"  Total test rows (20%)        : {total_test_rows:,}")
    print(f"  Total rows combined          : {total_train_rows + total_test_rows:,}")
    print(f"  Zero-capacity memory events  : {total_zero_cap_handled:,}")

    # ------------------------------------------------------------
    # Save Scaler
    # ------------------------------------------------------------
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"\nSaved Bitbrains StandardScaler: {SCALER_OUTPUT_PATH}")
    print("Scaler parameter summary:")
    for feat, mean, scale in zip(FEATURE_COLUMNS, scaler.mean_, scaler.scale_):
        print(f"  - {feat:18s}: mean = {mean:10.6f}, std = {scale:10.6f}")

    # ------------------------------------------------------------
    # Pass 2: Scale features and write X_train and X_test CSVs
    # ------------------------------------------------------------
    print("\n[Pass 2/2] Scaling features and streaming datasets to disk...")

    # Clear destination files if they exist
    if TRAIN_OUTPUT_PATH.exists():
        TRAIN_OUTPUT_PATH.unlink()
    if TEST_OUTPUT_PATH.exists():
        TEST_OUTPUT_PATH.unlink()

    total_missing_values = 0
    total_infinite_values = 0

    train_written = 0
    test_written = 0

    # Write in chunks file-by-file with header on first chunk
    write_train_header = True
    write_test_header = True

    for idx, (fpath, vm_id, n_train, n_rows) in enumerate(vm_trace_info, start=1):
        df = pd.read_csv(
            fpath,
            sep=r";\s*",
            engine="python",
            header=0
        )

        timestamps, features_unscaled, _ = transform_vm_dataframe(df, vm_id)

        # Apply StandardScaler fitted on training set
        features_scaled = scaler.transform(features_unscaled)

        # Check data integrity (NaN, Inf)
        nan_count = np.sum(np.isnan(features_scaled))
        inf_count = np.sum(np.isinf(features_scaled))
        total_missing_values += int(nan_count)
        total_infinite_values += int(inf_count)

        # Prepare train partition
        train_ts = timestamps[:n_train]
        train_feat = features_scaled[:n_train]
        train_df = pd.DataFrame(train_feat, columns=FEATURE_COLUMNS)
        train_df.insert(0, "timestamp", train_ts)
        train_df.insert(0, "vm_id", vm_id)

        # Prepare test partition
        test_ts = timestamps[n_train:]
        test_feat = features_scaled[n_train:]
        test_df = pd.DataFrame(test_feat, columns=FEATURE_COLUMNS)
        test_df.insert(0, "timestamp", test_ts)
        test_df.insert(0, "vm_id", vm_id)

        # Append to CSVs with float precision
        train_df.to_csv(
            TRAIN_OUTPUT_PATH,
            mode="a",
            header=write_train_header,
            index=False,
            float_format="%.6f"
        )
        write_train_header = False
        train_written += len(train_df)

        test_df.to_csv(
            TEST_OUTPUT_PATH,
            mode="a",
            header=write_test_header,
            index=False,
            float_format="%.6f"
        )
        write_test_header = False
        test_written += len(test_df)

        if idx % 250 == 0 or idx == len(vm_trace_info):
            print(f"  Written {idx}/{len(vm_trace_info)} VMs (Train rows: {train_written:,}, Test rows: {test_written:,})...")

    # ------------------------------------------------------------
    # Save Preprocessing Summary CSV
    # ------------------------------------------------------------
    summary_records = [
        {"metric": "vm_files_found", "value": num_files},
        {"metric": "vms_excluded", "value": num_excluded},
        {"metric": "vms_used", "value": num_used},
        {"metric": "min_observations_threshold", "value": MIN_OBSERVATIONS_PER_VM},
        {"metric": "train_split_ratio", "value": TRAIN_SPLIT_RATIO},
        {"metric": "training_rows", "value": train_written},
        {"metric": "test_rows", "value": test_written},
        {"metric": "total_rows", "value": train_written + test_written},
        {"metric": "missing_values", "value": total_missing_values},
        {"metric": "infinite_values", "value": total_infinite_values},
        {"metric": "memory_capacity_zero_handled", "value": total_zero_cap_handled},
    ]

    for feat, mean, scale in zip(FEATURE_COLUMNS, scaler.mean_, scaler.scale_):
        summary_records.append({"metric": f"scaler_mean_{feat}", "value": f"{mean:.6f}"})
        summary_records.append({"metric": f"scaler_std_{feat}", "value": f"{scale:.6f}"})

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"\nSaved preprocessing summary: {SUMMARY_OUTPUT_PATH}")

    # ------------------------------------------------------------
    # Final Output Summary
    # ------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE - EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Number of VM files found             : {num_files}")
    print(f"Number of VMs excluded (< 100 rows)  : {num_excluded}")
    print(f"Number of VMs used                   : {num_used}")
    print(f"Training rows (chronological 80%)    : {train_written:,}")
    print(f"Test rows (chronological 20%)        : {test_written:,}")
    print(f"Total rows                           : {train_written + test_written:,}")
    print(f"Feature names                        : {FEATURE_COLUMNS}")
    print(f"Missing values                       : {total_missing_values}")
    print(f"Infinite values                      : {total_infinite_values}")
    print(f"Memory-capacity-zero handling count  : {total_zero_cap_handled:,}")
    print("\nFitted StandardScaler Parameters:")
    for feat, mean, scale in zip(FEATURE_COLUMNS, scaler.mean_, scaler.scale_):
        print(f"  {feat:20s}: Mean = {mean:10.6f} | Std = {scale:10.6f}")

    print("\nGenerated Output Files:")
    print(f"  - Training Set : {TRAIN_OUTPUT_PATH} ({TRAIN_OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Test Set     : {TEST_OUTPUT_PATH} ({TEST_OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Scaler       : {SCALER_OUTPUT_PATH}")
    print(f"  - Summary      : {SUMMARY_OUTPUT_PATH}")
    print(f"\nTotal Pipeline Runtime: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
