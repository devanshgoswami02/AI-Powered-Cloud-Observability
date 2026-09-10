"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 2: Feature-Level Anomaly Explanation and Operational Severity Assessment

Operational Note:
Severity classifications (NORMAL, WARNING, CRITICAL) are operational deviation tiers
derived from empirical reconstruction-error thresholds of the Dense Autoencoder baseline.
They represent reconstruction error deviation tiers and do NOT represent actual failure
probability or time-to-failure.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ============================================================
# Paths and Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
SCALER_PATH = PROJECT_ROOT / "outputs" / "scaler.joblib"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "anomaly_detection_results.csv"

FEATURES = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent"
]

FEATURE_LABELS = {
    "cpu_util_percent": "CPU Utilization (%)",
    "mem_util_percent": "Memory Utilization (%)",
    "net_in": "Network In (KB/s)",
    "net_out": "Network Out (KB/s)",
    "disk_io_percent": "Disk I/O (%)"
}

# Baseline frozen 95th-percentile anomaly threshold
ANOMALY_THRESHOLD = 0.6037111008001094

# Exact 99th-percentile operational critical threshold calculated from outputs/anomaly_detection_results.csv
CRITICAL_THRESHOLD = 1.4040703128174425


# ============================================================
# Core Functions
# ============================================================

def load_artifacts():
    """Load the frozen scaler and autoencoder model."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return scaler, model


def classify_severity(reconstruction_error: float) -> str:
    """
    Classify operational severity based on transparent empirical reconstruction error thresholds:
      - NORMAL: error <= ANOMALY_THRESHOLD (within 95th percentile baseline)
      - WARNING: ANOMALY_THRESHOLD < error <= CRITICAL_THRESHOLD (95th-99th percentile anomaly)
      - CRITICAL: error > CRITICAL_THRESHOLD (top 1% extreme anomaly tail)

    Notice: This represents an operational anomaly severity tier based on model reconstruction
    error and does NOT represent actual system failure probability.
    """
    if reconstruction_error <= ANOMALY_THRESHOLD:
        return "NORMAL"
    elif reconstruction_error <= CRITICAL_THRESHOLD:
        return "WARNING"
    else:
        return "CRITICAL"


def explain_anomaly(
    cpu_util_percent: float,
    mem_util_percent: float,
    net_in: float,
    net_out: float,
    disk_io_percent: float,
    scaler=None,
    model=None
) -> dict:
    """
    Compute anomaly score, feature-level squared reconstruction error contributions,
    and operational severity for a single 5-feature system observation.
    """
    if scaler is None or model is None:
        loaded_scaler, loaded_model = load_artifacts()
        scaler = scaler or loaded_scaler
        model = model or loaded_model

    raw_data = pd.DataFrame(
        [[cpu_util_percent, mem_util_percent, net_in, net_out, disk_io_percent]],
        columns=FEATURES
    )

    # Scale using existing frozen baseline scaler
    scaled_data = scaler.transform(raw_data)

    # Reconstruct using trained autoencoder
    reconstructed_data = model.predict(scaled_data, verbose=0)

    # Per-feature squared reconstruction error: (original_scaled - reconstructed_scaled)^2
    squared_errors = np.square(scaled_data - reconstructed_data)[0]

    # Total reconstruction MSE across all 5 features
    reconstruction_mse = float(np.mean(squared_errors))
    total_squared_error = float(np.sum(squared_errors))

    # Binary anomaly status using frozen baseline 95th-percentile threshold
    status = "ANOMALY" if reconstruction_mse > ANOMALY_THRESHOLD else "NORMAL"

    # Operational severity tier
    severity = classify_severity(reconstruction_mse)

    # Calculate individual feature contributions and rank descending
    feature_contributions = []
    for feat, sq_err in zip(FEATURES, squared_errors):
        percentage = (float(sq_err) / total_squared_error * 100.0) if total_squared_error > 0 else 0.0
        feature_contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "squared_error": float(sq_err),
            "percentage": percentage
        })

    # Sort descending by squared reconstruction error (consistent with MSE anomaly score)
    feature_contributions.sort(key=lambda x: x["squared_error"], reverse=True)

    return {
        "inputs": {
            "cpu_util_percent": cpu_util_percent,
            "mem_util_percent": mem_util_percent,
            "net_in": net_in,
            "net_out": net_out,
            "disk_io_percent": disk_io_percent
        },
        "reconstruction_error": reconstruction_mse,
        "detection_threshold": ANOMALY_THRESHOLD,
        "critical_threshold": CRITICAL_THRESHOLD,
        "status": status,
        "severity": severity,
        "feature_contributions": feature_contributions
    }


def print_explanation(result: dict, title: str = None):
    """Print the explanation report formatted to specification."""
    if title:
        print("=" * 60)
        print(title)
        print("=" * 60)

    inputs = result["inputs"]
    print("\nInput Metrics:")
    print(f"CPU: {inputs['cpu_util_percent']:.2f}")
    print(f"Memory: {inputs['mem_util_percent']:.2f}")
    print(f"Network In: {inputs['net_in']:.2f}")
    print(f"Network Out: {inputs['net_out']:.2f}")
    print(f"Disk I/O: {inputs['disk_io_percent']:.2f}")

    print(f"\nReconstruction Error: {result['reconstruction_error']:.6f}")
    print(f"Detection Threshold: {result['detection_threshold']:.6f}")

    print(f"\nStatus: {result['status']}")
    print(f"Severity: {result['severity']}")

    print("\nFeature Contributions:")
    for idx, item in enumerate(result["feature_contributions"], start=1):
        feat_name = item["feature"]
        sq_err = item["squared_error"]
        pct = item["percentage"]
        print(f"{idx}. {feat_name}: squared_error = {sq_err:.6f} ({pct:.2f}%)")

    print("\nOperational Note:")
    print("Severity represents operational deviation based on reconstruction error.")
    print("It does NOT indicate system failure probability.")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature-Level Anomaly Explanation and Operational Severity"
    )
    parser.add_argument("--cpu", type=float, help="CPU utilization percent")
    parser.add_argument("--mem", type=float, help="Memory utilization percent")
    parser.add_argument("--net-in", type=float, help="Network In")
    parser.add_argument("--net-out", type=float, help="Network Out")
    parser.add_argument("--disk-io", type=float, help="Disk I/O percent")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with normal and extreme test inputs"
    )

    args = parser.parse_args()

    # Load artifacts once
    scaler, model = load_artifacts()

    # If all custom inputs provided, evaluate that observation
    if (
        args.cpu is not None and
        args.mem is not None and
        args.net_in is not None and
        args.net_out is not None and
        args.disk_io is not None
    ):
        result = explain_anomaly(
            args.cpu, args.mem, args.net_in, args.net_out, args.disk_io,
            scaler=scaler, model=model
        )
        print_explanation(result, title="ANOMALY EXPLANATION & OPERATIONAL SEVERITY")
        return

    # Default / Demo Mode: Run benchmark normal and extreme anomaly inputs
    print("=" * 60)
    print("AI ANOMALY EXPLANATION & OPERATIONAL SEVERITY")
    print("=" * 60)
    print(f"Baseline Anomaly Threshold (95th percentile): {ANOMALY_THRESHOLD:.16f}")
    print(f"Critical Severity Threshold (99th percentile): {CRITICAL_THRESHOLD:.16f}")

    # Case 1: Normal test input (typical operating observation from inference pipeline)
    normal_input = {
        "cpu_util_percent": 40.0,
        "mem_util_percent": 88.0,
        "net_in": 41.0,
        "net_out": 32.5,
        "disk_io_percent": 7.5
    }
    normal_result = explain_anomaly(
        normal_input["cpu_util_percent"],
        normal_input["mem_util_percent"],
        normal_input["net_in"],
        normal_input["net_out"],
        normal_input["disk_io_percent"],
        scaler=scaler,
        model=model
    )
    print_explanation(normal_result, title="BENCHMARK 1: NORMAL TEST INPUT")

    print("\n" + "-" * 60 + "\n")

    # Case 2: Extreme anomaly input (severe multi-resource saturation)
    extreme_input = {
        "cpu_util_percent": 98.0,
        "mem_util_percent": 95.0,
        "net_in": 150.0,
        "net_out": 120.0,
        "disk_io_percent": 85.0
    }
    extreme_result = explain_anomaly(
        extreme_input["cpu_util_percent"],
        extreme_input["mem_util_percent"],
        extreme_input["net_in"],
        extreme_input["net_out"],
        extreme_input["disk_io_percent"],
        scaler=scaler,
        model=model
    )
    print_explanation(extreme_result, title="BENCHMARK 2: EXTREME ANOMALY INPUT")


if __name__ == "__main__":
    main()
