import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from pathlib import Path


# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 7: Inference Pipeline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
SCALER_PATH = PROJECT_ROOT / "outputs" / "scaler.joblib"


FEATURES = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent"
]


# Threshold established during anomaly detection.
ANOMALY_THRESHOLD = 0.6037111008001094


# ------------------------------------------------------------
# 1. Load model and scaler
# ------------------------------------------------------------

print("=" * 60)
print("AI ANOMALY DETECTION - INFERENCE")
print("=" * 60)

print("\nLoading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading autoencoder...")
model = tf.keras.models.load_model(MODEL_PATH)


# ------------------------------------------------------------
# 2. Prediction function
# ------------------------------------------------------------

def predict_anomaly(
    cpu_util_percent,
    mem_util_percent,
    net_in,
    net_out,
    disk_io_percent
):

    data = pd.DataFrame(
        [[
            cpu_util_percent,
            mem_util_percent,
            net_in,
            net_out,
            disk_io_percent
        ]],
        columns=FEATURES
    )

    # Scale using the scaler fitted during training.
    scaled_data = scaler.transform(data)

    # Reconstruct using trained autoencoder.
    reconstructed = model.predict(
        scaled_data,
        verbose=0
    )

    # Calculate mean squared reconstruction error.
    reconstruction_error = np.mean(
        np.square(
            scaled_data - reconstructed
        )
    )

    # Classify.
    if reconstruction_error > ANOMALY_THRESHOLD:
        prediction = "ANOMALY"
    else:
        prediction = "NORMAL"

    return prediction, reconstruction_error


# ------------------------------------------------------------
# 3. Demonstration with sample observation
# ------------------------------------------------------------

sample = {
    "cpu_util_percent": 40.0,
    "mem_util_percent": 88.0,
    "net_in": 41.0,
    "net_out": 32.5,
    "disk_io_percent": 7.5
}


prediction, error = predict_anomaly(
    sample["cpu_util_percent"],
    sample["mem_util_percent"],
    sample["net_in"],
    sample["net_out"],
    sample["disk_io_percent"]
)


print("\nSample machine observation:")

for feature in FEATURES:
    print(
        f"{feature}: "
        f"{sample[feature]}"
    )


print("\nReconstruction error:")
print(f"{error:.6f}")

print("\nAnomaly threshold:")
print(f"{ANOMALY_THRESHOLD:.6f}")

print("\nPrediction:")
print(prediction)


# ------------------------------------------------------------
# 4. Completion
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)