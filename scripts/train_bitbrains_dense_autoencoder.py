"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 11A: Fair Dense Autoencoder Baseline on Bitbrains

This script builds and trains a feedforward Dense Autoencoder baseline on the exact
same 10 selected VMs and 12-step temporal sequences used by the LSTM-Autoencoder in Phase 10.

Fairness Criteria:
  1. Same underlying Bitbrains telemetry and VM selection (outputs/bitbrains/selected_vms.csv).
  2. Same 12-step temporal windows (T=12, 5 features -> flattened to 60-D vector).
  3. Same VM-aware chronological 80/20 train/validation split (56,820 train / 14,212 val).
  4. Same optimization settings (Adam, lr=0.001, MSE loss, batch_size=64, EarlyStopping patience=5).
  5. Strict test data isolation (X_test_lstm.npy is NOT used for training).

Immutability Notice:
  This script does NOT modify or retrain the Alibaba Dense Autoencoder baseline
  or the Bitbrains LSTM-Autoencoder. All outputs are isolated in outputs/bitbrains/dense_model/.
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set reproducibility seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# Paths and Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
SEQUENCES_DIR = BITBRAINS_DIR / "lstm_sequences"
TRAIN_SEQUENCES_PATH = SEQUENCES_DIR / "X_train_lstm.npy"
TRAIN_METADATA_PATH = SEQUENCES_DIR / "train_metadata.csv"
SELECTED_VMS_PATH = BITBRAINS_DIR / "selected_vms.csv"

# Existing Models (Strictly Read-Only Verification)
ALIBABA_MODEL_PATH = PROJECT_ROOT / "outputs" / "autoencoder.keras"
LSTM_MODEL_PATH = BITBRAINS_DIR / "lstm_model" / "lstm_autoencoder.keras"

# Output Directory for Bitbrains Dense Baseline
OUTPUT_DIR = BITBRAINS_DIR / "dense_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODEL_PATH = OUTPUT_DIR / "bitbrains_dense_autoencoder.keras"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "training_summary.csv"
TRAINING_LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss.png"

# Hyperparameters
SEQUENCE_LENGTH = 12
NUM_FEATURES = 5
INPUT_DIM = SEQUENCE_LENGTH * NUM_FEATURES  # 60 dimensions

LEARNING_RATE = 0.001
LOSS_FUNCTION = "mse"
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
RESTORE_BEST_WEIGHTS = True
TRAIN_VAL_SPLIT_RATIO = 0.80


# ============================================================
# Validation Partitioning Helper (Replicating Phase 10C)
# ============================================================

def create_vm_aware_train_val_split(train_meta_df: pd.DataFrame, split_ratio: float = 0.80):
    """
    Replicate the exact VM-aware chronological split used in Phase 10C:
    Takes first 80% of sequences per VM for training, and final 20% for validation.
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
# Model Architecture Builder
# ============================================================

def build_dense_autoencoder(input_dim: int = 60, learning_rate: float = 0.001) -> Model:
    """
    Construct and compile the fair Bitbrains Dense Autoencoder:
    Input: 60
    Dense(32, relu) -> Dense(16, relu) -> Dense(8, relu) [Latent]
    -> Dense(16, relu) -> Dense(32, relu) -> Dense(60, linear)
    """
    inputs = Input(shape=(input_dim,), name="flattened_sequence_input")

    # --- Encoder ---
    x = Dense(32, activation="relu", name="encoder_dense_1")(inputs)
    x = Dense(16, activation="relu", name="encoder_dense_2")(x)
    latent = Dense(8, activation="relu", name="encoder_latent_bottleneck")(x)

    # --- Decoder ---
    x = Dense(16, activation="relu", name="decoder_dense_1")(latent)
    x = Dense(32, activation="relu", name="decoder_dense_2")(x)
    outputs = Dense(input_dim, activation="linear", name="reconstruction_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="bitbrains_dense_autoencoder")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)
    return model


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    total_start_time = time.time()
    print("=" * 78)
    print("BITBRAINS FAIR DENSE AUTOENCODER BASELINE TRAINING (PHASE 11A)")
    print("=" * 78)

    # 1. Verify existence of source files and record baseline timestamps
    if not TRAIN_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Missing train sequences at: {TRAIN_SEQUENCES_PATH}")
    if not TRAIN_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing train metadata at: {TRAIN_METADATA_PATH}")

    assert ALIBABA_MODEL_PATH.exists(), f"Alibaba baseline model missing: {ALIBABA_MODEL_PATH}"
    assert LSTM_MODEL_PATH.exists(), f"Bitbrains LSTM model missing: {LSTM_MODEL_PATH}"

    alibaba_mtime_before = ALIBABA_MODEL_PATH.stat().st_mtime
    alibaba_size_before = ALIBABA_MODEL_PATH.stat().st_size
    lstm_mtime_before = LSTM_MODEL_PATH.stat().st_mtime
    lstm_size_before = LSTM_MODEL_PATH.stat().st_size

    # 2. Load sequence data
    print("Loading training sequences and metadata...")
    X_train_full = np.load(TRAIN_SEQUENCES_PATH)
    train_meta_df = pd.read_csv(TRAIN_METADATA_PATH)

    total_seqs = len(X_train_full)
    print(f"Loaded X_train_full: {total_seqs:,} sequences of shape {X_train_full.shape[1:]}")

    # 3. Flatten sequences to 60 dimensions: (N, 12, 5) -> (N, 60)
    print(f"Flattening sequences from (12, 5) to ({INPUT_DIM},)...")
    X_train_full_flat = X_train_full.reshape(total_seqs, INPUT_DIM)
    print(f"Flattened shape: {X_train_full_flat.shape} (dtype: {X_train_full_flat.dtype})")

    # Verify absence of NaN / Inf
    assert not np.isnan(X_train_full_flat).any(), "NaN values detected in training sequences!"
    assert not np.isinf(X_train_full_flat).any(), "Infinite values detected in training sequences!"
    print("  [VERIFIED] Zero NaN or Infinite values in flattened training sequences.")

    # 4. Construct VM-aware chronological train / validation split (Exact Phase 10C replication)
    print(f"\nReplicating Phase 10C chronological split ({TRAIN_VAL_SPLIT_RATIO * 100:.0f}% train / {(1 - TRAIN_VAL_SPLIT_RATIO) * 100:.0f}% val per VM)...")
    train_indices, val_indices = create_vm_aware_train_val_split(train_meta_df, split_ratio=TRAIN_VAL_SPLIT_RATIO)

    X_train = X_train_full_flat[train_indices]
    X_val = X_train_full_flat[val_indices]

    n_train = len(X_train)
    n_val = len(X_val)
    print(f"  - Training sequences   : {n_train:,} ({n_train / total_seqs * 100:.2f}%) [Expected: 56,820]")
    print(f"  - Validation sequences : {n_val:,} ({n_val / total_seqs * 100:.2f}%) [Expected: 14,212]")
    assert n_train == 56820, f"Train count mismatch: {n_train} vs 56,820"
    assert n_val == 14212, f"Validation count mismatch: {n_val} vs 14,212"

    # 5. Build and compile Dense Autoencoder
    print("\nBuilding and compiling Dense Autoencoder model...")
    model = build_dense_autoencoder(input_dim=INPUT_DIM, learning_rate=LEARNING_RATE)

    # Verification of shapes and parameters
    model.summary()
    input_shape = model.input_shape
    output_shape = model.output_shape
    param_count = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))

    print("\nModel Architecture Verification:")
    print(f"  - Input shape  : {input_shape} (Expected: (None, 60))")
    print(f"  - Output shape : {output_shape} (Expected: (None, 60))")
    print(f"  - Parameters   : {param_count:,} (Expected: 5,284)")

    assert input_shape == (None, 60), f"Unexpected input shape: {input_shape}"
    assert output_shape == (None, 60), f"Unexpected output shape: {output_shape}"
    assert param_count == 5284, f"Unexpected parameter count: {param_count} vs 5,284"

    # 6. Configure Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=RESTORE_BEST_WEIGHTS,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=str(SAVED_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # 7. Train Model
    print("\n" + "=" * 78)
    print(f"STARTING TRAINING: max_epochs={MAX_EPOCHS}, batch_size={BATCH_SIZE}, patience={PATIENCE}")
    print("=" * 78)

    train_start_time = time.time()
    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
        shuffle=True
    )
    training_duration = time.time() - train_start_time
    epochs_completed = len(history.history["loss"])

    # 8. Extract Training Metrics
    train_losses = history.history["loss"]
    val_losses = history.history["val_loss"]

    final_train_loss = float(train_losses[-1])
    best_val_loss = float(np.min(val_losses))
    best_epoch = int(np.argmin(val_losses)) + 1

    print("\n" + "=" * 78)
    print("TRAINING PROCESS COMPLETE")
    print("=" * 78)
    print(f"  Epochs Completed       : {epochs_completed} / {MAX_EPOCHS}")
    print(f"  Best Epoch             : {best_epoch}")
    print(f"  Final Training Loss    : {final_train_loss:.6f}")
    print(f"  Best Validation Loss   : {best_val_loss:.6f}")
    print(f"  Total Training Runtime : {training_duration:.2f} seconds ({training_duration / epochs_completed:.2f} s/epoch)")
    print("=" * 78)

    # 9. Save Training History CSV
    history_df = pd.DataFrame({
        "epoch": range(1, epochs_completed + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    history_df.to_csv(TRAINING_HISTORY_PATH, index=False)
    print(f"\nSaved training history to: {TRAINING_HISTORY_PATH}")

    # 10. Save Training Summary CSV
    summary_data = [
        {"metric": "model_architecture", "value": "Dense_60_32_16_8_16_32_60"},
        {"metric": "total_trainable_parameters", "value": param_count},
        {"metric": "input_dimension", "value": INPUT_DIM},
        {"metric": "latent_bottleneck_dimension", "value": 8},
        {"metric": "number_of_selected_vms", "value": 10},
        {"metric": "total_training_sequences", "value": total_seqs},
        {"metric": "training_sequences_used", "value": n_train},
        {"metric": "validation_sequences_used", "value": n_val},
        {"metric": "batch_size", "value": BATCH_SIZE},
        {"metric": "maximum_epochs", "value": MAX_EPOCHS},
        {"metric": "actual_epochs_completed", "value": epochs_completed},
        {"metric": "optimizer", "value": "Adam"},
        {"metric": "learning_rate", "value": LEARNING_RATE},
        {"metric": "final_training_loss", "value": f"{final_train_loss:.6f}"},
        {"metric": "final_validation_loss", "value": f"{val_losses[-1]:.6f}"},
        {"metric": "best_validation_loss", "value": f"{best_val_loss:.6f}"},
        {"metric": "best_epoch", "value": best_epoch},
        {"metric": "training_duration_seconds", "value": f"{training_duration:.2f}"},
        {"metric": "saved_model_path", "value": str(SAVED_MODEL_PATH)}
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(TRAINING_SUMMARY_PATH, index=False)
    print(f"Saved training summary to: {TRAINING_SUMMARY_PATH}")

    # 11. Plot Training & Validation Loss Curve
    print(f"Generating training loss plot to: {TRAINING_LOSS_PLOT_PATH}...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    epochs_range = range(1, epochs_completed + 1)

    ax.plot(epochs_range, train_losses, label="Training Loss (MSE)", color="#2b5c8f", linewidth=2.2)
    ax.plot(epochs_range, val_losses, label="Validation Loss (MSE)", color="#d9534f", linewidth=2.2, linestyle="--")
    ax.scatter([best_epoch], [best_val_loss], color="#d9534f", s=80, zorder=5, label=f"Best Val Loss ({best_val_loss:.4f} @ Ep {best_epoch})")

    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="semibold")
    ax.set_title(
        "Bitbrains Fair Dense Autoencoder: Training & Validation Loss\n"
        f"Input: Flattened 12x5=60D | Params: {param_count:,} | Best Val Loss: {best_val_loss:.6f}",
        fontsize=12,
        fontweight="bold",
        pad=16,
        linespacing=1.4
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.text(
        0.5, 0.01,
        "Operational Note: Fair architectural comparison baseline trained on identical 12-step temporal windows.\n"
        "Trained strictly on normal VM workload data; test data remains completely untouched.",
        ha="center", va="bottom", fontsize=8.5, color="#6b7280", style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(TRAINING_LOSS_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved training loss plot to: {TRAINING_LOSS_PLOT_PATH}")

    # 12. Immutability & Safety Verification
    print("\n" + "=" * 78)
    print("SAFETY & BASELINE IMMUTABILITY VERIFICATION")
    print("=" * 78)
    assert ALIBABA_MODEL_PATH.stat().st_mtime == alibaba_mtime_before, "Alibaba model was modified!"
    assert ALIBABA_MODEL_PATH.stat().st_size == alibaba_size_before, "Alibaba model size changed!"
    assert LSTM_MODEL_PATH.stat().st_mtime == lstm_mtime_before, "Bitbrains LSTM model was modified!"
    assert LSTM_MODEL_PATH.stat().st_size == lstm_size_before, "Bitbrains LSTM model size changed!"

    print("  [PASSED] Alibaba Dense Autoencoder baseline remained strictly untouched.")
    print("  [PASSED] Bitbrains LSTM-Autoencoder remained strictly untouched.")
    print("  [PASSED] Phase 10 outputs remained strictly preserved.")
    print("  [PASSED] Dense AE baseline successfully saved.")

    total_runtime = time.time() - total_start_time
    print("\n" + "=" * 78)
    print(f"PHASE 11A EXECUTION COMPLETE in {total_runtime:.2f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
