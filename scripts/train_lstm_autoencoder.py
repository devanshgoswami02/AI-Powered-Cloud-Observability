"""
AI-Driven DevOps Anomaly Detection and Predictive Maintenance
Phase 10C: Bitbrains LSTM-Autoencoder Training Pipeline

This script trains the Sequence-to-Sequence (Seq2Seq) recurrent LSTM-Autoencoder
on the multi-VM sequence telemetry generated in Phase 10A.

Model Architecture:
  Input: (batch_size, 12, 5)
  -> LSTM(32, activation='tanh', return_sequences=True)
  -> LSTM(16, activation='tanh', return_sequences=False) [Latent Bottleneck]
  -> RepeatVector(12)
  -> LSTM(16, activation='tanh', return_sequences=True)
  -> LSTM(32, activation='tanh', return_sequences=True)
  -> TimeDistributed(Dense(5, activation='linear')) [Reconstructed Output]

Validation Split Strategy:
  To prevent bias and preserve VM boundaries/chronological order, Keras validation_split=0.2
  is NOT blindly applied to the full array. Instead, an exact chronological 80/20 split
  is constructed within each selected VM's training sequences:
    - First 80% of each VM's training sequences -> Training partition (56,820 sequences)
    - Last 20% of each VM's training sequences -> Validation partition (14,212 sequences)
  Test sequences (X_test_lstm.npy) remain completely isolated for downstream evaluation.

Training Configuration:
  - Optimizer: Adam (lr = 0.001)
  - Loss: Mean Squared Error (MSE)
  - Batch size: 64
  - Max epochs: 30
  - EarlyStopping: patience = 5, restore_best_weights = True

Outputs Saved Under `outputs/bitbrains/lstm_model/`:
  - lstm_autoencoder.keras
  - training_history.csv
  - training_summary.csv
  - training_loss.png
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# Reproducibility Seeds
# ============================================================

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Paths and Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BITBRAINS_DIR = PROJECT_ROOT / "outputs" / "bitbrains"
SEQUENCES_DIR = BITBRAINS_DIR / "lstm_sequences"

TRAIN_SEQUENCES_PATH = SEQUENCES_DIR / "X_train_lstm.npy"
TEST_SEQUENCES_PATH = SEQUENCES_DIR / "X_test_lstm.npy"
TRAIN_METADATA_PATH = SEQUENCES_DIR / "train_metadata.csv"
SELECTED_VMS_PATH = BITBRAINS_DIR / "selected_vms.csv"

OUTPUT_DIR = BITBRAINS_DIR / "lstm_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODEL_PATH = OUTPUT_DIR / "lstm_autoencoder.keras"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "training_summary.csv"
TRAINING_LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss.png"

SEQUENCE_LENGTH = 12
NUM_FEATURES = 5
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_FEATURES)

LEARNING_RATE = 0.001
LOSS_FUNCTION = "mse"
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
RESTORE_BEST_WEIGHTS = True
TRAIN_VAL_SPLIT_RATIO = 0.80


# ============================================================
# Model Architecture Builder
# ============================================================

def build_lstm_autoencoder(
    sequence_length: int = 12,
    num_features: int = 5,
    learning_rate: float = 0.001
) -> Model:
    """
    Construct and compile the verified Seq2Seq LSTM-Autoencoder.
    """
    inputs = Input(
        shape=(sequence_length, num_features),
        name="sequence_input"
    )

    # --- Encoder ---
    x = LSTM(
        units=32,
        activation="tanh",
        return_sequences=True,
        name="encoder_lstm_1"
    )(inputs)

    latent = LSTM(
        units=16,
        activation="tanh",
        return_sequences=False,
        name="encoder_latent"
    )(x)

    # --- Latent Bottleneck ---
    x = RepeatVector(
        n=sequence_length,
        name="latent_repeat"
    )(latent)

    # --- Decoder ---
    x = LSTM(
        units=16,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_1"
    )(x)

    x = LSTM(
        units=32,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_2"
    )(x)

    # --- Reconstruction Output ---
    outputs = TimeDistributed(
        Dense(units=num_features, activation="linear"),
        name="reconstructed_output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Bitbrains_LSTM_Autoencoder"
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    return model


# ============================================================
# Validation Partitioning Helper
# ============================================================

def create_vm_aware_train_val_split(train_meta_df: pd.DataFrame, split_ratio: float = 0.80):
    """
    Construct train and validation index sets by taking the first 80% of sequences
    for each VM as training, and the final 20% as validation.
    Preserves VM boundaries and chronological order without dataset leakage.
    """
    train_indices = []
    val_indices = []

    for vm_id, group in train_meta_df.groupby("vm_id", sort=False):
        indices = group["sequence_index"].values
        n_seqs = len(indices)
        n_train_sub = int(n_seqs * split_ratio)

        train_indices.extend(indices[:n_train_sub])
        val_indices.extend(indices[n_train_sub:])

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)

    return train_indices, val_indices


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    total_start_time = time.time()
    print("=" * 75)
    print("BITBRAINS LSTM-AUTOENCODER TRAINING PIPELINE (PHASE 10C)")
    print("=" * 75)

    # 1. Verification of sequence data
    if not TRAIN_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Missing training sequences at: {TRAIN_SEQUENCES_PATH}")
    if not TRAIN_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata at: {TRAIN_METADATA_PATH}")
    if not TEST_SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Missing test sequences at: {TEST_SEQUENCES_PATH}")

    print("Loading training sequences and metadata...")
    X_train_full = np.load(TRAIN_SEQUENCES_PATH)
    train_meta_df = pd.read_csv(TRAIN_METADATA_PATH)

    num_selected_vms = train_meta_df["vm_id"].nunique()
    total_train_sequences = len(X_train_full)
    print(f"Loaded {total_train_sequences:,} sequences across {num_selected_vms} VMs of shape {X_train_full.shape[1:]}")

    # 2. Construct VM-aware chronological train / validation split
    print(f"\nConstructing VM-aware chronological split ({TRAIN_VAL_SPLIT_RATIO * 100:.0f}% train / {(1 - TRAIN_VAL_SPLIT_RATIO) * 100:.0f}% val per VM)...")
    train_indices, val_indices = create_vm_aware_train_val_split(train_meta_df, split_ratio=TRAIN_VAL_SPLIT_RATIO)

    X_train = X_train_full[train_indices]
    X_val = X_train_full[val_indices]

    print(f"  - Training sequences used   : {len(X_train):,} ({len(X_train) / total_train_sequences * 100:.2f}%)")
    print(f"  - Validation sequences used : {len(X_val):,} ({len(X_val) / total_train_sequences * 100:.2f}%)")
    assert len(X_train) + len(X_val) == total_train_sequences, "Mismatch in train/val sequence counts!"

    # 3. Build & compile model
    print("\nBuilding and compiling LSTM-Autoencoder...")
    model = build_lstm_autoencoder(
        sequence_length=SEQUENCE_LENGTH,
        num_features=NUM_FEATURES,
        learning_rate=LEARNING_RATE
    )
    model.summary()

    trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    print(f"\nTotal Trainable Parameters: {trainable_params:,}")

    # 4. Configure Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=RESTORE_BEST_WEIGHTS,
        verbose=1
    )

    # 5. Train Model
    print("\n" + "-" * 75)
    print(f"Starting model training: batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
    print("-" * 75)

    training_start_time = time.time()
    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=1
    )
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in: {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")

    # 6. Extract Training Metrics
    loss_history = history.history["loss"]
    val_loss_history = history.history["val_loss"]
    epochs_completed = len(loss_history)

    final_train_loss = float(loss_history[-1])
    final_val_loss = float(val_loss_history[-1])
    best_val_loss = float(min(val_loss_history))
    best_epoch = int(np.argmin(val_loss_history) + 1)

    print("\nTraining Metrics Summary:")
    print(f"  - Actual Epochs Completed : {epochs_completed}")
    print(f"  - Final Training Loss     : {final_train_loss:.6f}")
    print(f"  - Final Validation Loss   : {final_val_loss:.6f}")
    print(f"  - Best Validation Loss    : {best_val_loss:.6f} (at Epoch {best_epoch})")

    # 7. Save Trained Model
    print(f"\nSaving trained model to: {SAVED_MODEL_PATH}")
    model.save(SAVED_MODEL_PATH)

    # 8. Save Training History CSV
    history_df = pd.DataFrame({
        "epoch": list(range(1, epochs_completed + 1)),
        "loss": loss_history,
        "val_loss": val_loss_history
    })
    history_df.to_csv(TRAINING_HISTORY_PATH, index=False)
    print(f"Saved training history to: {TRAINING_HISTORY_PATH}")

    # 9. Save Training Summary CSV
    summary_records = [
        {"metric": "number_of_selected_vms", "value": num_selected_vms},
        {"metric": "total_training_sequences", "value": total_train_sequences},
        {"metric": "training_sequences_used", "value": len(X_train)},
        {"metric": "validation_sequences", "value": len(X_val)},
        {"metric": "sequence_length", "value": SEQUENCE_LENGTH},
        {"metric": "number_of_features", "value": NUM_FEATURES},
        {"metric": "batch_size", "value": BATCH_SIZE},
        {"metric": "maximum_epochs", "value": MAX_EPOCHS},
        {"metric": "actual_epochs_completed", "value": epochs_completed},
        {"metric": "optimizer", "value": "Adam"},
        {"metric": "learning_rate", "value": LEARNING_RATE},
        {"metric": "final_training_loss", "value": f"{final_train_loss:.6f}"},
        {"metric": "final_validation_loss", "value": f"{final_val_loss:.6f}"},
        {"metric": "best_validation_loss", "value": f"{best_val_loss:.6f}"},
        {"metric": "best_epoch", "value": best_epoch},
        {"metric": "total_trainable_parameters", "value": trainable_params},
        {"metric": "training_duration_seconds", "value": round(training_duration, 2)},
        {"metric": "saved_model_path", "value": str(SAVED_MODEL_PATH)}
    ]
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(TRAINING_SUMMARY_PATH, index=False)
    print(f"Saved training summary to: {TRAINING_SUMMARY_PATH}")

    # 10. Generate Training-Loss Plot
    print(f"Generating training loss plot to: {TRAINING_LOSS_PLOT_PATH}...")
    plt.figure(figsize=(9, 5))
    plt.plot(
        history_df["epoch"],
        history_df["loss"],
        marker="o",
        linewidth=2,
        label="Training Loss (MSE)"
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        marker="s",
        linewidth=2,
        label="Validation Loss (MSE)"
    )
    plt.axvline(
        best_epoch,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"Best Model Epoch ({best_epoch})"
    )
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Reconstruction Error (MSE)", fontsize=11)
    plt.title("Bitbrains LSTM-Autoencoder Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(TRAINING_LOSS_PLOT_PATH, dpi=300)
    plt.close()
    print("Saved training loss plot.")

    # 11. Post-Training Validation: Reload and Test Inference Check
    print("\n" + "-" * 75)
    print("POST-TRAINING VERIFICATION & MODEL RELOAD")
    print("-" * 75)

    print(f"Reloading model from: {SAVED_MODEL_PATH}...")
    reloaded_model = tf.keras.models.load_model(SAVED_MODEL_PATH)

    print(f"Loading test sequences from: {TEST_SEQUENCES_PATH} (mmap_mode='r')...")
    X_test_memmap = np.load(TEST_SEQUENCES_PATH, mmap_mode="r")
    print(f"Total test sequences: {X_test_memmap.shape[0]:,} | Shape: {X_test_memmap.shape}")

    # Slice single test sequence
    test_sample = X_test_memmap[0:1]  # shape: (1, 12, 5)
    print(f"Extracted test sample shape: {test_sample.shape} (dtype: {test_sample.dtype})")

    # Run forward pass
    reconstruction = reloaded_model.predict(test_sample, verbose=0)
    print(f"Reconstructed test output shape: {reconstruction.shape} (dtype: {reconstruction.dtype})")

    assert reconstruction.shape == (1, SEQUENCE_LENGTH, NUM_FEATURES), (
        f"Shape mismatch: expected (1, {SEQUENCE_LENGTH}, {NUM_FEATURES}), got {reconstruction.shape}"
    )
    assert not np.isnan(reconstruction).any(), "NaN found in post-training test reconstruction!"
    assert not np.isinf(reconstruction).any(), "Inf found in post-training test reconstruction!"

    sample_mse = float(np.mean(np.square(test_sample - reconstruction)))
    print(f"Sample Test Reconstruction MSE: {sample_mse:.6f}")

    print("\nPost-Training Verifications Passed:")
    print("  [PASSED] Successfully reloaded trained model from disk")
    print("  [PASSED] Model produced exact expected output shape (1, 12, 5)")
    print("  [PASSED] Reconstructed values contain zero NaN or Inf values")

    # Final Execution Wrap-up
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 75)
    print("PHASE 10C EXECUTION COMPLETE - TRAINING REPORT")
    print("=" * 75)
    print("Training Status           : COMPLETED SUCCESSFULLY")
    print(f"Saved Model Path          : {SAVED_MODEL_PATH}")
    print(f"Actual Epochs Completed   : {epochs_completed} / {MAX_EPOCHS}")
    print(f"Best Validation Loss      : {best_val_loss:.6f} (at Epoch {best_epoch})")
    print(f"Final Validation Loss     : {final_val_loss:.6f}")
    print(f"Final Training Loss       : {final_train_loss:.6f}")
    print(f"Total Trainable Parameters: {trainable_params:,}")
    print(f"Total Pipeline Duration   : {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    print("=" * 75)


if __name__ == "__main__":
    main()
