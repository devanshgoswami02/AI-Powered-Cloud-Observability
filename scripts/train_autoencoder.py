import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# AI-Driven Anomaly Detection and Predictive Maintenance
# Phase 3: Autoencoder Training
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "outputs" / "X_train_scaled.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Load scaled training data
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH, index_col=0)

X_train = train_df.values

print("=" * 60)
print("AUTOENCODER TRAINING")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)


# ------------------------------------------------------------
# 2. Build autoencoder
# ------------------------------------------------------------

input_layer = Input(shape=(5,), name="input")

encoder_1 = Dense(
    4,
    activation="relu",
    name="encoder_1"
)(input_layer)

latent = Dense(
    2,
    activation="relu",
    name="latent"
)(encoder_1)

decoder_1 = Dense(
    4,
    activation="relu",
    name="decoder_1"
)(latent)

output_layer = Dense(
    5,
    activation="linear",
    name="output"
)(decoder_1)

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)


# ------------------------------------------------------------
# 3. Compile model
# ------------------------------------------------------------

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("\nModel architecture:")
autoencoder.summary()


# ------------------------------------------------------------
# 4. Train model
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)


# ------------------------------------------------------------
# 5. Save trained model
# ------------------------------------------------------------

model_path = OUTPUT_DIR / "autoencoder.keras"

autoencoder.save(model_path)


# ------------------------------------------------------------
# 6. Plot training history
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training History")
plt.legend()

plt.tight_layout()

history_path = OUTPUT_DIR / "training_history.png"

plt.savefig(
    history_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 7. Final training information
# ------------------------------------------------------------

epochs_completed = len(history.history["loss"])

print("\nEpochs completed:")
print(epochs_completed)

print("\nFinal training loss:")
print(history.history["loss"][-1])

print("\nFinal validation loss:")
print(history.history["val_loss"][-1])

print("\nSaved model:")
print(model_path)

print("\nSaved training plot:")
print(history_path)

print("\n" + "=" * 60)
print("AUTOENCODER TRAINING COMPLETE")
print("=" * 60)