"""
train.py
Train the CNN model for music auto-tagging.
Compatible with TF 2.20 + Keras 3.x + Kapre 0.3.7
"""

import os
import numpy as np
from types import SimpleNamespace
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from compact_cnn import models
from compact_cnn.prepare_audio import prepare_audio

# --------------------
# CONFIG
# --------------------
DATA_DIR = "dataset"       # Folder containing audio files (e.g. MP3 or WAV)
LABELS_FILE = "labels.npy" # Numpy file with shape (num_samples, num_tags)
MODEL_OUT = "music_tagging_model.h5"

# Audio / spectrogram parameters
args = SimpleNamespace(
    n_mels=96,
    fmin=0,
    fmax=6000,
    decibel=True,
    trainable_fb=False,
    trainable_kernel=False,
)

# --------------------
# LOAD DATA
# --------------------
print("Loading data...")

# Load labels
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError("Missing labels.npy (create this with your ground-truth tags).")

Y = np.load(LABELS_FILE)  # shape (N, 50)

# Load audio features
X = []
files = sorted(os.listdir(DATA_DIR))
for f in files:
    if f.endswith((".mp3", ".wav")):
        path = os.path.join(DATA_DIR, f)
        mel = prepare_audio(path)  # shape (n_mels, time)
        mel = np.expand_dims(mel, axis=0)  # add channel dim → (1, n_mels, time)
        X.append(mel)

X = np.array(X, dtype=np.float32)
print("X shape:", X.shape, "Y shape:", Y.shape)

# --------------------
# TRAIN / VAL SPLIT
# --------------------
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# --------------------
# BUILD MODEL
# --------------------
print("Building model...")
model = models.build_convnet_model(args)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# --------------------
# TRAIN
# --------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT, save_best_only=True)
]

print("Training...")
model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print(f"✅ Training complete. Best model saved as {MODEL_OUT}")
