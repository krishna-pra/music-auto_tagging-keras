# train.py
"""
Train script (simple, loads all audio into memory — fine for small tests).
Saves: music_tagging_model.keras (Keras native format) and music_tagging_model_weights.h5
"""

import os
import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from compact_cnn import models as my_models
from compact_cnn.prepare_audio import prepare_audio

DATA_DIR = "dataset"
LABELS_FILE = "labels.npy"
OUT_MODEL = "music_tagging_model.keras"
OUT_WEIGHTS = "music_tagging_model_weights.h5"

def load_data(dataset_dir=DATA_DIR, labels_file=LABELS_FILE):
    files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith((".wav", ".mp3"))])
    if len(files) == 0:
        raise RuntimeError("No audio files found in dataset/.")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"{labels_file} not found. Run make_labels.py first.")
    Y = np.load(labels_file)
    if Y.shape[0] != len(files):
        raise RuntimeError(f"labels.npy has {Y.shape[0]} rows but dataset contains {len(files)} audio files. They must match.")

    X_list = []
    for f in files:
        path = os.path.join(dataset_dir, f)
        mel = prepare_audio(path)  # (n_mels, frames, 1)
        X_list.append(mel)
    X = np.stack(X_list, axis=0)  # (N, n_mels, frames, 1)
    return X, Y

def main():
    print("Loading data...")
    X, Y = load_data()
    print("X shape:", X.shape, "Y shape:", Y.shape)

    args = SimpleNamespace(n_mels=X.shape[1])
    model = my_models.build_convnet_model(args)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

    checkpoint = ModelCheckpoint(OUT_MODEL, save_best_only=True, monitor="val_loss")
    early = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=8, callbacks=[checkpoint, early])

    # Also save weights in legacy h5 if you want to load them in older code
    model.save_weights(OUT_WEIGHTS)
    print("Training finished. Saved model:", OUT_MODEL, "and weights:", OUT_WEIGHTS)

if __name__ == "__main__":
    main()
