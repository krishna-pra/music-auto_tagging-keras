"""
train.py
Trains the CNN model and saves it.
"""

import numpy as np
import argparse
from keras.optimizers import Adam
from compact_cnn.models import build_convnet_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mels", type=int, default=96)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=6000)
    parser.add_argument("--decibel", action="store_true")
    parser.add_argument("--trainable_fb", action="store_true")
    parser.add_argument("--trainable_kernel", action="store_true")
    args = parser.parse_args()

    X = np.load("X.npy")  # features (precomputed)
    y = np.load("labels.npy")  # labels

    model = build_convnet_model(args)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(1e-4),
        metrics=["accuracy"]
    )

    model.fit(
        X, y,
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    model.save("music_tagging_model.h5")
    print("Saved model to music_tagging_model.h5")

if __name__ == "__main__":
    main()
