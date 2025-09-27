"""
main.py
Training script for music auto-tagging CNN.
"""

import argparse
import numpy as np
from tensorflow.keras.optimizers import Adam
from compact_cnn import models


def train(args):
    model = models.build_convnet_model(args)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Dummy training loop for now (replace with dataset loader)
    X = np.random.rand(10, 1, args.n_mels, 646)  # fake input
    y = np.random.randint(0, 2, (10, 50))

    model.fit(X, y, epochs=2, batch_size=2)
    model.save("music_tagging_model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mels", type=int, default=96)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=6000)
    parser.add_argument("--decibel", type=bool, default=True)
    parser.add_argument("--trainable_fb", type=bool, default=False)
    parser.add_argument("--trainable_kernel", type=bool, default=False)
    args = parser.parse_args()

    train(args)
