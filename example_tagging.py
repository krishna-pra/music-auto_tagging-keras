# example_tagging.py
"""
Run inference on one audio file.
Usage: python example_tagging.py path/to/song.wav
"""

import sys
import os
import numpy as np
from types import SimpleNamespace

from tensorflow.keras.models import load_model
from compact_cnn.prepare_audio import prepare_audio
from compact_cnn import models as my_models

MODEL_FILE = "music_tagging_model.keras"
WEIGHTS_FILE = "music_tagging_model_weights.h5"

def load_or_build_model(n_mels=96):
    if os.path.exists(MODEL_FILE):
        print("Loading full model:", MODEL_FILE)
        m = load_model(MODEL_FILE, compile=False)
        return m
    else:
        print("Full model not found. Building model and trying to load weights...")
        args = SimpleNamespace(n_mels=n_mels)
        m = my_models.build_convnet_model(args)
        if os.path.exists(WEIGHTS_FILE):
            m.load_weights(WEIGHTS_FILE)
            print("Loaded weights from", WEIGHTS_FILE)
            return m
        else:
            raise FileNotFoundError("No model (.keras) or weights (.h5) found. Run training first.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_tagging.py <audio_file>")
        sys.exit(1)
    audio_path = sys.argv[1]
    x = prepare_audio(audio_path)           # (n_mels, frames, 1)
    x = np.expand_dims(x, axis=0)           # (1, n_mels, frames, 1)
    model = load_or_build_model(n_mels=x.shape[1])
    preds = model.predict(x)
    top_idx = np.argsort(preds[0])[-10:][::-1]
    print("Top predicted tag indices and scores:")
    for i in top_idx:
        print(f"{i}: {preds[0,i]:.4f}")
