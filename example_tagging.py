"""
example_tagging.py
Loads a trained model and predicts tags for an audio file.
"""

import sys
import numpy as np
from keras.models import load_model
from compact_cnn.prepare_audio import load_audio_file
from compact_cnn.models import build_convnet_model

if len(sys.argv) < 2:
    print("Usage: python example_tagging.py <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

try:
    model = load_model("music_tagging_model.h5", compile=False)
    print("Loaded full model (music_tagging_model.h5)")
except Exception:
    print("Full model not found, building model + loading weights...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mels", type=int, default=96)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=6000)
    parser.add_argument("--decibel", action="store_true")
    parser.add_argument("--trainable_fb", action="store_true")
    parser.add_argument("--trainable_kernel", action="store_true")
    args = parser.parse_args([])

    model = build_convnet_model(args)
    model.load_weights("music_tagging_model_weights.h5")

X = load_audio_file(audio_file)
X = np.expand_dims(X, 0)  # (1, 1, n_mels, time)

pred = model.predict(X)
print("Predictions:", pred[0][:10], "...")
