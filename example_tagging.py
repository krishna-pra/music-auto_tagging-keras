"""
example_tagging.py
Run inference on a single audio file.
"""

import sys
import numpy as np
from tensorflow.keras.models import load_model
from compact_cnn.prepare_audio import prepare_audio

if len(sys.argv) < 2:
    print("Usage: python example_tagging.py <audiofile>")
    sys.exit(1)

filename = sys.argv[1]
print(f"Loading model...")
model = load_model("music_tagging_model.h5", compile=False)

print(f"Processing {filename}...")
x = prepare_audio(filename)

# Add batch dimension: (1, 1, n_mels, time)
x = np.expand_dims(x, axis=0)

print("Running inference...")
pred = model.predict(x)
print("Predictions:", pred[0])
