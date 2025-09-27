"""
prepare_audio.py
Handles audio preprocessing for music auto-tagging.
"""

import librosa
import numpy as np

SR = 12000       # Sample rate
DURATION = 29.12 # Seconds
N_MELS = 96      # Mel bands


def prepare_audio(filename):
    """Load and preprocess audio file into mel spectrogram input."""
    y, sr = librosa.load(filename, sr=SR, duration=DURATION, mono=True, res_type="kaiser_fast")

    # Pad/trim to fixed length
    expected_len = int(SR * DURATION)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]

    # Convert to shape (1, n_mels, time)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel[np.newaxis, ...]  # add channel dim
