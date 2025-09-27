"""
prepare_audio.py
Loads audio and converts to log-mel spectrograms for training/inference.
"""

import librosa
import numpy as np


def load_audio_file(path, sr=12000, duration=29.0, n_mels=96):
    """
    Load audio file and convert to mel-spectrogram.
    Returns (1, n_mels, time) for channels_first.
    """
    y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)

    # Pad/trim
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=512, n_fft=1024, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel[np.newaxis, ...]  # (1, n_mels, time)
