# compact_cnn/prepare_audio.py
"""
Load audio and convert to log-mel spectrogram (channels-last).
Returns: numpy array shape (n_mels, frames, 1)
"""

import numpy as np
import librosa

SR = 12000
DURATION = 29.12   # seconds (original dataset likely ~29s)
N_MELS = 96
N_FFT = 2048
HOP_LENGTH = 512


def prepare_audio(filename: str, sr: int = SR, duration: float = DURATION,
                  n_mels: int = N_MELS, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
    # load (mono), pad/trim to duration
    y, _ = librosa.load(filename, sr=sr, duration=duration, mono=True, res_type="kaiser_fast")
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize to roughly [-1,1] (optional but often helps)
    # avoid dividing by zero
    mean = log_mel.mean()
    std = log_mel.std() if log_mel.std() > 0 else 1.0
    log_mel = (log_mel - mean) / std

    # channel-last with single channel
    return log_mel[..., np.newaxis].astype("float32")
