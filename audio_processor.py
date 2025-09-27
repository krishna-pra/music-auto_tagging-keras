import numpy as np
import librosa

SR = 12000
DURATION = 29
SAMPLES_PER_TRACK = SR * DURATION  # 348000


def prepare_audio(filename):
    """
    Load and preprocess audio for the model.
    Returns shape: (1, 1, 348000) → (batch_size, channels, length)
    """

    # Load mono audio at 12kHz
    src, sr = librosa.load(filename, sr=SR, duration=DURATION, mono=True)

    # Fix length (pad or trim to exactly 348000 samples)
    if src.shape[0] < SAMPLES_PER_TRACK:
        pad_width = SAMPLES_PER_TRACK - src.shape[0]
        src = np.pad(src, (0, pad_width), mode="constant")
    elif src.shape[0] > SAMPLES_PER_TRACK:
        src = src[:SAMPLES_PER_TRACK]

    # Add channel dimension → (1, 348000)
    src = src[np.newaxis, :]

    # Add batch dimension → (1, 1, 348000)
    src = np.expand_dims(src, axis=0)

    return src
