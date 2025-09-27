import numpy as np
import librosa

SR = 12000
DURATION = 29
SAMPLES_PER_TRACK = SR * DURATION

def prepare_audio(filename):
    # Load mono audio at 12kHz, fixed duration
    src, sr = librosa.load(filename, sr=SR, duration=DURATION, mono=True)

    # Pad or trim
    if src.shape[0] < SAMPLES_PER_TRACK:
        pad_width = SAMPLES_PER_TRACK - src.shape[0]
        src = np.pad(src, (0, pad_width), mode='constant')
    elif src.shape[0] > SAMPLES_PER_TRACK:
        src = src[:SAMPLES_PER_TRACK]

    # Add channel dimension
    src = src[np.newaxis, :]  # (1, 348000)
    return src
