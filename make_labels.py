# make_labels.py
"""
Create labels.npy for training.
If labels.csv exists, it should have columns: filename,tag0,tag1,...,tag49
Otherwise random dummy labels are created for the number of audio files in dataset/.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "dataset"
CSV_FILE = "labels.csv"
OUT_FILE = "labels.npy"
NUM_TAGS = 50

def create_dummy():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".wav", ".mp3"))])
    N = len(files)
    if N == 0:
        raise RuntimeError("No audio files found in dataset/. Put some .wav/.mp3 files there.")
    Y = np.random.randint(0, 2, size=(N, NUM_TAGS))
    np.save(OUT_FILE, Y)
    print(f"Saved dummy labels to {OUT_FILE} (shape {Y.shape})")


def create_from_csv():
    df = pd.read_csv(CSV_FILE)
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".wav", ".mp3"))])
    mapping = {row["filename"]: row.iloc[1:].values.astype(int) for _, row in df.iterrows()}
    Y = []
    for f in files:
        if f in mapping:
            Y.append(mapping[f])
        else:
            Y.append(np.zeros(NUM_TAGS, dtype=int))
    Y = np.vstack(Y)
    np.save(OUT_FILE, Y)
    print(f"Saved labels to {OUT_FILE} (shape {Y.shape})")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_FILE):
        create_from_csv()
    else:
        create_dummy()
