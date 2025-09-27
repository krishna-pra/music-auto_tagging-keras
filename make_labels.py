"""
make_labels.py
Helper script to create labels.npy for training.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "dataset"        # folder with your audio files
LABELS_NPY = "labels.npy"   # output file
CSV_FILE = "labels.csv"     # optional CSV with real labels
NUM_TAGS = 50               # number of tags (columns)

def create_dummy_labels():
    """Generate random multi-hot labels for each audio file."""
    files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith((".mp3", ".wav"))]
    N = len(files)
    Y = np.random.randint(0, 2, size=(N, NUM_TAGS))
    np.save(LABELS_NPY, Y)
    print(f"✅ Dummy labels saved: {LABELS_NPY} (shape {Y.shape})")

def create_labels_from_csv():
    """Load labels from CSV with format: filename,tag1,tag2,..."""
    df = pd.read_csv(CSV_FILE)
    files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith((".mp3", ".wav"))]
    
    # Map filenames to row indices
    Y = np.zeros((len(files), NUM_TAGS), dtype=int)
    for i, f in enumerate(files):
        row = df[df["filename"] == f]
        if not row.empty:
            tags = row.values[0][1:]  # skip filename col
            Y[i, :len(tags)] = tags
    np.save(LABELS_NPY, Y)
    print(f"✅ Labels from CSV saved: {LABELS_NPY} (shape {Y.shape})")

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        print("CSV file found, creating labels from CSV...")
        create_labels_from_csv()
    else:
        print("No CSV found, creating dummy random labels...")
        create_dummy_labels()
