"""
make_labels.py
Creates labels.npy from labels.csv (or generates dummy labels).
"""

import numpy as np
import pandas as pd
import os

def main():
    if os.path.exists("labels.csv"):
        df = pd.read_csv("labels.csv")
        y = df.iloc[:, 1:].values  # first col filename, rest are labels
        np.save("labels.npy", y)
        print("Saved labels.npy from labels.csv")
    else:
        print("No labels.csv found, creating dummy labels...")
        # Dummy labels: 100 samples × 50 tags
        y = np.random.randint(0, 2, size=(100, 50))
        np.save("labels.npy", y)

if __name__ == "__main__":
    main()
