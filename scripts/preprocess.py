import os
import numpy as np
from utils import parse_csi_line

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

WINDOW_SIZE = 50
STEP = 10

os.makedirs(OUT_DIR, exist_ok=True)

def load_features(file_path, label):
    features = []

    print(f"Reading: {file_path}")

    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            feat = parse_csi_line(line)
            if feat is not None:
                features.append(feat)

    if len(features) < WINDOW_SIZE:
        raise ValueError(f"Not enough valid CSI rows in {file_path}. Found {len(features)}")

    features = np.stack(features, axis=0)

    X = []
    y = []

    for i in range(0, len(features) - WINDOW_SIZE + 1, STEP):
        X.append(features[i:i + WINDOW_SIZE])
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

intrusion_path = os.path.join(RAW_DIR, "intrusion.csv")
empty_path = os.path.join(RAW_DIR, "empty_room.csv")

print("Processing intrusion...")
X1, y1 = load_features(intrusion_path, 1)

print("Processing empty room...")
X0, y0 = load_features(empty_path, 0)

X = np.concatenate([X1, X0], axis=0)
y = np.concatenate([y1, y0], axis=0)

idx = np.random.permutation(len(y))
X = X[idx]
y = y[idx]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

print("Saved processed dataset.")