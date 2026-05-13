import os
import sys
import serial
import numpy as np
from collections import deque

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

from utils import parse_csi_line

PORT = "/dev/ttyUSB0"
BAUD = 115200
WINDOW = 50

MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "cnn_lstm_tvat_binary.h5")

buffer = deque(maxlen=WINDOW)

model = load_model(MODEL_PATH)

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Real-time intrusion detection started...")

while True:
    line = ser.readline().decode(errors="ignore")
    feat = parse_csi_line(line)

    if feat is None:
        continue

    buffer.append(feat)

    if len(buffer) == WINDOW:
        X = np.array(buffer, dtype=np.float32)
        X = np.expand_dims(X, axis=0)

        pred = float(model.predict(X, verbose=0)[0][0])

        if pred > 0.5:
            print(f"INTRUSION DETECTED | probability={pred:.3f}")
        else:
            print(f"EMPTY ROOM | probability={pred:.3f}")