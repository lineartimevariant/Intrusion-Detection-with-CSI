#!/bin/bash
set -e

echo "========== CSI Intrusion Pipeline =========="

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Step 1: Preprocessing CSI..."
cd "$BASE_DIR/scripts"
python preprocess.py

echo "Step 2: Training CNN+LSTM..."
python train_model.py

echo "Training complete."

echo "To start real-time inference, run:"
echo "cd $BASE_DIR/realtime"
echo "python live_predict.py"