# CSI Sentinel: Real-Time WiFi CSI Intrusion Detection using ESP32 and Temporal Convolutional Networks

## Overview

CSI Sentinel is a real-time, device-free intrusion detection system that leverages WiFi Channel State Information (CSI) captured from an ESP32-E operating in passive sensing mode. The system detects human presence and motion by analyzing disturbances in wireless signal propagation without requiring cameras, wearable devices, PIR sensors, or dedicated tracking hardware.

The ESP32-E continuously captures CSI packets from ambient WiFi traffic and streams them to a GPU-enabled edge machine for real-time parsing, feature extraction, visualization, and deep learning inference.

The final detection engine is based on a **Temporal Convolutional Network (TCN)**, which efficiently learns temporal patterns in CSI streams and provides high intrusion detection accuracy with low latency, making it suitable for real-time deployment.

---

## System Architecture

```text
WiFi Environment
        │
        ▼
ESP32-E Passive CSI Sniffer
        │
        ▼
Real-Time CSI Serial Stream
        │
        ▼
CSI Packet Parsing
        │
        ▼
I/Q Extraction
        │
        ▼
Amplitude + Phase Feature Extraction
        │
        ▼
Feature Normalization
        │
        ▼
Sliding Window Generation
        │
        ▼
Temporal Convolutional Network (TCN)
        │
        ▼
Intrusion Probability Estimation
        │
        ▼
Real-Time Dashboard Visualization
```

---

## Deep Learning Pipeline

The system uses a Temporal Convolutional Network (TCN) to classify CSI activity into:

- Normal / Empty Room
- Human Movement / Intrusion

### Model Flow

```text
CSI Window Input
        │
        ▼
Amplitude + Phase Features
        │
        ▼
Conv1D Feature Extraction
        │
        ▼
Batch Normalization
        │
        ▼
TCN Residual Block 1
        │
        ▼
TCN Residual Block 2
        │
        ▼
TCN Residual Block 3
        │
        ▼
TCN Residual Block 4
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer
        │
        ▼
Dropout
        │
        ▼
Sigmoid Output
        │
        ▼
Intrusion Probability
```

### Why TCN?

Human movement causes temporal variations in CSI amplitude and phase across WiFi subcarriers. TCNs capture these variations using dilated convolutions:

```text
Dilation = 1  → Local packet variations
Dilation = 2  → Short-term movement patterns
Dilation = 4  → Medium-range temporal dependencies
Dilation = 8  → Long-range CSI disturbances
```

Compared to recurrent architectures such as LSTM and GRU, TCNs provide:

- Faster inference
- Better parallelization
- Lower latency
- Improved real-time deployment capability
- Reduced false alarms

---

## CSI Feature Representation

Each CSI packet contains complex I/Q values.

### Feature Extraction

```text
Amplitude = √(I² + Q²)

Phase = atan2(Q, I)
```

For each packet:

```text
64 Amplitude Features
64 Phase Features
--------------------
128 Total Features
```

### Sliding Window Input

```text
Window Size : 50 CSI Packets

Feature Size : 128

Input Shape  : (50 × 128)
```

Each sample therefore represents:

- 50 consecutive CSI packets
- 64 WiFi subcarriers
- Amplitude and phase information
- Temporal environmental dynamics

---

## Real-Time Inference Pipeline

```text
ESP32-E CSI Stream
        │
        ▼
Serial Reader
        │
        ▼
CSI Line Parser
        │
        ▼
Amplitude + Phase Extraction
        │
        ▼
Feature Buffer
        │
        ▼
50-Packet Sliding Window
        │
        ▼
TCN Inference
        │
        ▼
Probability Smoothing
        │
        ▼
Threshold Decision
        │
        ▼
Dashboard Update
```

Decision Logic:

```text
Probability < Threshold
    → Empty Room

Probability ≥ Threshold
    → Intrusion Detected
```

---

## Features

- Passive WiFi CSI sensing using ESP32-E
- Device-free intrusion detection
- Real-time CSI packet streaming
- CSI amplitude extraction
- CSI phase extraction
- Sliding window temporal processing
- Temporal Convolutional Network (TCN) inference
- GPU-accelerated TensorFlow execution
- Real-time intrusion probability estimation
- CSI heatmap visualization
- Live subcarrier monitoring
- Modular preprocessing and training pipeline
- Jupyter Notebook experimentation workflow
- Real-time dashboard visualization

---

## Model Comparison

| Model | Accuracy | Remarks |
|---------|---------|---------|
| CNN + LSTM | ~97% | Strong baseline |
| CNN + GRU | ~98% | Faster than LSTM |
| CNN + BiLSTM | ~98% | Higher complexity |
| CNN + Attention + LSTM | ~96% | Overfitting observed |
| TCN | ~99% | Best balance of speed and accuracy |

The TCN model was selected as the final deployment architecture due to its superior real-time performance and reduced false-positive rate.

---

## Real-Time Dashboard

The dashboard provides:

- Intrusion probability timeline
- Real-time detection status
- CSI amplitude heatmaps
- Live subcarrier visualization
- Continuous streaming inference
- Packet statistics and monitoring

---

## Hardware Requirements

### Sensing Node

- ESP32-E
- WiFi Router / Access Point

### Processing Node

- Ubuntu Linux Machine
- NVIDIA GPU (recommended)
- CUDA-enabled TensorFlow environment

---

## Software Stack

### Core Frameworks

- Python 3.10
- TensorFlow GPU
- ESP-IDF v4.4.4

### Libraries

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PySerial
- Jupyter Notebook

### Development Tools

- VS Code
- Git
- GitHub

---

## Environment

| Component | Version |
|------------|------------|
| OS | Ubuntu 26.04 LTS |
| Python | 3.10 |
| TensorFlow | GPU Enabled |
| CUDA | Enabled |
| ESP-IDF | v4.4.4 |

---

## Repository Structure

```text
csi_intrusion_pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── realtime/
│   ├── live_dashboard.py
│   ├── live_dashboard_fast.py
│   ├── live_dashboard_improved.py
│   └── live_predict.py
│
├── scripts/
│   ├── preprocess.py
│   ├── train_model.py
│   └── utils.py
│
├── Notebook/
│   └── Train and test.ipynb
│
├── Latex/
│
├── run_pipeline.sh
│
└── README.md
```


- Edge AI
- Deep Learning
- Temporal Signal Processing
- Semantic Edge Intelligence
