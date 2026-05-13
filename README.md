Real-Time WiFi CSI Intrusion Detection using ESP32 and Deep Learning

CSI Sentinel is a real-time device-free intrusion detection system that uses WiFi Channel State Information (CSI) captured from an ESP32-E in passive sensing mode. The project combines wireless sensing, signal processing, and deep learning to detect human presence and motion without requiring cameras, wearable devices, or dedicated sensors.

The ESP32 continuously captures CSI packets from ambient WiFi traffic and streams them to a GPU-enabled edge machine for real-time parsing, feature extraction, visualization, and inference. A CNN + LSTM deep learning pipeline processes CSI amplitude and phase information to classify environmental activity as either normal or intrusion.

System Architecture

WiFi Environment
↓
ESP32-E Passive CSI Sniffer
↓
Real-Time CSI Serial Stream
↓
CSI Parsing & Feature Extraction
↓
Amplitude + Phase Processing
↓
Sliding Window Generation
↓
CNN + LSTM Deep Learning Model
↓
Real-Time Intrusion Prediction
↓
Live Dashboard Visualization

Key Features
Passive WiFi CSI sensing using ESP32-E
Device-free intrusion detection
Real-time CSI packet streaming
CSI amplitude and phase feature extraction
Sliding window temporal processing
CNN + LSTM deep learning inference
GPU-accelerated TensorFlow processing
Live intrusion probability visualization
Real-time CSI heatmaps and subcarrier monitoring
Modular preprocessing and training pipeline
Jupyter Notebook based experimentation workflow
Deep Learning Pipeline

The project uses a hybrid CNN + LSTM architecture for spatial-temporal CSI analysis:

Conv1D layers extract spatial CSI subcarrier features
Pooling layers reduce dimensionality and noise
LSTM layers capture temporal motion dynamics
Dense layers perform intrusion classification
Sigmoid activation outputs intrusion probability

Input representation:

Window Size : 50 CSI packets
Feature Size: 128
(64 amplitude + 64 phase features)

Real-Time Dashboard

The live dashboard provides:

Intrusion probability timeline
CSI amplitude visualization
Live CSI heatmaps
Real-time subcarrier behavior analysis
Continuous streaming inference
Technologies Used

Hardware

ESP32-E
WiFi Router / Access Point
NVIDIA GPU-enabled Laptop

Software

Python 3.10
TensorFlow GPU
ESP-IDF v4.4.4
NumPy
Pandas
Matplotlib
PySerial
Scikit-learn
Jupyter Notebook
VS Code
Environment

OS: Ubuntu 26.04 LTS
Python: 3.10
TensorFlow: GPU Enabled
ESP-IDF: v4.4.4
CUDA: Enabled
GPU: NVIDIA RTX Series

Repository Structure

csi_intrusion_pipeline/

data/
├── raw/
├── processed/
└── models/

realtime/
├── live_dashboard.py
└── live_predict.py

scripts/
├── preprocess.py
├── train_model.py
└── utils.py

Notebook/
└── Train and test.ipynb

Latex/

run_pipeline.sh

README.md

Applications
Smart home intrusion detection
Device-free indoor monitoring
Human activity sensing
Edge AI sensing systems
WiFi-based occupancy detection
Low-cost wireless surveillance
Ambient intelligence systems
Future Improvements
Transformer-based CSI models
Multi-ESP32 sensing fusion
TinyML deployment
Federated CSI learning
CSI denoising and calibration
Cross-environment adaptation
Semantic wireless sensing
