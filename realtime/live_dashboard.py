import os
import re
import serial
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = "/home/rubix-irat/esp_projects/csi_intrusion_pipeline"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "cnn_lstm_tvat_binary.h5"
)

# ==========================================================
# SERIAL CONFIG
# ==========================================================

PORT = "/dev/ttyUSB1"
BAUD = 115200

# ==========================================================
# CSI SETTINGS
# ==========================================================

TARGET_SUBCARRIERS = 64
WINDOW_SIZE = 50
FEATURE_SIZE = 128

THRESHOLD = 0.5

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading CNN+LSTM model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

print("GPU Devices:")
print(tf.config.list_physical_devices("GPU"))

# ==========================================================
# SERIAL CONNECTION
# ==========================================================

print("Connecting to ESP32...")
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Connected.")

# ==========================================================
# BUFFERS
# ==========================================================

feature_buffer = deque(maxlen=WINDOW_SIZE)
probability_buffer = deque(maxlen=200)
heatmap_buffer = deque(maxlen=100)

motion_buffer = deque(maxlen=200)
energy_buffer = deque(maxlen=200)
variance_buffer = deque(maxlen=200)
prediction_buffer = deque(maxlen=200)

alert_count = 0
packet_count = 0
latest_prob = 0.0
latest_prediction = "WAITING"

# ==========================================================
# CSI PARSER
# ==========================================================

def parse_csi_line(line):

    if "CSI_DATA" not in line:
        return None

    try:
        matches = re.findall(r"\[(.*?)\]", line)

        if not matches:
            return None

        values = list(map(int, matches[-1].split()))

        if len(values) < 2:
            return None

        if len(values) % 2 != 0:
            values = values[:-1]

        iq = np.array(values, dtype=np.float32)

        real = iq[::2]
        imag = iq[1::2]

        amplitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)

        n = min(len(amplitude), TARGET_SUBCARRIERS)

        amp_fixed = np.zeros(TARGET_SUBCARRIERS)
        phase_fixed = np.zeros(TARGET_SUBCARRIERS)

        amp_fixed[:n] = amplitude[:n]
        phase_fixed[:n] = phase[:n]

        amp_fixed = (amp_fixed - np.mean(amp_fixed)) / (np.std(amp_fixed) + 1e-6)
        phase_fixed = (phase_fixed - np.mean(phase_fixed)) / (np.std(phase_fixed) + 1e-6)

        features = np.concatenate([amp_fixed, phase_fixed])

        return features, amp_fixed, phase_fixed

    except Exception:
        return None


# ==========================================================
# PLOT SETUP
# ==========================================================

plt.style.use("dark_background")

fig = plt.figure(figsize=(18, 10))
fig.suptitle("Real-Time ESP32 CSI Intrusion Detection Dashboard", fontsize=18)

ax_prob = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax_state = plt.subplot2grid((3, 3), (0, 2))

ax_amp = plt.subplot2grid((3, 3), (1, 0))
ax_heatmap = plt.subplot2grid((3, 3), (1, 1))
ax_var = plt.subplot2grid((3, 3), (1, 2))

ax_motion = plt.subplot2grid((3, 3), (2, 0))
ax_energy = plt.subplot2grid((3, 3), (2, 1))
ax_alert = plt.subplot2grid((3, 3), (2, 2))


# ==========================================================
# LIVE UPDATE
# ==========================================================

def update(frame):

    global alert_count, packet_count, latest_prob, latest_prediction

    try:
        line = ser.readline().decode(errors="ignore")
        parsed = parse_csi_line(line)

        if parsed is None:
            return

        features, amplitude, phase = parsed

        packet_count += 1

        feature_buffer.append(features)
        heatmap_buffer.append(amplitude)

        # ==================================================
        # FEATURE VISUAL METRICS
        # ==================================================

        csi_energy = np.mean(amplitude ** 2)
        csi_variance = np.var(amplitude)

        if len(heatmap_buffer) > 1:
            previous_amp = np.array(heatmap_buffer)[-2]
            motion_score = np.mean(np.abs(amplitude - previous_amp))
        else:
            motion_score = 0.0

        energy_buffer.append(csi_energy)
        variance_buffer.append(csi_variance)
        motion_buffer.append(motion_score)

        # ==================================================
        # MODEL PREDICTION
        # ==================================================

        if len(feature_buffer) == WINDOW_SIZE:

            X = np.array(feature_buffer)
            X = np.expand_dims(X, axis=0)

            prob = float(model.predict(X, verbose=0)[0][0])
            probability_buffer.append(prob)

            latest_prob = prob

            if prob > THRESHOLD:
                latest_prediction = "INTRUSION"
                prediction_buffer.append(1)
                alert_count += 1
            else:
                latest_prediction = "EMPTY ROOM"
                prediction_buffer.append(0)

            print(f"{latest_prediction} | probability={prob:.3f}")

        # ==================================================
        # PLOT 1: LIVE INTRUSION PROBABILITY
        # ==================================================

        ax_prob.clear()
        ax_prob.plot(probability_buffer, linewidth=2)
        ax_prob.axhline(THRESHOLD, linestyle="--", linewidth=1.5)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_title("Live Intrusion Probability")
        ax_prob.set_ylabel("Probability")
        ax_prob.set_xlabel("Time Window")
        ax_prob.grid(True, alpha=0.3)

        # ==================================================
        # PLOT 2: CURRENT PREDICTION STATE
        # ==================================================

        ax_state.clear()
        ax_state.axis("off")

        if latest_prediction == "INTRUSION":
            color = "red"
        elif latest_prediction == "EMPTY ROOM":
            color = "lime"
        else:
            color = "yellow"

        ax_state.text(
            0.5,
            0.65,
            latest_prediction,
            fontsize=28,
            ha="center",
            va="center",
            color=color,
            fontweight="bold"
        )

        ax_state.text(
            0.5,
            0.35,
            f"Probability: {latest_prob:.3f}",
            fontsize=18,
            ha="center",
            va="center",
            color="white"
        )

        ax_state.set_title("Current State")

        # ==================================================
        # PLOT 3: CURRENT CSI AMPLITUDE
        # ==================================================

        ax_amp.clear()
        ax_amp.plot(amplitude, linewidth=2)
        ax_amp.set_title("Current CSI Amplitude")
        ax_amp.set_xlabel("Subcarrier Index")
        ax_amp.set_ylabel("Normalized Amplitude")
        ax_amp.grid(True, alpha=0.3)

        # ==================================================
        # PLOT 4: LIVE CSI HEATMAP
        # ==================================================

        ax_heatmap.clear()

        if len(heatmap_buffer) > 5:
            heatmap = np.array(heatmap_buffer)

            ax_heatmap.imshow(
                heatmap.T,
                aspect="auto",
                origin="lower",
                cmap="turbo"
            )

        ax_heatmap.set_title("Live CSI Heatmap")
        ax_heatmap.set_xlabel("Recent Packets")
        ax_heatmap.set_ylabel("Subcarrier Index")

        # ==================================================
        # PLOT 5: SUBCARRIER VARIANCE BAR
        # ==================================================

        ax_var.clear()

        if len(heatmap_buffer) > 5:
            heatmap = np.array(heatmap_buffer)
            subcarrier_variance = np.var(heatmap, axis=0)

            ax_var.bar(
                np.arange(TARGET_SUBCARRIERS),
                subcarrier_variance
            )

        ax_var.set_title("Subcarrier Activity / Variance")
        ax_var.set_xlabel("Subcarrier Index")
        ax_var.set_ylabel("Variance")
        ax_var.grid(True, alpha=0.3)

        # ==================================================
        # PLOT 6: MOTION SCORE
        # ==================================================

        ax_motion.clear()
        ax_motion.plot(motion_buffer, linewidth=2)
        ax_motion.set_title("CSI Motion Score")
        ax_motion.set_xlabel("Time")
        ax_motion.set_ylabel("Mean Amplitude Change")
        ax_motion.grid(True, alpha=0.3)

        # ==================================================
        # PLOT 7: CSI ENERGY
        # ==================================================

        ax_energy.clear()
        ax_energy.plot(energy_buffer, linewidth=2)
        ax_energy.set_title("CSI Energy")
        ax_energy.set_xlabel("Time")
        ax_energy.set_ylabel("Mean Squared Amplitude")
        ax_energy.grid(True, alpha=0.3)

        # ==================================================
        # PLOT 8: ALERT AND PACKET MONITOR
        # ==================================================

        ax_alert.clear()
        ax_alert.axis("off")

        ax_alert.text(
            0.5,
            0.70,
            f"Alerts: {alert_count}",
            fontsize=26,
            ha="center",
            va="center",
            color="red",
            fontweight="bold"
        )

        ax_alert.text(
            0.5,
            0.45,
            f"Packets: {packet_count}",
            fontsize=20,
            ha="center",
            va="center",
            color="white"
        )

        ax_alert.text(
            0.5,
            0.22,
            f"Window: {len(feature_buffer)}/{WINDOW_SIZE}",
            fontsize=18,
            ha="center",
            va="center",
            color="cyan"
        )

        ax_alert.set_title("System Monitor")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    except Exception as e:
        print("Error:", e)


# ==========================================================
# ANIMATION
# ==========================================================

ani = FuncAnimation(
    fig,
    update,
    interval=100,
    cache_frame_data=False
)

plt.show()

# ==========================================================
# CLEANUP
# ==========================================================

ser.close()