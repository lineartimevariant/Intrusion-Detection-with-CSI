import os, re, serial, threading, queue
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model

BASE_DIR = "/home/rubix-irat/esp_projects/csi_intrusion_pipeline"
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "tcn_tvat_binary.h5")

PORT = "/dev/ttyUSB0"
BAUD = 115200

TARGET_SUBCARRIERS = 64
WINDOW_SIZE = 50
FEATURE_SIZE = 128
THRESHOLD = 0.65

PLOT_INTERVAL_MS = 300
INFER_EVERY_N_PACKETS = 10

model = load_model(MODEL_PATH)
model(np.zeros((1, WINDOW_SIZE, FEATURE_SIZE), dtype=np.float32), training=False)

ser = serial.Serial(PORT, BAUD, timeout=0.001)

feature_buffer = deque(maxlen=WINDOW_SIZE)
prob_buffer = deque(maxlen=100)
smooth_buffer = deque(maxlen=5)
heatmap_buffer = deque(maxlen=80)

line_queue = queue.Queue(maxsize=2000)

latest_prob = 0.0
latest_state = "WAITING"
packet_count = 0

def parse_csi_line(line):
    if "CSI_DATA" not in line:
        return None

    try:
        m = re.findall(r"\[(.*?)\]", line)
        if not m:
            return None

        values = list(map(int, m[-1].split()))
        if len(values) % 2 != 0:
            values = values[:-1]

        iq = np.array(values, dtype=np.float32)
        real = iq[::2]
        imag = iq[1::2]

        amp = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)

        n = min(len(amp), TARGET_SUBCARRIERS)

        amp_fixed = np.zeros(TARGET_SUBCARRIERS, dtype=np.float32)
        phase_fixed = np.zeros(TARGET_SUBCARRIERS, dtype=np.float32)

        amp_fixed[:n] = amp[:n]
        phase_fixed[:n] = phase[:n]

        amp_fixed = (amp_fixed - amp_fixed.mean()) / (amp_fixed.std() + 1e-6)
        phase_fixed = (phase_fixed - phase_fixed.mean()) / (phase_fixed.std() + 1e-6)

        features = np.concatenate([amp_fixed, phase_fixed]).astype(np.float32)

        return features, amp_fixed

    except Exception:
        return None

def serial_reader():
    while True:
        try:
            line = ser.readline().decode(errors="ignore")
            if line:
                try:
                    line_queue.put_nowait(line)
                except queue.Full:
                    pass
        except Exception:
            pass

def inference_worker():
    global latest_prob, latest_state, packet_count

    while True:
        line = line_queue.get()
        parsed = parse_csi_line(line)

        if parsed is None:
            continue

        features, amp = parsed

        packet_count += 1
        feature_buffer.append(features)
        heatmap_buffer.append(amp)

        if len(feature_buffer) == WINDOW_SIZE and packet_count % INFER_EVERY_N_PACKETS == 0:
            X_live = np.array(feature_buffer, dtype=np.float32)[None, :, :]
            prob = float(model(X_live, training=False).numpy()[0][0])

            smooth_buffer.append(prob)
            latest_prob = float(np.mean(smooth_buffer))
            prob_buffer.append(latest_prob)

            latest_state = "INTRUSION" if latest_prob > THRESHOLD else "EMPTY ROOM"

            print(f"{latest_state} | probability={latest_prob:.3f} | packets={packet_count}")

threading.Thread(target=serial_reader, daemon=True).start()
threading.Thread(target=inference_worker, daemon=True).start()

plt.style.use("dark_background")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle("Fast Real-Time CSI Intrusion Dashboard")

def update(frame):
    ax1.clear()
    ax1.plot(prob_buffer, linewidth=2)
    ax1.axhline(THRESHOLD, linestyle="--")
    ax1.set_ylim(0, 1)
    ax1.set_title("Intrusion Probability")
    ax1.set_ylabel("Probability")
    ax1.grid(True, alpha=0.3)

    ax2.clear()
    ax2.axis("off")

    color = "red" if latest_state == "INTRUSION" else "lime"
    if latest_state == "WAITING":
        color = "yellow"

    ax2.text(
        0.5,
        0.6,
        latest_state,
        ha="center",
        va="center",
        fontsize=34,
        color=color,
        fontweight="bold"
    )

    ax2.text(
        0.5,
        0.25,
        f"Probability: {latest_prob:.3f} | Packets: {packet_count}",
        ha="center",
        va="center",
        fontsize=16,
        color="white"
    )

    ax3.clear()
    if len(heatmap_buffer) > 5:
        heatmap = np.array(heatmap_buffer)
        ax3.imshow(heatmap.T, aspect="auto", origin="lower", cmap="turbo")

    ax3.set_title("CSI Amplitude Heatmap")
    ax3.set_xlabel("Recent Packets")
    ax3.set_ylabel("Subcarrier")

    plt.tight_layout()

ani = FuncAnimation(
    fig,
    update,
    interval=PLOT_INTERVAL_MS,
    cache_frame_data=False
)

plt.show()

ser.close()