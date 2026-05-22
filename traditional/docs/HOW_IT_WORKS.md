# ESPMotion — How It Works

ESPMotion detects motion by watching how a moving body disturbs WiFi radio waves.
This document walks the full pipeline, from the radio physics down to the MQTT
message, in the order the data actually flows.

```
WiFi packet ─▶ ESP32 CSI capture ─▶ spatial turbulence ─▶ filters
            ─▶ moving variance ─▶ IDLE/MOTION state machine ─▶ MQTT
```

## 1. WiFi CSI — a primer

A WiFi channel (HT20) is split into **64 subcarriers**, each a narrow slice of
frequency. When the receiver decodes a packet it estimates, for every subcarrier,
how the channel altered that slice — an amplitude and a phase. That set of
estimates is the **Channel State Information (CSI)**.

CSI is sensitive to the physical environment: every reflecting surface between
transmitter and receiver contributes a path, and the paths add up differently as
things move. A person walking through the room changes the multipath pattern, so
the per-subcarrier amplitudes wobble. **That wobble is the motion signal.** No
camera, no PIR — just the WiFi already in the air.

ESPMotion uses only the **amplitude** of each subcarrier (computed as
`sqrt(real² + imag²)` from the raw I/Q pair), and only the valid data subcarriers
inside the HT20 guard bands (indices 11–52, DC subcarrier 32 excluded).

## 2. ESP32 CSI capture

The device firmware is a MicroPython build that exposes the ESP32's CSI API
([micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi)).
At boot `src/main.py`:

1. Connects to your 2.4 GHz WiFi and forces **802.11b/g/n, HT20** so every packet
   carries 64 subcarriers.
2. Calls `wlan.csi_enable()` — from then on each received packet drops a CSI frame
   into a small circular buffer.
3. The main loop polls `wlan.csi_read()`, normalizes the payload to a fixed
   128-byte (64 × I/Q) layout, and hands it to the detector.

CSI is only produced when packets *arrive*. An idle ESP32 sitting on a quiet
network receives almost nothing — which is why ESPMotion generates its own
traffic.

## 3. Traffic generator — why it's needed

Motion detection needs a **steady, fast** stream of CSI frames (the default is 100
per second). Real networks don't deliver that on demand, so `traffic_generator.py`
runs a background thread that sends small UDP packets to the gateway at a fixed
rate. Each packet the AP routes back keeps CSI flowing at a predictable cadence.

This also makes the sample rate deterministic: the moving-variance window and the
filters are all tuned assuming ~100 packets/second.

The firmware watches for CSI **starvation** — if no frame arrives for 6 s it drops
MQTT and re-associates WiFi; after 20 s it reboots. A silently dropped WiFi link is
the usual culprit (`sendto()` keeps succeeding while nothing actually transmits).

## 4. Gain lock and the CV-normalization fallback

The ESP32's receiver runs an **Automatic Gain Control (AGC)** loop. AGC is great
for decoding packets but bad for sensing: it rescales amplitudes on its own, and
that rescaling looks just like motion.

The fix on capable chips (C3/S3/C5/C6) is **gain lock** — sample the AGC/FFT gain
over ~3 s while the room is still, then pin it with `csi_force_gain()` so
amplitudes stay comparable over time.

**The original ESP32 has no gain-lock support.** `run_gain_lock()` detects this
(`csi_gain_lock_supported()` returns false) and falls back to **CV normalization**:
instead of using raw amplitude spread, turbulence is computed as the *coefficient
of variation* — standard deviation **divided by the mean** (see §6). Dividing by
the mean cancels any uniform gain change, so detection stays stable even though the
hardware keeps rescaling. On the original ESP32 you will see
`Gain lock: Not supported … CV normalization enabled` at boot — that is expected.

(In `"auto"` mode even a gain-lock-capable chip skips the lock if the signal is too
strong, AGC < 30, and uses the same CV path.)

## 5. NBVI band calibration

Of the 64 subcarriers only some are useful — many are inside guard bands, sit on
the DC null, or are simply noisy. ESPMotion picks **12 subcarriers** automatically
at boot using **NBVI (Normalized Band Variance Index)**.

During the ~7 s calibration window (room kept still) the device collects baseline
packets and, for each candidate subcarrier, measures how variable it is *at rest*.
NBVI then selects 12 **non-consecutive** subcarriers that balance low baseline
noise with spectral spread — non-consecutive picks resist narrowband interference,
since a single interferer can't knock out the whole band.

For the MVS algorithm the same baseline data sets the detection **threshold**
(see §7). If calibration times out, a fixed default band is used instead.

## 6. Spatial turbulence

This is the core per-packet metric. For one CSI frame, take the amplitudes of the
12 selected subcarriers and measure how *uneven* they are:

- **Gain-locked chips:** `turbulence = std(amplitudes)` — raw standard deviation.
- **Original ESP32 (CV path):** `turbulence = std(amplitudes) / mean(amplitudes)`
  — the coefficient of variation, gain-invariant (see §4).

A still room gives a smooth, low-turbulence spectrum. A moving body distorts the
subcarriers unevenly, so turbulence rises. Each packet yields one turbulence value;
the stream of those values is the signal everything downstream works on.

## 7. Filters and moving-variance segmentation (MVS)

MVS is the default algorithm (`DETECTION_ALGORITHM = "mvs"`).

### Filters

Each turbulence value passes through an optional filter chain
(`raw → Hampel → low-pass → buffer`):

- **Hampel filter** (on by default) — a sliding-window median/MAD outlier
  rejecter. It removes isolated spikes (a dropped packet, an RF glitch) that would
  otherwise look like a burst of motion.
- **Low-pass filter** (off by default) — a 1st-order Butterworth, 11 Hz cutoff.
  Human movement is roughly 0.5–10 Hz; RF noise is higher. Enable it
  (`ENABLE_LOWPASS_FILTER`) in electrically noisy rooms.

### Moving variance

Filtered turbulence values go into a circular buffer of `SEG_WINDOW_SIZE` packets
(default 75, ~0.75 s). Each publish interval the **variance** of that window is
computed with a numerically stable two-pass formula. A still room → near-constant
turbulence → low variance. Motion → fluctuating turbulence → high variance. This
windowed variance is the **movement metric** (`mvmt` in the console, `movement` in
MQTT).

### Threshold

The threshold separating idle from motion is set at calibration from the baseline
variance (`src/threshold.py`):

- `SEG_THRESHOLD = "auto"` (default) — `P95(baseline) × 1.1`: the 95th percentile
  of resting variance, plus 10 % headroom to suppress false positives.
- `SEG_THRESHOLD = "min"` — `P100(baseline) × 1.0`: maximum sensitivity.
- A number — a fixed manual threshold (0.0–10.0).

### IDLE / MOTION state machine

A two-state machine compares the movement metric to the threshold:

- **IDLE → MOTION** when `moving_variance > threshold`.
- **MOTION → IDLE** when `moving_variance < threshold`.

If the AP hops WiFi channels mid-run, the resulting CSI spike would fake motion, so
the detection buffer is reset on a detected channel change.

## 8. MQTT publishing

ESPMotion is **publish-only** — it connects to the broker, never subscribes, and
never accepts commands.

- At startup it publishes one **info** message to `<MQTT_TOPIC>/info`
  (device id, algorithm, threshold, window size).
- Every publish interval it publishes a **state** message to `MQTT_TOPIC`
  (movement, threshold, state, packet counters, pps, timestamp).

Publishing is best-effort: transient `ENOMEM`/`EAGAIN` errors are skipped, and a
genuinely broken connection is dropped so detection keeps running. The exact JSON
schema is in [USAGE.md](USAGE.md#7-mqtt-payloads).

## ML detector (foundation)

ESPMotion ships a second, optional detector — a small neural network — enabled with
`DETECTION_ALGORITHM = "ml"` in `config.py`. It is included as a **working
foundation to rebuild from**, not a finished feature.

### What is present

The complete **on-device inference path** (`src/ml_detector.py`,
`src/ml_weights.py`):

- **Architecture:** a multilayer perceptron, `12 → 16 (ReLU) → 8 (ReLU) → 1
  (Sigmoid)`. The sigmoid output is scaled to 0–10 and compared to a threshold.
- **Input:** the same turbulence signal as MVS — but instead of a single variance,
  **12 statistical features** are extracted from the turbulence window
  (`src/features.py`):

  | # | Feature | Meaning |
  |---|---------|---------|
  | 1 | `turb_mean` | mean turbulence |
  | 2 | `turb_std` | standard deviation |
  | 3 | `turb_max` | window maximum |
  | 4 | `turb_min` | window minimum |
  | 5 | `turb_zcr` | zero-crossing rate (around the mean) |
  | 6 | `turb_skewness` | distribution asymmetry |
  | 7 | `turb_kurtosis` | distribution peakedness |
  | 8 | `turb_entropy` | signal entropy |
  | 9 | `turb_autocorr` | lag-1 autocorrelation |
  | 10 | `turb_mad` | median absolute deviation |
  | 11 | `turb_slope` | linear trend over the window |
  | 12 | `waveform_length` | cumulative sample-to-sample change |

- Features are standardized with the `FEATURE_MEAN`/`FEATURE_SCALE` vectors baked
  into `ml_weights.py`, then run through the MLP.
- The ML path needs only gain lock (no NBVI band calibration), so it boots in ~3 s
  with a fixed subcarrier set.

### What was removed

The **host-side training pipeline** that produced `ml_weights.py` — labeled-data
collection, the TensorFlow/scikit-learn training scripts, dataset tooling — is
**not** part of this codebase. The shipped `ml_weights.py` still works for
inference, but it cannot be retrained as-is.

### How to rebuild training

To turn this foundation back into a trainable system you would:

1. **Collect labeled CSI.** Capture turbulence windows tagged `idle` vs `motion`
   (and any other classes you want).
2. **Extract the 12 features** for each window using the *same* logic as
   `src/features.py` — the on-device feature extractor is the contract the model
   must match.
3. **Standardize and train.** Fit a scaler (mean/scale) on the training features,
   then train an MLP with the `12 → 16 → 8 → 1` shape (ReLU hidden, sigmoid out)
   in any framework.
4. **Export weights.** Write the trained scaler and weight/bias matrices into
   `src/ml_weights.py` in the format the file already uses
   (`FEATURE_MEAN`, `FEATURE_SCALE`, `W1`/`B1`, `W2`/`B2`, `W3`/`B3`).
5. **Deploy** with `./me deploy` and set `DETECTION_ALGORITHM = "ml"`.

Because inference is plain Python arithmetic in `ml_detector.py`, the only contract
between training and the device is the feature set (§ above) and the weight-file
layout — keep those identical and any training stack will work.
