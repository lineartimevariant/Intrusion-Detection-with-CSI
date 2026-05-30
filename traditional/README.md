# 🛜 ESPMotion

**WiFi CSI motion detection on the ESP32 — pure Python (MicroPython).**

ESPMotion turns a single ESP32 into a motion sensor with no cameras and no PIR.
It reads the *Channel State Information* (CSI) of WiFi packets — the fine-grained
amplitude/phase of each subcarrier — and detects when a person moves through the
radio path between the ESP32 and the WiFi router. Detection results are published
over MQTT and can be watched live in a browser.

The device firmware runs entirely in MicroPython; a small host-side CLI (`./me`)
flashes, deploys, and runs it.

## How it works (in one paragraph)

The ESP32 connects to your WiFi, generates a steady stream of background traffic
so CSI keeps flowing, and captures the CSI of every received packet. Per packet it
computes a *spatial turbulence* value across 12 subcarriers; the moving variance of
that signal over a sliding window is compared to a threshold to decide `IDLE` vs
`MOTION`. The threshold and the 12 subcarriers are auto-calibrated at boot. See
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full pipeline.

## Hardware needed

- An **ESP32** with CSI support (the original ESP32 works; C3/S3/C5/C6 also supported by the firmware).
- A **2.4 GHz WiFi** network.
- An **MQTT broker** (e.g. Mosquitto, or the Home Assistant add-on) — optional, detection still runs without it.
- A host computer with **Python 3.12** and a USB cable to the ESP32.

## Quick start

```bash
# 1. Install host dependencies (a venv is recommended)
pip install -r requirements.txt

# 2. Flash the MicroPython + CSI firmware (once per device)
./me flash --erase

# 3. Configure WiFi + MQTT credentials
cp src/config_local.py.example src/config_local.py
#   then edit src/config_local.py

# 4. Deploy the application code to the device
./me deploy

# 5. Run it
./me run
```

On `./me run` the ESP32 connects to WiFi, calibrates (keep the room still for
~10 s), then prints a live `IDLE`/`MOTION` line and publishes to MQTT.

Open the web monitor with `./me ui`.

Full step-by-step instructions, the MQTT payload schema, and troubleshooting are
in [docs/USAGE.md](docs/USAGE.md).

## The `./me` CLI

| Command | What it does |
|---------|--------------|
| `./me flash`  | Flash the MicroPython CSI firmware to the ESP32 |
| `./me deploy` | Upload the `src/` application code to the device |
| `./me run`    | Run the detector and stream its console output |
| `./me verify` | Check that firmware + code are correctly installed |
| `./me ui`     | Open the web monitor (`espmotion-monitor.html`) |

## Configuration summary

All defaults live in `src/config.py`; your secrets and overrides go in
`src/config_local.py` (gitignored). Key settings:

| Setting | Default | Purpose |
|---------|---------|---------|
| `WIFI_SSID` / `WIFI_PASSWORD` | — | WiFi credentials (set in `config_local.py`) |
| `MQTT_BROKER` / `MQTT_PORT` | `homeassistant.local` / `1883` | MQTT broker |
| `MQTT_TOPIC` | `home/espmotion/node1` | Topic state is published to |
| `SEG_THRESHOLD` | `auto` | `auto`, `min`, or a fixed number |
| `SEG_WINDOW_SIZE` | `75` | Moving-variance window, in packets |
| `TRAFFIC_GENERATOR_RATE` | `100` | Background packets/sec (keeps CSI flowing) |

## Detection algorithm

**MVS** — Moving Variance Segmentation. Fast, no training needed, auto-calibrates
the threshold and the 12 subcarriers at boot. See
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full pipeline.

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — how to install, flash, configure, and operate ESPMotion.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — the detection pipeline, end to end.

## Credits

Built on [micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi),
a MicroPython fork that exposes the ESP32's CSI API to Python.
