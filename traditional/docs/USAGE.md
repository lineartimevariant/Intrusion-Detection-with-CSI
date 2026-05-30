# ESPMotion — Usage

How to install, flash, configure, and operate ESPMotion. For *how the detection
works*, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## 1. Prerequisites

- An **ESP32** with CSI support and a USB data cable (not charge-only).
- A **2.4 GHz WiFi** network you have the password for.
- An **MQTT broker** (optional — detection runs without one, you just won't get
  MQTT output or the web monitor).
- **Python 3.12** on the host computer.

## 2. Install host dependencies

The `./me` CLI needs four Python packages. A virtual environment keeps them
isolated:

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs `esptool` (flashing), `mpremote` (deploying),
`pyserial` (port detection), and `colorama` (colored output).

Re-activate the venv (`source venv/bin/activate`) in every new terminal.

## 3. Flash the firmware

The device runs a MicroPython build with the CSI API enabled. The correct binary
is downloaded automatically from the
[micropython-esp32-csi releases](https://github.com/francescopace/micropython-esp32-csi/releases)
and cached under `firmware/`.

```bash
./me flash --erase
```

The CLI auto-detects the serial port and chip type. If auto-detection fails:

```bash
./me flash --chip esp32 --port /dev/ttyUSB0 --erase
```

If the chip will not respond: hold **BOOT**, tap **RESET**, release **BOOT**, then
retry.

Confirm the firmware is good:

```bash
./me verify
```

`verify` checks that the CSI methods exist on the WLAN object, reports the
MicroPython version, and lists the deployed files.

## 4. Configure WiFi and MQTT

Create your local config from the template (this file is gitignored — it holds
your secrets):

```bash
cp src/config_local.py.example src/config_local.py
```

Edit `src/config_local.py`:

```python
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"

MQTT_BROKER = "homeassistant.local"   # broker hostname or IP
MQTT_PORT = 1883
MQTT_USERNAME = "username"
MQTT_PASSWORD = "password"
```

Any `UPPER_CASE` name set here overrides the default in `src/config.py`. To change
the MQTT topic, add `MQTT_TOPIC = "home/espmotion/node1"` here as well.

If you previously ran the project under the old `home/espectre/node1` topic,
update any broker rule or Home-Assistant sensor that subscribed to it.

## 5. Deploy and run

```bash
./me deploy     # uploads src/ to the device (~5 seconds, no compile)
./me run        # starts the detector and streams its console output
```

`./me run` keeps the serial console attached. Press **Ctrl+C** to stop — the CLI
resets the ESP32 to release WiFi/CSI state cleanly.

## 6. Reading the console output

At startup the device prints its progress: WiFi connection, gain-lock result, NBVI
band calibration, and CSI verification. **Keep the room still for the first ~10
seconds** so calibration measures a clean baseline.

Once running, each publish interval prints one line:

```
[████████████░░░|░░░░░] 80% | pkts:100 drop:0 pps:104 | mvmt:0.8124 thr:1.0150 | IDLE
```

| Field | Meaning |
|-------|---------|
| `[…] 80%` | Progress bar — current movement metric relative to the threshold (the `|` marks the threshold) |
| `pkts:100` | CSI packets processed since the last line |
| `drop:0`   | CSI packets dropped since the last line |
| `pps:104`  | CSI packets per second |
| `mvmt:0.8124` | Movement metric (moving variance) |
| `thr:1.0150`  | Detection threshold |
| `IDLE` / `MOTION` | Current detection state |

`MOTION` appears when `mvmt` rises above `thr`.

## 7. MQTT payloads

ESPMotion is **publish-only** — it never subscribes or accepts commands.

**State** — published every publish interval to the base topic (`MQTT_TOPIC`,
default `home/espmotion/node1`):

```json
{
  "movement": 0.8124,
  "threshold": 1.0150,
  "state": "idle",
  "packets_processed": 100,
  "packets_dropped": 0,
  "pps": 104,
  "timestamp": 1700000000
}
```

`state` is `"idle"` or `"motion"`.

**Info** — published once at startup to `<MQTT_TOPIC>/info`:

```json
{
  "device": "espmotion",
  "algorithm": "MVS",
  "threshold": 1.0150,
  "window_size": 75
}
```

### Home Assistant example

```yaml
mqtt:
  binary_sensor:
    - name: "ESPMotion"
      state_topic: "home/espmotion/node1"
      value_template: "{{ value_json.state }}"
      payload_on: "motion"
      payload_off: "idle"
      device_class: motion
```

## 8. Web monitor

```bash
./me ui
```

This opens `espmotion-monitor.html` in your browser. Enter your MQTT broker's
**WebSocket** host/port and the topic (`home/espmotion/node1`), then connect to see
the live movement/threshold chart and the current state.

> **Chrome:** if the WebSocket connection to a `.local` broker is blocked, enable
> `chrome://flags/#local-network-access-check-websockets` and restart Chrome.

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `No serial ports found` | Bad/charge-only cable, or missing USB-serial driver. Try another cable/port. |
| `Device is not running a valid MicroPython firmware` | Firmware missing or corrupt — re-run `./me flash --erase`. |
| `CSI methods not found` (from `./me verify`) | Wrong firmware flashed — use `./me flash --erase` to get the CSI build. |
| WiFi connect timeout | Wrong SSID/password, or a 5 GHz-only network — ESPMotion needs 2.4 GHz. |
| `Signal too strong … skipping gain lock` | Normal on the original ESP32 — it falls back to CV normalization. Move the device 2–3 m from the router for best results. |
| `No CSI for 6s/20s` warnings | Network stack stalled; the firmware self-heals by re-associating WiFi or rebooting. |
| `MQTT unavailable … continuing` | Broker unreachable — detection still runs, only MQTT output is lost. Check broker host/port/credentials. |
| Constant false `MOTION` | Calibrated while the room was not still — restart and stay out of the area for the first ~10 s. |
