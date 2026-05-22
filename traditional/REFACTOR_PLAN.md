# Refactor: slim down the codebase and rename to ESPMotion

## Context

`/workspaces/espmotion` is a working ESP32 WiFi-CSI motion detector (MicroPython on the
device, a Python `./me` CLI on the host). It currently carries a full research platform:
11 host-side analysis/ML-training scripts, Jupyter notebooks, labeled datasets, TFLite
models, a CSI UDP streaming mode, host-side data collection (with git-username extraction),
a 16-file pytest suite, and an interactive MQTT tuning console.

The user only needs this as a **final project for a "Wireless Network" course**, running on
a **single original ESP32**. The deployed firmware already works (`./me deploy`, `./me run`
— CSI captured, motion detected, MQTT published). The goal is a **minimal but still-working**
codebase: strip the research scaffolding, keep the core detection pipeline intact, keep a
**foundation** of the ML detector for the user to rebuild later, rename everything from
"Micro-ESPectre" to **ESPMotion**, and add clear documentation.

Decisions confirmed with the user: keep the web monitor (drop the theremin toy); remove the
interactive MQTT console; remove the entire test suite.

## Naming convention

Every variant — `Micro-ESPectre`, `Micro ESPectre`, `ESPectre`, `micro-espectre`,
`espectre` — becomes **`ESPMotion`** (display/title) or **`espmotion`** (identifiers,
topics, filenames).

## Part 1 — Delete (research scaffolding & unused capabilities)

Delete these entirely:

- `tools/` — all 11 analysis/training scripts, `csi_utils.py` (host-side CSI/data
  collection + git-username extraction), `README.md`, `__init__.py`.
- `notebooks/` — Jupyter notebooks + README.
- `data/` — `baseline/`, `movement/`, `test/`, `dataset_info.json`, `DATASET_QUALITY_CHECK.md`.
- `models/` — `.tflite` + `.npz` files.
- `mosquitto/` — local broker config (user connects to their own broker).
- `tests/` — entire 16-file pytest suite.
- `.coveragerc`, `.github/` (dependabot) — dev-only infra.
- `src/csi_streamer.py` — UDP CSI streaming (only invoked by the removed `me stream`).
- `src/mqtt/commands.py` — remote command handler; never reached (firmware is publish-only).
- `espectre-theremin.html` — experimental toy.
- `ALGORITHMS.md`, `ML_DATA_COLLECTION.md` — replaced by new docs (Part 7).

Keep: `.devcontainer/` (the Codespaces environment), `firmware/ESP32_CSI.bin` (needed to
flash; only the ESP32 binary is present), `venv/`, `.gitignore`.

## Part 2 — ML detector: keep as a foundation

The user will rebuild ML themselves later, so **keep the on-device ML path** (it is already
a clean, self-contained foundation — only the heavy *training* side is being removed):

- Keep `src/ml_detector.py`, `src/ml_weights.py`, `src/features.py`, `src/detector_interface.py`
  (`detector_interface.py` and `features.py` are also used by the MVS path).
- `src/main.py` keeps the lazy `MLDetector` import and the `DETECTION_ALGORITHM = "ml"`
  option in `config.py`.
- Add a short header comment to `src/ml_detector.py`: it is a working inference foundation
  (MLP 12→16→8→1 on pre-trained `ml_weights.py`); the training pipeline was removed and can
  be rebuilt — see `docs/HOW_IT_WORKS.md`.

## Part 3 — Trim the `./me` CLI

Keep subcommands: `flash`, `deploy`, `run`, `verify`, `ui`.

Remove from `me`:
- `stream` + `collect` subparsers and their handler functions (`stream_csi`, `collect_csi_data`).
- Interactive MQTT mode: `EspectreCLI_MQTT` class, the `--broker/--port-mqtt/--topic/
  --username/--password` args, the `CompactDumper` yaml helper, and the no-subcommand
  routing. With no subcommand, print `--help`.
- Now-unused imports: `yaml`, `paho.mqtt`, `prompt_toolkit`, `dotenv` (+ its `load_dotenv()`
  call). Keep `colorama` (used for colored output by flash/deploy/run).
- Update the banner and example/help text; rename the `espectre-monitor.html` reference in
  `open_web_ui()` to `espmotion-monitor.html`.

Leave `flash`'s multi-chip auto-detect logic untouched — it is self-contained, works, and
harmlessly falls through to ESP32. Check `deploy` and `verify` for any hardcoded file list
that names `csi_streamer.py` and update it if present.

## Part 4 — MQTT: make the handler cleanly publish-only

`src/mqtt/handler.py` currently depends on the deleted `commands.py`:

- Remove `from src.mqtt.commands import MQTTCommands`, the `cmd_handler` field, `_on_message`,
  and `check_messages` (already no-ops — the firmware never subscribes).
- Replace `publish_info()` with a small inline JSON publish (device id, algorithm, threshold,
  window size) to `{topic}/info`, so the web monitor's info display still works.
- Simplify the constructor to `MQTTHandler(config, detector, wlan)`; update the call site in
  `src/main.py` (drop the `run_band_calibration`/`g_state` args, which only fed commands).
- `src/mqtt/__init__.py` — remove the `from .commands import MQTTCommands` export.

`publish_state()` (the core periodic publish) is unchanged.

## Part 5 — Rename everything to ESPMotion

- `src/config.py`: `MQTT_CLIENT_ID = "espmotion"`, `MQTT_TOPIC = "home/espmotion/node1"`,
  docstring.
- All `src/**/*.py` module docstrings/comments: `Micro-ESPectre -> ESPMotion`.
- `src/main.py`: startup print + the ASCII banner → a simple `ESPMotion` banner.
- `me`: banner + help/example text.
- `espectre-monitor.html` → rename file to `espmotion-monitor.html`; inside it update the
  MQTT topic `home/espectre/node1 -> home/espmotion/node1`, client id prefix
  `espectre_monitor_ -> espmotion_monitor_`, and page title.
- Do **not** edit `src/config_local.py` (gitignored user secrets). Note for the user: if it
  overrides `MQTT_TOPIC`, update it there too, and update any broker/Home-Assistant
  subscriber bound to the old `home/espectre/node1` topic.

## Part 6 — Trim `requirements.txt`

Keep only: `esptool==5.2.0`, `mpremote>=1.26.1`, `pyserial>=3.5`, `colorama>=0.4.6`.
Drop: `paho-mqtt`, `python-dotenv`, `PyYAML`, `prompt_toolkit`, `numpy`, `matplotlib`,
`scipy`, `PyWavelets`, `pytest`, `pytest-cov`, `gcovr`, `tensorflow`, `scikit-learn`, `shap`.

## Part 7 — Documentation

- Rewrite `README.md` — concise: what ESPMotion is, hardware needed, quick start
  (flash → configure `config_local.py` → deploy → run), a short config summary, links to
  the two docs below.
- New `docs/USAGE.md` — **how to use it**: prerequisites, install deps, flash firmware,
  create `config_local.py` (WiFi + MQTT), `./me deploy` / `./me run`, how to read the
  console output, the MQTT JSON payload schema, the web monitor (`./me ui`), troubleshooting.
- New `docs/HOW_IT_WORKS.md` — **how it works end-to-end**: WiFi CSI primer → ESP32 CSI
  capture → traffic generator (why it's needed) → gain lock & CV-normalization fallback
  (the original ESP32 has no gain-lock support, so it uses the CV path) → NBVI band
  calibration → spatial turbulence → Hampel / low-pass filters → moving-variance
  segmentation + IDLE/MOTION state machine → MQTT publishing. Final section: "ML detector
  (foundation)" — the MLP architecture, the 12 features, and how to rebuild training.
  This consolidates the deleted `ALGORITHMS.md`.

## Final structure (after refactor)

```
espmotion/
  me                      flash / deploy / run / verify / ui
  requirements.txt        4 deps
  README.md
  docs/USAGE.md
  docs/HOW_IT_WORKS.md
  espmotion-monitor.html
  firmware/ESP32_CSI.bin
  src/
    main.py  config.py  config_local.py(.example)
    detector_interface.py  mvs_detector.py  segmentation.py
    filters.py  threshold.py  nbvi_calibrator.py
    traffic_generator.py  utils.py
    features.py  ml_detector.py  ml_weights.py     # ML foundation
    mqtt/__init__.py  mqtt/handler.py              # publish-only
  .devcontainer/
```

## Verification

1. `./me deploy` then `./me run` on the original ESP32 — confirm: WiFi connects, gain
   lock reports "not supported → CV normalization", NBVI band calibration completes, CSI
   packets flow, the loop prints `IDLE`/`MOTION` with a progress bar, and MQTT publishes
   to `home/espmotion/node1`.
2. Temporarily set `DETECTION_ALGORITHM = "ml"` in `config.py`, redeploy, and confirm the
   ML foundation still imports and runs; revert to `"mvs"`.
3. `./me ui` — open `espmotion-monitor.html`, confirm it shows live state from the renamed
   topic.
4. Grep `src/` and `me` for leftover `espectre`/`csi_streamer`/`commands`/`MQTTCommands`
   references and dangling imports — expect none.
5. `./me verify` passes.
