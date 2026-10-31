# SeptiSense - laptop backend, ML model and dashboard

Educational / research prototype. **Not a medical device, not clinically validated.**

```
Sensors -> ESP32 (SeptiSense.ino) --Wi-Fi POST--> server.py (SQLite + ML inference + alerts) -> dashboard
```

## Files
| File | Purpose |
|---|---|
| `features.py` | Feature engineering shared by training AND live inference |
| `train_model.py` | Trains + evaluates the model on the PhysioNet 2019 sepsis data |
| `server.py` | Backend: receives telemetry, stores history, runs model, raises alerts, serves dashboard |
| `static/index.html` | Dashboard (no internet/CDN needed) |
| `simulate_device.py` | Pipeline test only - sends SIMULATED data (clearly labelled) |
| `models/` | Created by training: `sepsis_model.joblib`, `metrics.json`, `evaluation.png` |

## Step 1 - Install Python packages (once)
Install Python 3.10+ from python.org (tick "Add to PATH"), then in a terminal inside this folder:
```
pip install -r requirements.txt
```

## Step 2 - Get the dataset (free)
1. Go to https://physionet.org/content/challenge-2019/1.0.0/ (PhysioNet/Computing in Cardiology Challenge 2019, open access).
2. Download the training database (about 42 MB): **training_setA** and **training_setB**.
3. Unzip so you have one folder containing both, e.g. `C:\data\physionet2019\training_setA` and `...\training_setB` (each holds ~20,000 `.psv` files).
4. Read the licence/terms on that page and cite the dataset (DOI 10.13026/v64v-d857) in your submission.

**Why this dataset is appropriate:** ICU patients, hourly time series; Sepsis-3 based labels; includes HR, O2Sat and Temp (what your hardware measures); a clearly defined label timing (the label turns 1 six hours before onset); ~40,000 patients from two separate hospitals (lets you test on a different hospital); patient IDs (lets you split by patient to avoid leakage); open access; published benchmark with a paper you can cite.

## Step 3 - Train and evaluate
Quick test on a small subset first (about a minute):
```
python train_model.py --data-dir C:\data\physionet2019 --sample-patients 500
```
Full training (several minutes):
```
python train_model.py --data-dir C:\data\physionet2019
```
It prints and saves held-out results (`models/metrics.json`, `models/evaluation.png`).
- **Prediction target:** sepsis onset within the next 6 hours.
- **Split:** by patient. Hospital A -> 70% train / 15% validation / 15% test. Hospital B -> entirely unseen external test.
- **Metrics:** AUROC, AUPRC (compare with prevalence - classes are very imbalanced), calibration (Brier, ECE), sensitivity/specificity/precision at the alert threshold, patient-level alarm rates, 95% bootstrap CIs, plus a logistic-regression baseline.
- Leakage precautions: patient-level split with an assertion; rows after onset removed; calibration and threshold chosen on validation patients only.

## Step 4 - Start the backend
```
python server.py
```
Open http://localhost:8000. It prints the address the ESP32 must post to.

Real-time mode uses **1-hour bins** exactly like the training data, so the model needs about 4+ hours of continuous data before it can predict. For a short demo use:
```
python server.py --demo-fast
```
which treats every **60 seconds as one "hour"**. The dashboard shows a red banner while this is on - predictions in this mode are **not clinically meaningful**; it only demonstrates the pipeline.

Windows may ask to allow Python through the firewall - allow it on *Private* networks.

## Step 5 - Connect the ESP32
1. On the laptop: Settings > Network > **Mobile hotspot** > ON, band **2.4 GHz** (ESP32 cannot use 5 GHz). The laptop is then always `192.168.137.1`.
2. In `SeptiSense.ino` set `STA_SSID` and `STA_PASS` to the hotspot's name/password. `BACKEND_URL` is already `http://192.168.137.1:8000/api/telemetry`.
3. Upload. The ESP32 still hosts its own `ESP32_LED_Control` network/website for the phone and additionally sends data to the laptop every 5 s. (If the phone briefly disconnects when the ESP32 joins the hotspot, reconnect it.)
4. Serial Monitor shows `Telemetry: Wi-Fi OK | sent batches: N | last HTTP code: 200`.
5. If the laptop is unreachable the ESP32 keeps up to 60 samples and sends them when it reconnects.

No ESP32 handy? `python simulate_device.py` sends SIMULATED data to test the pipeline (dashboard flags it).

## Dashboard
Current readings with validity badges, risk estimate with alert threshold and forecast horizon, trends (HR, SpO2, temperature, risk), per-sensor data-quality/connectivity table, configurable alert settings, alert history with acknowledge, and a model card showing the held-out results.

## API (for your documentation)
`POST /api/telemetry` - from ESP32 | `GET /api/status` | `GET /api/latest` | `GET /api/history?minutes=60` | `GET /api/predictions` | `GET /api/alerts` | `POST /api/alerts/ack` | `GET|POST /api/config` | `GET /api/model`

## Limitations - say these in your submission
- **Only 3 input signals.** The model uses HR, SpO2 and temperature, because that is what the hardware measures. Published sepsis models used up to 40 variables including blood pressure, respiration and lab tests, so expect modest performance; judge it by the held-out numbers you actually get.
- **Temperature mismatch.** Training temperatures are mostly core body temperature; the MLX90614 measures *surface* temperature, which reads lower. The offset setting is a rough demo correction, not a calibration.
- **Population mismatch.** The data are ICU patients; a healthy volunteer is outside what the model was trained on, so their risk values mean little.
- **Optical HR/SpO2** come from a hobby sensor with an uncalibrated algorithm; the ECG heart rate is an experimental R-peak detector.
- **Time resolution.** Training used hourly summaries; live mode averages telemetry into the same bins. `--demo-fast` breaks this on purpose for demonstration.
- Row-level evaluation (every hour is a sample); no prospective or clinical validation.
- No authentication or encryption on the backend; run it only on a trusted local network.
