#!/usr/bin/env python3
"""
PIPELINE TEST ONLY. Sends SIMULATED telemetry (device id 'simulator-01') so you can
check the backend + dashboard without the ESP32. The dashboard labels such data as
SIMULATED. Never use this for the real demo of sensor readings.
"""
import json, random, time, urllib.request, sys

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api/telemetry"
hr, spo2, temp = 75.0, 97.0, 36.6
t0 = time.time()
while True:
    hr += random.gauss(0, 0.8); spo2 += random.gauss(0, 0.15); temp += random.gauss(0, 0.02)
    hr = min(max(hr, 55), 130); spo2 = min(max(spo2, 90), 100); temp = min(max(temp, 35.5), 39.5)
    ms = int((time.time() - t0) * 1000)
    rec = {"t": ms, "mlx": 1, "tOK": 1, "obj": round(temp, 2), "amb": 24.0, "led": 0,
           "max": 1, "fin": 1, "hrOK": 1, "hr": round(hr, 1), "spo2OK": 1, "spo2": round(spo2),
           "ecgOK": 1, "ecg": round(hr + random.gauss(0, 1), 1), "ecgQ": 7, "lo": 0}
    body = json.dumps({"device_id": "simulator-01", "sent_ms": ms, "records": [rec]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        print(urllib.request.urlopen(req, timeout=3).read().decode())
    except Exception as e:
        print("send failed:", e)
    time.sleep(5)
