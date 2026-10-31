#!/usr/bin/env python3
"""
SeptiSense laptop backend  (Python standard library + numpy/pandas/scikit-learn)

  ESP32  --Wi-Fi/HTTP POST-->  this server  -->  SQLite  -->  ML inference  -->  dashboard

Run:   python server.py                 (real time: 1 model-bin = 1 hour)
       python server.py --demo-fast     (DEMO ONLY: 1 model-bin = 60 seconds)

Dashboard: http://localhost:8000
"""
import argparse
import csv
import io
import json
import re
import socket
import sqlite3
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).parent
STATIC = BASE / "static"
DB_PATH = BASE / "septisense.db"
MODEL_PATH = BASE / "models" / "sepsis_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

BIN_SECONDS = 3600
INFER_EVERY_S = 10
MIN_OBSERVED_BINS = 4          # of the last WINDOW bins
LOOKBACK_BINS = 24

DEFAULT_CONFIG = {
    "alert_enabled": 1,
    "alert_threshold": 0.5,        # replaced by the model's threshold when available
    "consecutive_required": 2,     # predictions in a row above threshold
    "cooldown_seconds": 600,
    "temp_offset_c": 0.0,          # surface -> core correction (see README!)
    "offline_seconds": 45,
}
CONFIG_LIMITS = {
    "alert_enabled": (0, 1), "alert_threshold": (0.001, 0.999),
    "consecutive_required": (1, 10), "cooldown_seconds": (0, 86400),
    "temp_offset_c": (-5.0, 10.0), "offline_seconds": (10, 600),
}

MODEL = None            # loaded bundle
MODEL_ERROR = None
STATE = {"streak": {}, "last_alert": {}, "offline_flag": {}}
STATE_LOCK = threading.Lock()


# ----------------------------------------------------------------- database
def db():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS telemetry(
          id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT, ts REAL, received REAL,
          source TEXT, mlx_ok INT, obj REAL, amb REAL, led INT,
          max_ok INT, finger INT, hr REAL, spo2 REAL,
          ecg_bpm REAL, ecg_q INT, lead_off INT);
        CREATE INDEX IF NOT EXISTS ix_tel ON telemetry(device_id, ts);
        CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, device_id TEXT, status TEXT,
          reason TEXT, risk REAL, level TEXT, inputs TEXT, bins_used INT, bin_seconds INT);
        CREATE TABLE IF NOT EXISTS alerts(
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, device_id TEXT, type TEXT,
          level TEXT, message TEXT, risk REAL, acknowledged INT DEFAULT 0);
        CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value REAL);
        CREATE TABLE IF NOT EXISTS sessions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, note TEXT,
          started REAL NOT NULL, ended REAL);
        """)
        # older databases (before test sessions existed): add the new column
        for table in ("telemetry", "predictions", "alerts"):
            have = [r["name"] for r in c.execute(f"PRAGMA table_info({table})")]
            if "session_id" not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN session_id INTEGER")
        c.executescript("""
        CREATE INDEX IF NOT EXISTS ix_tel_sess ON telemetry(session_id, ts);
        CREATE INDEX IF NOT EXISTS ix_pred_sess ON predictions(session_id, ts);
        CREATE INDEX IF NOT EXISTS ix_alert_sess ON alerts(session_id);
        """)


def get_config():
    cfg = dict(DEFAULT_CONFIG)
    if MODEL is not None:
        cfg["alert_threshold"] = round(float(MODEL["threshold"]), 4)
    with db() as c:
        for r in c.execute("SELECT key,value FROM config"):
            if r["key"] in cfg:
                cfg[r["key"]] = r["value"]
    return cfg


def num_or_none(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v or v < lo or v > hi:
        return None
    return v


# ------------------------------------------------------------ test sessions
# A "test" (session) is one recording run for one patient/subject. Readings,
# predictions and alerts are tagged with the session they belong to, so past
# tests can be browsed, exported or deleted, and a new test starts clean.
# Readings that arrive while no test is running are stored as "unassigned"
# (session_id NULL) so live monitoring still works without starting a test.
LABEL_MAX, NOTE_MAX = 60, 500


def clean_text(v, maxlen):
    s = "".join(ch for ch in str(v or "") if ch.isprintable())
    return s.strip()[:maxlen]


def active_session(c):
    r = c.execute("SELECT * FROM sessions WHERE ended IS NULL ORDER BY id DESC LIMIT 1").fetchone()
    return dict(r) if r else None


def scope_sql(c, alias=""):
    """SQL condition selecting the 'current' data: the running test, or
    the unassigned readings when no test is running."""
    col = f"{alias}session_id"
    s = active_session(c)
    return (f"{col}=?", [s["id"]]) if s else (f"{col} IS NULL", [])


def reset_runtime_state():
    with STATE_LOCK:                      # a new/ended test must not inherit alert streaks
        STATE["streak"].clear()
        STATE["last_alert"].clear()


SESSION_SELECT = """
SELECT s.id, s.label, s.note, s.started, s.ended,
  (SELECT COUNT(*) FROM telemetry t WHERE t.session_id=s.id) AS n_samples,
  (SELECT COUNT(*) FROM telemetry t WHERE t.session_id=s.id AND t.source='simulator') AS n_sim,
  (SELECT MIN(t.device_id) FROM telemetry t WHERE t.session_id=s.id) AS device_id,
  (SELECT AVG(t.hr) FROM telemetry t WHERE t.session_id=s.id) AS avg_hr,
  (SELECT AVG(t.spo2) FROM telemetry t WHERE t.session_id=s.id) AS avg_spo2,
  (SELECT AVG(t.obj) FROM telemetry t WHERE t.session_id=s.id) AS avg_temp,
  (SELECT COUNT(*) FROM predictions p WHERE p.session_id=s.id AND p.status='ok') AS n_pred,
  (SELECT MAX(p.risk) FROM predictions p WHERE p.session_id=s.id AND p.status='ok') AS max_risk,
  (SELECT COUNT(*) FROM alerts a WHERE a.session_id=s.id) AS n_alerts,
  (SELECT COUNT(*) FROM alerts a WHERE a.session_id=s.id AND a.type='RISK') AS n_risk_alerts
FROM sessions s"""


def get_session(sid):
    with db() as c:
        r = c.execute(SESSION_SELECT + " WHERE s.id=?", (sid,)).fetchone()
    return dict(r) if r else None


def sessions_payload(limit=200):
    with db() as c:
        rows = c.execute(SESSION_SELECT + " ORDER BY s.started DESC LIMIT ?", (limit,)).fetchall()
    return {"server_time": time.time(), "sessions": [dict(r) for r in rows]}


def start_session(label, note):
    now = time.time()
    with db() as c:
        c.execute("UPDATE sessions SET ended=? WHERE ended IS NULL", (now,))   # one test at a time
        label = clean_text(label, LABEL_MAX)
        if not label:
            n = c.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"] + 1
            label = f"Test {n}"
        cur = c.execute("INSERT INTO sessions(label,note,started) VALUES(?,?,?)",
                        (label, clean_text(note, NOTE_MAX), now))
        sid = cur.lastrowid
    reset_runtime_state()
    return get_session(sid)


def end_session():
    with db() as c:
        s = active_session(c)
        if not s:
            return None
        c.execute("UPDATE sessions SET ended=? WHERE id=?", (time.time(), s["id"]))
    reset_runtime_state()
    return get_session(s["id"])


def update_session(sid, label, note):
    with db() as c:
        if not c.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
            return None
        if label is not None:
            label = clean_text(label, LABEL_MAX)
            if label:
                c.execute("UPDATE sessions SET label=? WHERE id=?", (label, sid))
        if note is not None:
            c.execute("UPDATE sessions SET note=? WHERE id=?", (clean_text(note, NOTE_MAX), sid))
    return get_session(sid)


def delete_session(sid):
    """Returns 'ok', 'not_found' or 'active' (a running test cannot be deleted)."""
    with db() as c:
        r = c.execute("SELECT ended FROM sessions WHERE id=?", (sid,)).fetchone()
        if not r:
            return "not_found"
        if r["ended"] is None:
            return "active"
        for t in ("telemetry", "predictions", "alerts"):
            c.execute(f"DELETE FROM {t} WHERE session_id=?", (sid,))
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return "ok"


def session_detail(sid):
    s = get_session(sid)
    if not s:
        return None
    with db() as c:
        tel = c.execute("SELECT ts,hr,spo2,obj,ecg_bpm FROM telemetry WHERE session_id=? ORDER BY ts",
                        (sid,)).fetchall()
        pred = c.execute("SELECT ts,risk,level FROM predictions WHERE session_id=? AND status='ok' "
                         "ORDER BY ts", (sid,)).fetchall()
        alerts = [dict(r) for r in c.execute(
            "SELECT * FROM alerts WHERE session_id=? ORDER BY ts DESC LIMIT 200", (sid,))]
    ts_, ps_ = max(1, len(tel) // 1200), max(1, len(pred) // 1200)
    return {"session": s, "server_time": time.time(),
            "alert_threshold": get_config()["alert_threshold"],
            "telemetry": [dict(r) for r in tel[::ts_]],
            "predictions": [dict(r) for r in pred[::ps_]],
            "alerts": alerts}


EXPORT_COLUMNS = {
    "telemetry": "ts,device_id,source,hr,spo2,obj,amb,ecg_bpm,finger,lead_off,led,mlx_ok,max_ok",
    "predictions": "ts,status,reason,risk,level,inputs,bins_used,bin_seconds",
    "alerts": "ts,type,level,message,risk,acknowledged",
}


def export_csv(sid, kind):
    """Returns (filename, bytes) or None."""
    s = get_session(sid)
    if not s or kind not in EXPORT_COLUMNS:
        return None
    cols = EXPORT_COLUMNS[kind].split(",")
    with db() as c:
        rows = c.execute(f"SELECT {EXPORT_COLUMNS[kind]} FROM {kind} WHERE session_id=? ORDER BY ts",
                         (sid,)).fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    nice = {"obj": "surface_temp_c", "amb": "ambient_temp_c", "ts": "unix_time"}
    w.writerow([nice.get("ts")] + ["time_local"] + [nice.get(k, k) for k in cols[1:]])
    for r in rows:
        w.writerow([r["ts"], datetime.fromtimestamp(r["ts"]).isoformat(timespec="seconds")]
                   + [r[k] for k in cols[1:]])
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", s["label"])[:30].strip("_") or "test"
    return f"septisense_test{sid}_{safe}_{kind}.csv", buf.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------- telemetry
def ingest(payload):
    device = str(payload.get("device_id", "unknown"))[:40]
    now = time.time()
    sent_ms = payload.get("sent_ms")
    rows = []
    for r in payload.get("records", []):
        # ESP32 has no real-time clock: reconstruct sample time from the
        # device's millisecond counter relative to the moment it was sent.
        try:
            age = (float(sent_ms) - float(r["t"])) / 1000.0
        except (TypeError, ValueError, KeyError):
            age = 0.0
        age = min(max(age, 0.0), 6 * 3600)
        ts = now - age
        hr = num_or_none(r.get("hr"), 30, 220) if r.get("hrOK") else None
        spo2 = num_or_none(r.get("spo2"), 70, 100) if r.get("spo2OK") else None
        ecg = num_or_none(r.get("ecg"), 30, 220) if r.get("ecgOK") else None
        obj = num_or_none(r.get("obj"), -20, 80) if r.get("tOK") else None
        amb = num_or_none(r.get("amb"), -20, 80) if r.get("tOK") else None
        rows.append((device, ts, now, "simulator" if device.startswith("sim") else "device",
                     int(bool(r.get("mlx"))), obj, amb, int(bool(r.get("led"))),
                     int(bool(r.get("max"))), int(bool(r.get("fin"))), hr, spo2,
                     ecg, int(r.get("ecgQ", -1)), int(bool(r.get("lo")))))
    if not rows:
        return 0
    with db() as c:
        # assign each reading to the test that was running when it was SAMPLED
        # (not when it arrived), so readings queued during a Wi-Fi drop land correctly
        sess = c.execute("SELECT id,started,ended FROM sessions WHERE ended IS NULL OR ended>? "
                         "ORDER BY started DESC", (min(r[1] for r in rows),)).fetchall()

        def sid_for(ts):
            for s_ in sess:
                if s_["started"] <= ts and (s_["ended"] is None or ts < s_["ended"]):
                    return s_["id"]
            return None
        rows = [r + (sid_for(r[1]),) for r in rows]
        c.executemany("""INSERT INTO telemetry(device_id,ts,received,source,mlx_ok,obj,amb,led,
                         max_ok,finger,hr,spo2,ecg_bpm,ecg_q,lead_off,session_id)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    return len(rows)


# ---------------------------------------------------------------- inference
def load_model():
    global MODEL, MODEL_ERROR
    try:
        import joblib
        MODEL = joblib.load(MODEL_PATH)
        MODEL_ERROR = None
        print(f"Model loaded: {MODEL_PATH.name} (alert threshold {MODEL['threshold']:.3f})")
    except FileNotFoundError:
        MODEL_ERROR = "No trained model found. Run train_model.py first."
        print("WARNING:", MODEL_ERROR)
    except Exception as e:                       # noqa
        MODEL_ERROR = f"Model failed to load: {e}"
        print("WARNING:", MODEL_ERROR)


def predict_device(device_id, now, cfg):
    """Returns a dict describing the prediction (or why none is possible)."""
    if MODEL is None:
        return {"status": "no_model", "reason": MODEL_ERROR or "model not loaded"}
    import numpy as np
    import pandas as pd
    from features import FILL_LIMIT, VITALS, WINDOW, add_features

    since = now - LOOKBACK_BINS * BIN_SECONDS
    with db() as c:
        where, args = scope_sql(c)         # a new test never inherits the previous patient's data
        rows = c.execute(f"""SELECT ts,hr,ecg_bpm,spo2,obj FROM telemetry
                            WHERE device_id=? AND ts>=? AND {where} ORDER BY ts""",
                         (device_id, since, *args)).fetchall()
    if not rows:
        return {"status": "insufficient", "reason": "no recent telemetry"}

    bins = {}
    for r in rows:
        b = int(r["ts"] // BIN_SECONDS)
        d = bins.setdefault(b, {v: [] for v in VITALS})
        hr = r["hr"] if r["hr"] is not None else r["ecg_bpm"]
        if hr is not None:
            d["HR"].append(hr)
        if r["spo2"] is not None:
            d["O2Sat"].append(r["spo2"])
        if r["obj"] is not None:
            d["Temp"].append(r["obj"] + cfg["temp_offset_c"])
    lo, hi = min(bins), max(bins)
    idx = list(range(lo, hi + 1))
    df = pd.DataFrame({v: [float(np.mean(bins[b][v])) if b in bins and bins[b][v] else np.nan
                           for b in idx] for v in VITALS}, index=idx)

    tail = df.tail(WINDOW)
    if (tail["HR"].notna().sum() < MIN_OBSERVED_BINS):
        return {"status": "insufficient",
                "reason": f"need heart-rate data in at least {MIN_OBSERVED_BINS} of the last "
                          f"{WINDOW} bins ({BIN_SECONDS}s each); have {int(tail['HR'].notna().sum())}"}
    for v in VITALS:                       # staleness check: observed in last 2 bins
        if df[v].tail(2).notna().sum() == 0:
            return {"status": "insufficient", "reason": f"no recent valid {v} reading"}

    dff = df.ffill(limit=FILL_LIMIT).reset_index(drop=True)
    dff["pid"] = "live"
    feats = add_features(dff)
    x = feats.iloc[[-1]].astype("float32")
    from features import REQUIRED_NON_NULL
    if x[REQUIRED_NON_NULL].isna().any(axis=None):
        return {"status": "insufficient", "reason": "not enough history for rolling features"}
    raw = float(MODEL["model"].predict_proba(x[MODEL["features"]])[:, 1][0])
    risk = float(MODEL["calibrator"].predict([raw])[0])
    thr = cfg["alert_threshold"]
    level = "HIGH" if risk >= thr else ("ELEVATED" if risk >= 0.5 * thr else "LOW")
    last = dff.iloc[-1]
    return {"status": "ok", "risk": risk, "level": level,
            "bins_used": int(tail["HR"].notna().sum()),
            "inputs": {"HR": round(float(last["HR"]), 1), "O2Sat": round(float(last["O2Sat"]), 1),
                       "Temp_model_input": round(float(last["Temp"]), 2)}}


def raise_alert(c, now, device, typ, level, msg, risk=None):
    s = active_session(c)
    c.execute("INSERT INTO alerts(ts,device_id,type,level,message,risk,session_id) VALUES(?,?,?,?,?,?,?)",
              (now, device, typ, level, msg, risk, s["id"] if s else None))
    print(f"[ALERT] {typ} {level}: {msg}")


def inference_cycle():
    now = time.time()
    cfg = get_config()
    with db() as c:
        cur_sess = active_session(c)
        cur_sid = cur_sess["id"] if cur_sess else None
        devices = [r["device_id"] for r in c.execute(
            "SELECT DISTINCT device_id FROM telemetry WHERE ts>=?", (now - 24 * 3600,))]
        for dev in devices:
            last = c.execute("SELECT MAX(received) m FROM telemetry WHERE device_id=?",
                             (dev,)).fetchone()["m"]
            # ---- connectivity alert
            offline = (now - last) > cfg["offline_seconds"]
            with STATE_LOCK:
                was = STATE["offline_flag"].get(dev, False)
                STATE["offline_flag"][dev] = offline
            if offline and not was and cfg["alert_enabled"]:
                raise_alert(c, now, dev, "DEVICE_OFFLINE", "WARNING",
                            f"No telemetry from {dev} for over {int(cfg['offline_seconds'])} s.")
            if offline:
                continue
            # ---- prediction
            res = predict_device(dev, now, cfg)
            c.execute("""INSERT INTO predictions(ts,device_id,status,reason,risk,level,inputs,
                         bins_used,bin_seconds,session_id) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                      (now, dev, res["status"], res.get("reason"), res.get("risk"),
                       res.get("level"), json.dumps(res.get("inputs")),
                       res.get("bins_used"), BIN_SECONDS, cur_sid))
            if res["status"] != "ok":
                continue
            with STATE_LOCK:
                streak = STATE["streak"].get(dev, 0)
                streak = streak + 1 if res["risk"] >= cfg["alert_threshold"] else 0
                STATE["streak"][dev] = streak
                last_alert = STATE["last_alert"].get(dev, 0)
                fire = (cfg["alert_enabled"] and streak >= cfg["consecutive_required"]
                        and now - last_alert >= cfg["cooldown_seconds"])
                if fire:
                    STATE["last_alert"][dev] = now
            if fire:
                raise_alert(c, now, dev, "RISK", "HIGH",
                            f"Estimated risk {res['risk']:.1%} is at/above the alert threshold "
                            f"{cfg['alert_threshold']:.1%} (6 h horizon, experimental).", res["risk"])
        c.commit()


def inference_loop():
    while True:
        try:
            inference_cycle()
        except Exception:
            traceback.print_exc()
        time.sleep(INFER_EVERY_S)


# ---------------------------------------------------------------------- API
def status_payload():
    now = time.time()
    cfg = get_config()
    out = {"server_time": now, "bin_seconds": BIN_SECONDS,
           "demo_time_compression": BIN_SECONDS < 3600,
           "model_loaded": MODEL is not None, "model_error": MODEL_ERROR,
           "horizon_hours": MODEL["meta"]["horizon_hours"] if MODEL else 6,
           "config": cfg, "devices": [], "active_session": None}
    with db() as c:
        a = active_session(c)
        if a:
            a["n_samples"] = c.execute("SELECT COUNT(*) n FROM telemetry WHERE session_id=?",
                                       (a["id"],)).fetchone()["n"]
            out["active_session"] = a
        for d in c.execute("SELECT DISTINCT device_id FROM telemetry WHERE ts>=?", (now - 86400,)):
            dev = d["device_id"]
            last = c.execute("SELECT * FROM telemetry WHERE device_id=? ORDER BY ts DESC LIMIT 1",
                             (dev,)).fetchone()
            recv = c.execute("SELECT MAX(received) m FROM telemetry WHERE device_id=?",
                             (dev,)).fetchone()["m"]
            w = c.execute("""SELECT COUNT(*) n, COUNT(hr) hr, COUNT(spo2) spo2, COUNT(obj) obj,
                             COUNT(ecg_bpm) ecg, MAX(CASE WHEN hr IS NOT NULL THEN ts END) hr_ts,
                             MAX(CASE WHEN spo2 IS NOT NULL THEN ts END) spo2_ts,
                             MAX(CASE WHEN obj IS NOT NULL THEN ts END) obj_ts,
                             MAX(CASE WHEN ecg_bpm IS NOT NULL THEN ts END) ecg_ts
                             FROM telemetry WHERE device_id=? AND ts>=?""",
                          (dev, now - 300)).fetchone()
            n = max(w["n"], 1)
            def q(valid, ts):
                return {"valid_pct": round(100.0 * valid / n, 1),
                        "last_valid_s_ago": round(now - ts, 1) if ts else None}
            out["devices"].append({
                "device_id": dev, "source": last["source"],
                "last_seen_s_ago": round(now - recv, 1),
                "connected": (now - recv) <= cfg["offline_seconds"],
                "samples_5min": w["n"],
                "sensors": {
                    "temperature (MLX90614)": {"hardware_ok": bool(last["mlx_ok"]), **q(w["obj"], w["obj_ts"])},
                    "pulse ox HR (MAX30102)": {"hardware_ok": bool(last["max_ok"]), **q(w["hr"], w["hr_ts"])},
                    "SpO2 (MAX30102)": {"hardware_ok": bool(last["max_ok"]), **q(w["spo2"], w["spo2_ts"])},
                    "ECG HR (AD8232)": {"hardware_ok": not bool(last["lead_off"]), **q(w["ecg"], w["ecg_ts"])},
                },
                "finger_present": bool(last["finger"]),
            })
    return out


def latest_payload(device=None):
    with db() as c:
        if device:
            r = c.execute("SELECT * FROM telemetry WHERE device_id=? ORDER BY ts DESC LIMIT 1", (device,)).fetchone()
        else:
            r = c.execute("SELECT * FROM telemetry ORDER BY ts DESC LIMIT 1").fetchone()
        if not r:
            return {"telemetry": None, "prediction": None}
        where, args = scope_sql(c)
        p = c.execute(f"SELECT * FROM predictions WHERE device_id=? AND {where} "
                      "ORDER BY ts DESC LIMIT 1", (r["device_id"], *args)).fetchone()
    tel = dict(r)
    pred = None
    if p:
        pred = dict(p)
        pred["inputs"] = json.loads(pred["inputs"]) if pred["inputs"] else None
    return {"telemetry": tel, "prediction": pred}


def history_payload(minutes, device=None):
    now = time.time()
    with db() as c:
        dev = device or (c.execute("SELECT device_id FROM telemetry ORDER BY ts DESC LIMIT 1").fetchone() or [None])[0]
        where, args = scope_sql(c)
        rows = c.execute(f"""SELECT ts,hr,spo2,obj,ecg_bpm FROM telemetry
                            WHERE device_id=? AND ts>=? AND {where} ORDER BY ts""",
                         (dev, now - minutes * 60, *args)).fetchall()
    step = max(1, len(rows) // 800)
    return {"device_id": dev, "points": [dict(r) for r in rows[::step]]}


def rows_payload(table, limit, scoped=False, order="ts DESC"):
    with db() as c:
        if scoped:                         # only the running test (or unassigned data)
            where, args = scope_sql(c)
            q = f"SELECT * FROM {table} WHERE {where} ORDER BY {order} LIMIT ?"
            rows = [dict(r) for r in c.execute(q, (*args, limit))]
        else:
            rows = [dict(r) for r in c.execute(f"SELECT * FROM {table} ORDER BY {order} LIMIT ?", (limit,))]
    if table == "predictions":
        for r in rows:
            r["inputs"] = json.loads(r["inputs"]) if r["inputs"] else None
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "SeptiSense/1.0"

    def log_message(self, format, *args):    # noqa: A002 - keep the server quiet
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._send(200, status_payload())
            if u.path == "/api/latest":
                return self._send(200, latest_payload(qs.get("device", [None])[0]))
            if u.path == "/api/history":
                return self._send(200, history_payload(float(qs.get("minutes", ["30"])[0]), qs.get("device", [None])[0]))
            if u.path == "/api/predictions":
                scoped = qs.get("scope", ["current"])[0] != "all"
                return self._send(200, rows_payload("predictions", int(qs.get("limit", ["200"])[0]), scoped))
            if u.path == "/api/alerts":
                scoped = qs.get("scope", ["all"])[0] == "current"
                return self._send(200, rows_payload("alerts", int(qs.get("limit", ["50"])[0]), scoped))
            if u.path == "/api/sessions":
                return self._send(200, sessions_payload(int(qs.get("limit", ["200"])[0])))
            if u.path == "/api/sessions/detail":
                d = session_detail(int(qs.get("id", ["-1"])[0]))
                return self._send(200, d) if d else self._send(404, {"error": "test not found"})
            if u.path == "/api/sessions/export":
                out = export_csv(int(qs.get("id", ["-1"])[0]), qs.get("kind", ["telemetry"])[0])
                if not out:
                    return self._send(404, {"error": "test or export type not found"})
                return self._send(200, out[1], "text/csv; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{out[0]}"'})
            if u.path == "/api/config":
                return self._send(200, get_config())
            if u.path == "/api/model":
                info = {"loaded": MODEL is not None, "error": MODEL_ERROR}
                if METRICS_PATH.exists():
                    info["metrics"] = json.loads(METRICS_PATH.read_text())
                return self._send(200, info)
            return self._send(404, {"error": "not found"})
        except Exception as e:                   # noqa
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._json_body()
            if u.path == "/api/telemetry":
                return self._send(200, {"stored": ingest(body)})
            if u.path == "/api/config":
                with db() as c:
                    for k, v in body.items():
                        if k in CONFIG_LIMITS:
                            lo, hi = CONFIG_LIMITS[k]
                            v = min(max(float(v), lo), hi)
                            c.execute("INSERT INTO config(key,value) VALUES(?,?) "
                                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, v))
                return self._send(200, get_config())
            if u.path == "/api/sessions/start":
                return self._send(200, start_session(body.get("label"), body.get("note")))
            if u.path == "/api/sessions/end":
                r = end_session()
                return self._send(200, r) if r else self._send(409, {"error": "no test is running"})
            if u.path == "/api/sessions/update":
                r = update_session(int(body.get("id", -1)), body.get("label"), body.get("note"))
                return self._send(200, r) if r else self._send(404, {"error": "test not found"})
            if u.path == "/api/sessions/delete":
                r = delete_session(int(body.get("id", -1)))
                if r == "ok":
                    return self._send(200, {"ok": True})
                if r == "active":
                    return self._send(409, {"error": "end the running test before deleting it"})
                return self._send(404, {"error": "test not found"})
            if u.path == "/api/alerts/ack":
                with db() as c:
                    c.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (int(body.get("id", -1)),))
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:                   # noqa
            traceback.print_exc()
            return self._send(400, {"error": str(e)})


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:                            # noqa
        return "unknown"


def main():
    global BIN_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bin-seconds", type=int, default=3600,
                    help="seconds per model time-bin (training used 3600 = 1 hour)")
    ap.add_argument("--demo-fast", action="store_true",
                    help="DEMO ONLY: 60-second bins so predictions appear within minutes")
    args = ap.parse_args()
    BIN_SECONDS = 60 if args.demo_fast else args.bin_seconds

    init_db()
    with db() as c:
        s = active_session(c)
    if s:
        print(f"NOTE: test #{s['id']} \"{s['label']}\" is still recording from an earlier run "
              "(end it from the dashboard).")
    load_model()
    threading.Thread(target=inference_loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 60)
    print(f" SeptiSense backend running  (model bin = {BIN_SECONDS} s)")
    if BIN_SECONDS < 3600:
        print(" !! DEMO TIME COMPRESSION ACTIVE - predictions are NOT clinically meaningful !!")
    print(f" Dashboard:        http://localhost:{args.port}")
    print(f" ESP32 should POST to: http://{lan_ip()}:{args.port}/api/telemetry")
    print("=" * 60)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
