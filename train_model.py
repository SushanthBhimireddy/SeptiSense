#!/usr/bin/env python3
"""
SeptiSense - train and evaluate the sepsis early-warning model.

DATASET : PhysioNet/Computing in Cardiology Challenge 2019 (open access)
          https://physionet.org/content/challenge-2019/1.0.0/
          training_setA (20,336 ICU patients) + training_setB (20,000 patients)
          Sepsis-3 labels, hourly data.

TARGET  : "Will this patient reach sepsis onset within the NEXT 6 HOURS?"
          The dataset's SepsisLabel is already shifted: for septic patients
          it is 1 from (onset - 6 h). We keep only rows BEFORE onset, so a
          positive row means "onset is 1-6 hours away". Rows after onset
          (already septic) are removed so the model must truly forecast.

INPUTS  : only HR, O2Sat and Temp (what the SeptiSense hardware measures),
          plus rolling statistics/trends over the previous 6 hours.

SPLIT   : by PATIENT (never by row) to prevent data leakage:
            Hospital A: 70% train / 15% validation / 15% internal test
            Hospital B: 100% external test (a different hospital)

Usage:
    python train_model.py --data-dir path/to/folder_containing_training_setA_and_B
Quick smoke test on a small subset:
    python train_model.py --data-dir ... --sample-patients 500
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (FEATURE_NAMES, FILL_LIMIT, REQUIRED_NON_NULL, VITALS,
                      WINDOW, add_features, fill_vitals)

HORIZON_H = 6
SEED = 42


# ------------------------------------------------------------------ loading
def load_set(folder, hospital, sample_patients=None):
    files = sorted(Path(folder).glob("*.psv"))
    if not files:
        raise SystemExit(f"No .psv files found in {folder}")
    if sample_patients:
        files = files[:sample_patients]
    frames = []
    for i, f in enumerate(files, 1):
        d = pd.read_csv(f, sep="|", usecols=VITALS + ["SepsisLabel"])
        d["pid"] = f"{hospital}_{f.stem}"
        d["hour"] = np.arange(len(d))
        frames.append(d)
        if i % 5000 == 0:
            print(f"  {hospital}: read {i}/{len(files)} patients")
    df = pd.concat(frames, ignore_index=True)
    df["hospital"] = hospital
    print(f"  {hospital}: {len(files)} patients, {len(df):,} hourly rows")
    return df


def prepare(df):
    """Apply the forecasting target definition and build features."""
    # first hour with SepsisLabel == 1 (this is onset - 6 h)
    first_pos = df[df["SepsisLabel"] == 1].groupby("pid")["hour"].min()
    df = df.copy()
    df["first_pos"] = df["pid"].map(first_pos)
    # drop every row at/after sepsis onset (onset = first_pos + 6 h)
    keep = df["first_pos"].isna() | (df["hour"] < df["first_pos"] + HORIZON_H)
    df = df[keep].reset_index(drop=True)
    df["y"] = df["SepsisLabel"].astype(int)

    df = fill_vitals(df)
    X = add_features(df)
    ok = X[REQUIRED_NON_NULL].notna().all(axis=1)
    meta = df.loc[ok, ["pid", "hour", "y", "hospital"]].reset_index(drop=True)
    X = X.loc[ok].astype("float32").reset_index(drop=True)
    return meta, X


# ------------------------------------------------------------------ metrics
def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def patient_slices(pids):
    codes, _ = pd.factorize(pids)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(counts))]


def bootstrap_ci(y, p, pids, n_boot, rng):
    if n_boot <= 0:
        return {}
    slices = patient_slices(pids)
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(slices), len(slices))
        idx = np.concatenate([slices[k] for k in pick])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        aurocs.append(roc_auc_score(yy, pp))
        auprcs.append(average_precision_score(yy, pp))
    if not aurocs:
        return {}
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"auroc_95ci": q(aurocs), "auprc_95ci": q(auprcs)}


def evaluate(name, meta, p, thr, n_boot, rng, extra_points=None):
    y = meta["y"].values
    res = {"set": name, "rows": int(len(y)), "patients": int(meta["pid"].nunique()),
           "positive_rows": int(y.sum()), "prevalence": float(y.mean())}
    if y.min() == y.max():
        res["note"] = "only one class present - metrics not computable"
        return res
    res["auroc"] = float(roc_auc_score(y, p))
    res["auprc"] = float(average_precision_score(y, p))
    res["auprc_baseline_(prevalence)"] = float(y.mean())
    res["brier"] = float(brier_score_loss(y, p))
    res["ece_10bin"] = ece_score(y, p)
    res.update(bootstrap_ci(y, p, meta["pid"].values, n_boot, rng))

    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    res["threshold"] = float(thr)
    res["row_sensitivity"] = tp / max(tp + fn, 1)
    res["row_specificity"] = tn / max(tn + fp, 1)
    res["row_precision"] = tp / max(tp + fp, 1)
    res["row_npv"] = tn / max(tn + fn, 1)

    # patient-level view
    df = meta.assign(alarm=pred)
    g = df.groupby("pid")
    ever_septic = g["y"].max() == 1
    any_alarm = g["alarm"].max()
    res["septic_patients"] = int(ever_septic.sum())
    res["septic_patients_with_alarm_before_onset"] = float(
        any_alarm[ever_septic].mean()) if ever_septic.any() else None
    neg = df[df["pid"].isin(ever_septic[~ever_septic].index)]
    hours = max(len(neg), 1)
    res["false_alarm_rows_per_24h_nonseptic"] = float(neg["alarm"].sum() / hours * 24)
    res["nonseptic_patients_with_any_alarm"] = float(
        any_alarm[~ever_septic].mean()) if (~ever_septic).any() else None
    if extra_points:
        res["operating_points"] = {k: op_point(meta, p, t) for k, t in extra_points.items()}
    return res


def pick_threshold(y_val, p_val, target_sens):
    fpr, tpr, thr = roc_curve(y_val, p_val)
    idx = int(np.argmax(tpr >= target_sens))
    return float(thr[idx])


def pick_threshold_specificity(y_val, p_val, target_spec):
    """Highest-sensitivity threshold whose row-level false-positive rate is
    at most (1 - target_spec) on the validation rows. This is an alarm
    BUDGET: it caps how often non-sepsis hours raise an alarm."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(fpr <= (1.0 - target_spec))[0]
    idx = int(ok[-1]) if len(ok) else 1
    idx = max(idx, 1)              # index 0 is the "no alarms" (inf) threshold
    return float(thr[idx])


def op_point(meta, p, thr):
    """Metrics at one alert threshold, including patient-level false alarms."""
    y = meta["y"].values
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    df = meta.assign(alarm=pred)
    g = df.groupby("pid")
    ever_septic = g["y"].max() == 1
    any_alarm = g["alarm"].max()
    neg = df[df["pid"].isin(ever_septic[~ever_septic].index)]
    return {
        "threshold": float(thr),
        "row_sensitivity": tp / max(tp + fn, 1),
        "row_specificity": tn / max(tn + fp, 1),
        "row_precision": tp / max(tp + fp, 1),
        "septic_patients_with_alarm_before_onset":
            float(any_alarm[ever_septic].mean()) if ever_septic.any() else None,
        "nonseptic_patients_with_any_alarm":
            float(any_alarm[~ever_septic].mean()) if (~ever_septic).any() else None,
        "false_alarm_rows_per_24h_nonseptic":
            float(neg["alarm"].sum() / max(len(neg), 1) * 24),
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="folder that contains training_setA and training_setB")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--sample-patients", type=int, default=None,
                    help="use only the first N patients of each set (quick test)")
    ap.add_argument("--threshold-mode", choices=["specificity", "sensitivity"],
                    default="specificity",
                    help="how the alert threshold is chosen on validation data: "
                         "'specificity' = false-alarm budget (default), "
                         "'sensitivity' = target sensitivity (old behaviour)")
    ap.add_argument("--target-specificity", type=float, default=0.90,
                    help="row-level specificity for --threshold-mode specificity "
                         "(0.90 = about 10%% of non-sepsis hours may alarm)")
    ap.add_argument("--target-sensitivity", type=float, default=0.70,
                    help="row-level sensitivity for --threshold-mode sensitivity")
    ap.add_argument("--bootstrap", type=int, default=50,
                    help="patient-level bootstrap resamples for 95%% CIs (0=off)")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    dfA = load_set(data_dir / "training_setA", "A", args.sample_patients)
    dfB = load_set(data_dir / "training_setB", "B", args.sample_patients)

    print("Building target + features ...")
    metaA, XA = prepare(dfA)
    metaB, XB = prepare(dfB)
    print(f"  usable rows  A: {len(metaA):,}   B: {len(metaB):,}")

    # ---- patient-level split of hospital A
    pat = metaA.groupby("pid")["y"].max()
    train_p, tmp_p = train_test_split(pat.index.values, test_size=0.30,
                                      stratify=pat.values, random_state=SEED)
    val_p, test_p = train_test_split(tmp_p, test_size=0.50,
                                     stratify=pat.loc[tmp_p].values, random_state=SEED)
    mtr = metaA["pid"].isin(train_p).values
    mva = metaA["pid"].isin(val_p).values
    mte = metaA["pid"].isin(test_p).values
    assert not (set(train_p) & set(val_p)) and not (set(train_p) & set(test_p)) \
        and not (set(val_p) & set(test_p)), "patient leakage between splits!"

    Xtr, ytr = XA[mtr], metaA.loc[mtr, "y"].values
    Xva, yva = XA[mva], metaA.loc[mva, "y"].values
    Xte, metaTe = XA[mte], metaA[mte].reset_index(drop=True)
    print(f"  train {len(ytr):,} rows ({len(train_p)} pts, prevalence {ytr.mean():.3%})")
    print(f"  val   {len(yva):,} rows ({len(val_p)} pts, prevalence {yva.mean():.3%})")
    print(f"  testA {len(metaTe):,} rows ({len(test_p)} pts)   testB {len(metaB):,} rows (external)")

    # ---- choose hyper-parameters + boosting iterations on validation patients
    print("Training gradient boosting model ...")
    configs = [
        dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0),
        dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=5.0),
        dict(learning_rate=0.03, max_leaf_nodes=7,  min_samples_leaf=1000, l2_regularization=10.0),
    ]
    best = (-1, None, None)            # (val AUPRC, config, iterations)
    for ci, cfg in enumerate(configs):
        probe = HistGradientBoostingClassifier(
            max_iter=600, early_stopping=False, random_state=SEED, **cfg).fit(Xtr, ytr)
        c_iter, c_ap = 5, -1
        for i, p in enumerate(probe.staged_predict_proba(Xva), 1):
            if i % 5 == 0:
                a = average_precision_score(yva, p[:, 1])
                if a > c_ap:
                    c_ap, c_iter = a, i
        print(f"  config {ci}: best iterations {c_iter}, validation AUPRC {c_ap:.4f}")
        if c_ap > best[0]:
            best = (c_ap, cfg, c_iter)
    best_ap, best_cfg, best_iter = best
    print(f"  selected config {best_cfg} with {best_iter} iterations "
          f"(validation AUPRC {best_ap:.4f})")
    clf = HistGradientBoostingClassifier(max_iter=best_iter, early_stopping=False,
                                         random_state=SEED, **best_cfg).fit(Xtr, ytr)

    # ---- probability calibration + alert threshold (validation patients only)
    pva_raw = clf.predict_proba(Xva)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pva_raw, yva)
    pva = iso.predict(pva_raw)
    if args.threshold_mode == "specificity":
        thr = pick_threshold_specificity(yva, pva, args.target_specificity)
        print(f"  alert threshold (calibrated risk) for >= {args.target_specificity:.0%} "
              f"row specificity on validation: {thr:.4f}")
    else:
        thr = pick_threshold(yva, pva, args.target_sensitivity)
        print(f"  alert threshold (calibrated risk) for ~{args.target_sensitivity:.0%} "
              f"row sensitivity on validation: {thr:.4f}")
    n_levels = int(len(np.unique(pva)))
    v_pred = pva >= thr
    v_sens = float((v_pred & (yva == 1)).sum() / max((yva == 1).sum(), 1))
    v_spec = float((~v_pred & (yva == 0)).sum() / max((yva == 0).sum(), 1))
    print(f"  validation at this threshold: sensitivity {v_sens:.2f}, specificity {v_spec:.2f} "
          f"({n_levels} distinct calibrated risk levels)")
    if args.threshold_mode == "specificity" and v_spec < args.target_specificity - 0.05:
        print("  WARNING: target specificity not reachable - isotonic calibration has too few "
              "distinct risk levels, so thresholds jump in steps. Consider a smoother calibrator.")
    # alternative operating points, all chosen on validation patients only
    extra = {f"specificity_{int(sp*100)}": pick_threshold_specificity(yva, pva, sp)
             for sp in (0.80, 0.90, 0.95)}
    extra.update({f"sensitivity_{int(se*100)}": pick_threshold(yva, pva, se)
                  for se in (0.50, 0.70)})

    # ---- baseline for comparison
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=300))
    base.fit(Xtr, ytr)

    # ---- held-out evaluation
    print("Evaluating on held-out data ...")
    pte = iso.predict(clf.predict_proba(Xte)[:, 1])
    pB = iso.predict(clf.predict_proba(XB)[:, 1])
    results = [evaluate("internal_test_hospital_A_unseen_patients", metaTe, pte, thr,
                        args.bootstrap, rng, extra),
               evaluate("external_test_hospital_B", metaB, pB, thr,
                        args.bootstrap, rng, extra)]
    baseline = {}
    for name, X, m in [("internal_test_A", Xte, metaTe), ("external_test_B", XB, metaB)]:
        yb = m["y"].values
        pb = base.predict_proba(X)[:, 1]
        baseline[name] = {"auroc": float(roc_auc_score(yb, pb)),
                          "auprc": float(average_precision_score(yb, pb))}

    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for (m, p, lab) in [(metaTe, pte, "Internal test (A)"), (metaB, pB, "External test (B)")]:
        y = m["y"].values
        if y.min() == y.max():
            continue
        fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, label=lab)
        pr, rc, _ = precision_recall_curve(y, p); ax[1].plot(rc, pr, label=lab)
        edges = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
        ax[2].plot(xs, ys, "o-", label=lab)
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC", xlabel="False positive rate", ylabel="Sensitivity")
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set(title="Calibration", xlabel="Predicted risk", ylabel="Observed fraction")
    for a in ax: a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out_dir / "evaluation.png", dpi=130); plt.close()

    # ---- save
    split_counts = {
        "train":      {"patients": int(len(train_p)), "rows": int(len(ytr)), "positive_rows": int(ytr.sum())},
        "validation": {"patients": int(len(val_p)),   "rows": int(len(yva)), "positive_rows": int(yva.sum())},
        "internal_test_A": {"patients": int(len(test_p)), "rows": int(len(metaTe))},
        "external_test_B": {"patients": int(metaB["pid"].nunique()), "rows": int(len(metaB))},
    }
    meta = {
        "dataset": "PhysioNet/Computing in Cardiology Challenge 2019 (training sets A and B)",
        "dataset_url": "https://physionet.org/content/challenge-2019/1.0.0/",
        "dataset_doi": "10.13026/v64v-d857",
        "label_definition": "Sepsis-3; target = sepsis onset within the next 6 hours "
                            "(rows at/after onset removed)",
        "horizon_hours": HORIZON_H,
        "input_signals": VITALS,
        "features": FEATURE_NAMES,
        "window_bins": WINDOW, "fill_limit_bins": FILL_LIMIT,
        "split": "patient-level: hospital A 70/15/15 train/val/test; hospital B external test",
        "threshold_mode": args.threshold_mode,
        "target_specificity_for_threshold": args.target_specificity,
        "target_sensitivity_for_threshold": args.target_sensitivity,
        "alert_threshold": thr,
        "alternative_thresholds_(chosen_on_validation)": extra,
        "model_hyperparameters": best_cfg,
        "split_counts": split_counts,
        "boosting_iterations": best_iter,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "sample_patients_limit": args.sample_patients,
        "disclaimer": "Educational/research prototype. Not clinically validated.",
    }
    joblib.dump({"model": clf, "calibrator": iso, "features": FEATURE_NAMES,
                 "threshold": thr, "meta": meta}, out_dir / "sepsis_model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"meta": meta, "results": results, "baseline_logistic_regression": baseline},
                  f, indent=2)

    print("\n================ RESULTS (held-out) ================")
    for r in results:
        print(json.dumps(r, indent=2))
    print("Baseline logistic regression:", json.dumps(baseline))
    print(f"\nSaved: {out_dir/'sepsis_model.joblib'}, metrics.json, evaluation.png")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
