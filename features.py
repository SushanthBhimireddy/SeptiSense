"""
Shared feature engineering for SeptiSense.

IMPORTANT: training (train_model.py) and live inference (server.py) BOTH use
this file, so the model always sees features computed the same way.

The model only uses the three signals the SeptiSense hardware can measure:
  HR    heart rate (beats/min)
  O2Sat oxygen saturation (%)
  Temp  temperature (deg C)

Data is organised as ONE ROW PER TIME BIN (1 hour in the PhysioNet data).
"""
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp"]
WINDOW = 6          # look-back window (bins) for rolling statistics
MIN_PERIODS = 3     # need at least this many observed bins in the window
FILL_LIMIT = 8      # forward-fill a missing value for at most 8 bins


def _make_names():
    names = []
    for v in VITALS:
        names += [f"{v}_cur", f"{v}_mean{WINDOW}", f"{v}_std{WINDOW}",
                  f"{v}_min{WINDOW}", f"{v}_max{WINDOW}", f"{v}_d1", f"{v}_d3"]
    return names


FEATURE_NAMES = _make_names()
REQUIRED_NON_NULL = ([f"{v}_cur" for v in VITALS] +
                     [f"{v}_mean{WINDOW}" for v in VITALS])


def fill_vitals(df, group_col="pid"):
    """Forward-fill short gaps inside each patient (never across patients)."""
    df = df.copy()
    df[VITALS] = df.groupby(group_col, sort=False)[VITALS].ffill(limit=FILL_LIMIT)
    return df


def add_features(df, group_col="pid"):
    """
    df must be sorted by (patient, time), have a unique RangeIndex, the
    columns in VITALS and a patient-id column. Returns the feature table.
    """
    out = pd.DataFrame(index=df.index)
    g = df.groupby(group_col, sort=False)
    for v in VITALS:
        s = df[v]
        gv = g[v]
        roll = gv.rolling(WINDOW, min_periods=MIN_PERIODS)
        out[f"{v}_cur"] = s
        out[f"{v}_mean{WINDOW}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{v}_std{WINDOW}"] = roll.std().reset_index(level=0, drop=True)
        out[f"{v}_min{WINDOW}"] = roll.min().reset_index(level=0, drop=True)
        out[f"{v}_max{WINDOW}"] = roll.max().reset_index(level=0, drop=True)
        out[f"{v}_d1"] = s - gv.shift(1)
        out[f"{v}_d3"] = s - gv.shift(3)
    return out[FEATURE_NAMES]
