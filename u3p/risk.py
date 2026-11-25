"""Time-to-collision risk and unsupervised anomaly detection over tracks."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def assess_risk(x, z, vz, cfg):
    """TTC = distance / closing speed. Only objects inside our lane corridor
    (|X| < lane half-width) can be HIGH or CRITICAL."""
    closing = -vz
    ttc = z / closing if closing > 0.3 else np.inf
    in_path = abs(x) < cfg.lane_half_width_m
    if in_path and (ttc < cfg.ttc_critical_s or z < 5):
        level = "CRITICAL"
    elif in_path and (ttc < cfg.ttc_high_s or z < 10):
        level = "HIGH"
    elif ttc < cfg.ttc_medium_s or z < 15:
        level = "MEDIUM"
    else:
        level = "LOW"
    return level, float(ttc)


FEATURES = ["min_dist", "mean_dist", "max_closing", "mean_lateral", "min_ttc"]


def track_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("track_id")
    f = g.agg(
        cls_name=("cls_name", "first"),
        frames=("frame", "count"),
        first_t=("time_s", "min"),
        last_t=("time_s", "max"),
        min_dist=("z_m", "min"),
        mean_dist=("z_m", "mean"),
        max_closing=("closing_mps", "max"),
        mean_lateral=("lateral_mps", lambda s: float(np.abs(s).mean())),
        min_ttc=("ttc_s", "min"),
        peak_risk=("risk", lambda s: max(s, key=LEVELS.index)),
    ).reset_index()
    f["min_ttc"] = f["min_ttc"].replace(np.inf, 99.0).clip(upper=99.0)
    return f


def detect_anomalies(feats: pd.DataFrame, contamination=0.1, min_frames=5) -> pd.DataFrame:
    """Isolation Forest isolates tracks whose behaviour differs from the rest of the
    scene, e.g. a sudden cut-in or an unusually fast approach. Features are
    standardised so no single unit dominates."""
    feats = feats.copy()
    feats["anomaly"] = False
    feats["anomaly_score"] = 0.0
    valid = feats["frames"] >= min_frames
    if valid.sum() < 5:
        return feats
    X = feats.loc[valid, FEATURES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    iso = IsolationForest(contamination=contamination, random_state=42).fit(X)
    feats.loc[valid, "anomaly"] = iso.predict(X) == -1
    feats.loc[valid, "anomaly_score"] = -iso.score_samples(X)
    return feats
