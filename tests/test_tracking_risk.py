import numpy as np
import pandas as pd
import pytest

from u3p.config import Config
from u3p.risk import assess_risk, detect_anomalies, track_features
from u3p.tracking3d import KalmanCV2D, Tracker3D, jitter


def test_kalman_estimates_constant_approach_speed():
    rng = np.random.default_rng(0)
    kf = KalmanCV2D(0.0, 30.0, 0.0)
    for k in range(1, 60):
        t = k / 15
        z_true = 30.0 - 5.0 * t                       # approaching at 5 m/s
        st = kf.update(0.0, z_true * (1 + rng.normal(0, 0.05)), t)
    assert st["vz"] == pytest.approx(-5.0, abs=1.0)
    assert st["z"] == pytest.approx(30.0 - 5.0 * 59 / 15, abs=1.0)


def test_kalman_reduces_jitter():
    rng = np.random.default_rng(3)
    tr = Tracker3D()
    raw, filt = [], []
    for k in range(80):
        z = 20.0 * (1 + rng.normal(0, 0.06))
        st, _ = tr.update(1, 0.0, z, k / 15)
        raw.append(z)
        filt.append(st["z"])
    assert jitter(filt) < 0.4 * jitter(raw)


def test_far_measurements_are_trusted_less():
    near, far = KalmanCV2D(0, 5, 0), KalmanCV2D(0, 50, 0)
    near.update(0, 6, 0.1)    # +1 m surprise at 5 m
    far.update(0, 51, 0.1)    # +1 m surprise at 50 m
    assert (near.state["z"] - 5) > (far.state["z"] - 50)


@pytest.mark.parametrize("x,z,vz,expected", [
    (0.0, 20.0, -15.0, "CRITICAL"),   # TTC 1.33 s in our lane
    (0.0, 20.0, -8.0, "HIGH"),        # TTC 2.5 s
    (0.0, 4.0, 0.0, "CRITICAL"),      # very close, in lane
    (5.0, 4.0, 0.0, "MEDIUM"),        # very close but in the next lane
    (0.0, 40.0, 2.0, "LOW"),          # pulling away
])
def test_risk_levels(x, z, vz, expected):
    assert assess_risk(x, z, vz, Config().risk)[0] == expected


def test_isolation_forest_flags_the_odd_track():
    rng = np.random.default_rng(0)
    rows = []
    for tid in range(12):
        fast = tid == 7
        for f in range(10):
            z = (30 - f * (3.0 if fast else 0.2)) + rng.normal(0, 0.3)
            rows.append({"track_id": tid, "cls_name": "car", "frame": f, "time_s": f / 15, "z_m": z,
                         "closing_mps": 45.0 if fast else 3.0 + rng.normal(0, 0.5),
                         "lateral_mps": rng.normal(0, 0.2), "ttc_s": 0.7 if fast else 9.0 + rng.normal(),
                         "risk": "CRITICAL" if fast else "LOW"})
    feats = detect_anomalies(track_features(pd.DataFrame(rows)), contamination=0.09)
    flagged = feats.loc[feats.anomaly, "track_id"].tolist()
    assert flagged == [7]


def test_cockpit_config_loads():
    from pathlib import Path

    from u3p.config import Config
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs" / "cockpit.yaml")
    assert cfg.ego_mask.bottom_frac == pytest.approx(0.32)
    assert cfg.camera.hfov_deg == 70.0      # unlisted keys keep defaults
