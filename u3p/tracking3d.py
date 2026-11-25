"""3D state estimation for tracked objects.

Each track gets a constant-velocity Kalman filter over its ground-plane position
(X lateral, Z forward). Monocular depth error grows with distance, so the measurement
noise is depth-dependent: sigma = max(min, rel * Z). Far objects are trusted less,
which suppresses the jitter that raw per-frame depth produces.
"""
import numpy as np


class KalmanCV2D:
    """State s = [x, z, vx, vz]; measurement m = [x, z]."""

    def __init__(self, x, z, t, q=2.0, r_rel=0.05, r_min=0.3):
        self.s = np.array([x, z, 0.0, 0.0])
        self.P = np.diag([1.0, (r_rel * z) ** 2 + 1.0, 25.0, 25.0])
        self.t = t
        self.q, self.r_rel, self.r_min = q, r_rel, r_min
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.n_updates = 1

    def _predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # discrete white-noise acceleration model
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q ** 2
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q

    def update(self, x, z, t):
        dt = max(t - self.t, 1e-3)
        self._predict(dt)
        self.t = t
        sz = max(self.r_min, self.r_rel * z)
        sx = max(self.r_min * 0.5, self.r_rel * 0.5 * z)  # lateral error is smaller than depth error
        R = np.diag([sx ** 2, sz ** 2])
        y = np.array([x, z]) - self.H @ self.s
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.n_updates += 1
        return self.state

    @property
    def state(self):
        x, z, vx, vz = self.s
        return {"x": float(x), "z": float(z), "vx": float(vx), "vz": float(vz),
                "z_std": float(np.sqrt(self.P[1, 1]))}


class Tracker3D:
    """Keeps one Kalman filter per ByteTrack ID."""

    def __init__(self, q=2.0, r_rel=0.05, r_min=0.3):
        self.kw = dict(q=q, r_rel=r_rel, r_min=r_min)
        self.filters = {}

    def update(self, track_id, x, z, t):
        kf = self.filters.get(track_id)
        if kf is None:
            kf = self.filters[track_id] = KalmanCV2D(x, z, t, **self.kw)
            return kf.state, kf.n_updates
        return kf.update(x, z, t), kf.n_updates


def jitter(series):
    """Mean absolute second difference: how much a distance signal 'shakes'
    frame-to-frame. Lower is smoother. Used for the raw vs Kalman ablation."""
    s = np.asarray(series, dtype=float)
    if len(s) < 3:
        return np.nan
    return float(np.mean(np.abs(np.diff(s, n=2))))
