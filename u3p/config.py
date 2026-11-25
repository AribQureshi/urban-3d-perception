"""Typed configuration loaded from YAML, with CLI overrides."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class CameraCfg:
    hfov_deg: float = 70.0          # assumed horizontal field of view when intrinsics are unknown


@dataclass
class DetectorCfg:
    weights: str = "yolov8n.pt"
    conf: float = 0.35
    tracker: str = "bytetrack.yaml"


@dataclass
class DepthCfg:
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    max_depth_m: float = 80.0


@dataclass
class KalmanCfg:
    process_noise: float = 2.0      # acceleration noise std (m/s^2)
    meas_noise_rel: float = 0.05    # depth noise std as a fraction of distance (monocular error grows with range)
    meas_noise_min: float = 0.3     # floor on measurement noise (m)


@dataclass
class RiskCfg:
    lane_half_width_m: float = 1.8
    ttc_critical_s: float = 1.5
    ttc_high_s: float = 3.0
    ttc_medium_s: float = 5.0


@dataclass
class GroundCfg:
    ransac_iters: int = 300
    inlier_thresh_m: float = 0.15
    seed_top_frac: float = 0.55     # ground is fitted on image rows below this fraction of the height...
    max_tilt_deg: float = 30.0      # ...and only planes within this angle of horizontal are accepted


@dataclass
class EgoMaskCfg:
    """Image bands occupied by the ego vehicle (bonnet, dashboard, steering wheel, roof).
    Detections mostly inside them are dropped and their pixels are excluded from the
    point cloud and the ground fit. 0 disables a band."""
    bottom_frac: float = 0.0        # fraction of image height masked from the bottom
    top_frac: float = 0.0           # fraction of image height masked from the top
    max_overlap: float = 0.2        # drop a detection if more than this share of its box is masked


@dataclass
class AnomalyCfg:
    contamination: float = 0.1
    min_track_frames: int = 5


@dataclass
class RunCfg:
    max_frames: int = 300
    stride: int = 2                 # process every Nth frame
    cloud_stride: int = 3           # pixel stride when building the exported point cloud


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    kalman: KalmanCfg = field(default_factory=KalmanCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    ground: GroundCfg = field(default_factory=GroundCfg)
    ego_mask: EgoMaskCfg = field(default_factory=EgoMaskCfg)
    anomaly: AnomalyCfg = field(default_factory=AnomalyCfg)
    run: RunCfg = field(default_factory=RunCfg)

    @classmethod
    def load(cls, path=None):
        cfg = cls()
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text()) or {}
            for f in fields(cls):
                sub = getattr(cfg, f.name)
                for k, v in (data.get(f.name) or {}).items():
                    if not hasattr(sub, k):
                        raise KeyError(f"Unknown config key: {f.name}.{k}")
                    setattr(sub, k, type(getattr(sub, k))(v))
        return cfg

    def to_dict(self):
        return asdict(self)
