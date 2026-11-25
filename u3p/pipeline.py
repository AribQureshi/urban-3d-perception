"""End-to-end pipeline: video -> detections/tracks -> metric depth -> 3D (Kalman) ->
risk + anomalies -> evaluation numbers -> outputs + web dashboard bundle."""
import json
import logging
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .geometry import Camera, depth_to_points, ego_rows, masked_fraction, object_position, save_ply, segment_ground
from .risk import LEVELS, assess_risk, detect_anomalies, track_features
from .tracking3d import Tracker3D, jitter
from .viz import colorize_depth, draw_bev, draw_objects, shade_ego

log = logging.getLogger("u3p")
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
WEB_MAX_W = 960


class StageTimer:
    def __init__(self):
        self.times = defaultdict(list)

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                timer.times[name].append((time.perf_counter() - self.t) * 1000)
        return _T()

    def summary(self):
        means = {k: float(np.mean(v)) for k, v in self.times.items()}
        total = sum(means.values())
        return {k: {"mean_ms": round(v, 2), "share": round(v / total, 3)} for k, v in means.items()}


def _to_h264(src, dst):
    """Browsers only play H.264 mp4; OpenCV writes mp4v. Re-encode when ffmpeg exists."""
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(dst)], check=True)
    else:
        log.warning("ffmpeg not found: %s may not play in browsers", dst)
        shutil.copy(src, dst)


def run(video, cfg: Config, out_dir="outputs", web_dir="docs/data", detector=None, depth_model=None,
        progress=None):
    out, web = Path(out_dir), Path(web_dir)
    out.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride, max_frames = cfg.run.stride, cfg.run.max_frames
    out_fps = src_fps / stride
    cam = Camera(W, H, cfg.camera.hfov_deg)

    if detector is None:
        from .models import DetectorTracker
        detector = DetectorTracker(cfg.detector.weights, cfg.detector.conf, cfg.detector.tracker)
    if depth_model is None:
        from .models import DepthEstimator
        depth_model = DepthEstimator(cfg.depth.model_id, max_depth=cfg.depth.max_depth_m)

    em = cfg.ego_mask
    free_rows = ego_rows(H, em.top_frac, em.bottom_frac)     # rows not covered by the ego vehicle
    n_masked = 0

    tracker3d = Tracker3D(cfg.kalman.process_noise, cfg.kalman.meas_noise_rel, cfg.kalman.meas_noise_min)
    timer = StageTimer()

    ws = min(1.0, WEB_MAX_W / W)
    web_size = (int(W * ws) // 2 * 2, int(H * ws) // 2 * 2)
    tmp = Path(tempfile.mkdtemp())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_rgb = cv2.VideoWriter(str(tmp / "rgb.mp4"), fourcc, out_fps, web_size)
    w_dep = cv2.VideoWriter(str(tmp / "depth.mp4"), fourcc, out_fps, web_size)
    w_ann = cv2.VideoWriter(str(tmp / "annotated.mp4"), fourcc, out_fps, (W + H, H))

    rows, frames_web = [], []
    key = {"score": -1.0}
    idx = processed = 0
    t_start = time.perf_counter()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames * stride

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        t = idx / src_fps

        with timer("detection_tracking"):
            dets = detector(frame)
            if em.top_frac or em.bottom_frac:
                kept = [d for d in dets if masked_fraction(d["box"], H, em.top_frac, em.bottom_frac) <= em.max_overlap]
                n_masked += len(dets) - len(kept)
                dets = kept
        with timer("depth"):
            depth = depth_model(frame)
        with timer("lifting_kalman_risk"):
            objects = []
            for d in dets:
                centre, size_w, size_h = object_position(d["box"], depth, cam)
                xr, _, zr = centre
                st, n_upd = tracker3d.update(d["track_id"], xr, zr, t)
                ready = n_upd >= 3          # wait a few frames before trusting velocity
                vz, vx = (st["vz"], st["vx"]) if ready else (0.0, 0.0)
                risk, ttc = assess_risk(st["x"], st["z"], vz, cfg.risk)
                o = {**d, "x": st["x"], "z": st["z"], "z_raw": float(zr), "x_raw": float(xr),
                     "vx": vx, "vz": vz, "z_std": st["z_std"], "risk": risk, "ttc": ttc,
                     "size_w": size_w, "size_h": size_h, "y": float(centre[1])}
                objects.append(o)
                rows.append({"frame": processed, "src_frame": idx, "time_s": round(t, 3),
                             "track_id": d["track_id"], "cls_name": d["cls_name"], "conf": round(d["conf"], 3),
                             "x_m": st["x"], "z_m": st["z"], "z_raw_m": float(zr), "x_raw_m": float(xr),
                             "z_std_m": st["z_std"], "closing_mps": -vz, "lateral_mps": vx,
                             "ttc_s": ttc, "risk": risk})

        score = max((LEVELS.index(o["risk"]) for o in objects), default=-1) + 0.01 * len(objects)
        if score > key["score"]:
            key = {"score": score, "frame": frame.copy(), "depth": depth, "idx": processed, "objects": objects}

        frames_web.append([[o["track_id"], CLASSES.index(o["cls_name"]) if o["cls_name"] in CLASSES else 0,
                            *[round(float(v), 1) for v in o["box"]],
                            round(o["x"], 2), round(o["z"], 2), round(o["z_raw"], 2),
                            LEVELS.index(o["risk"]), round(o["ttc"], 2) if np.isfinite(o["ttc"]) else None,
                            round(o["vz"], 2)] for o in objects])

        w_rgb.write(cv2.resize(frame, web_size))
        w_dep.write(cv2.resize(colorize_depth(depth, cfg.depth.max_depth_m), web_size))
        w_ann.write(np.hstack([shade_ego(draw_objects(frame.copy(), objects), free_rows), draw_bev(objects, H)]))
        processed += 1
        idx += 1
        if progress:
            progress(min(idx / total, 1.0))
    cap.release()
    for w in (w_rgb, w_dep, w_ann):
        w.release()
    wall = time.perf_counter() - t_start

    if not rows:
        raise RuntimeError("No road users detected. Try another video or lower detector.conf.")
    df = pd.DataFrame(rows)
    df.to_csv(out / "frame_log.csv", index=False)
    feats = detect_anomalies(track_features(df), cfg.anomaly.contamination, cfg.anomaly.min_track_frames)
    feats.to_csv(out / "track_summary.csv", index=False)

    # ---- Evaluation: raw per-frame depth vs Kalman-filtered (no ground truth needed) ----
    abl = []
    for tid, g in df.groupby("track_id"):
        if len(g) >= 6:
            raw_v = np.diff(g.z_raw_m) * out_fps
            abl.append({"track_id": int(tid), "jitter_raw": jitter(g.z_raw_m), "jitter_kf": jitter(g.z_m),
                        "vel_std_raw": float(np.std(raw_v)), "vel_std_kf": float(np.std(-g.closing_mps.iloc[3:]))
                        if len(g) > 4 else np.nan})
    abl = pd.DataFrame(abl)
    ablation = {}
    if len(abl):
        jr, jk = abl.jitter_raw.mean(), abl.jitter_kf.mean()
        vr, vk = abl.vel_std_raw.mean(), abl.vel_std_kf.mean()
        ablation = {"tracks_evaluated": int(len(abl)),
                    "distance_jitter_raw_m": round(float(jr), 3), "distance_jitter_kalman_m": round(float(jk), 3),
                    "jitter_reduction_pct": round(float(100 * (1 - jk / jr)), 1) if jr > 0 else None,
                    "velocity_std_raw_mps": round(float(vr), 2), "velocity_std_kalman_mps": round(float(vk), 2)}
        abl.to_csv(out / "ablation_per_track.csv", index=False)

    # ---- 3D scene of the most critical frame + RANSAC ground plane ----
    pts, cols, prow = depth_to_points(key["frame"], key["depth"], cam, cfg.run.cloud_stride, cfg.depth.max_depth_m,
                                      rows=free_rows)
    ground_info, ground = segment_ground(pts, prow, H, cfg.ground.ransac_iters, cfg.ground.inlier_thresh_m,
                                         cfg.ground.seed_top_frac, free_rows[1], cfg.ground.max_tilt_deg)
    save_ply(out / "scene.ply", pts, cols, ground)
    if len(pts) > 80000:
        sel = np.random.default_rng(0).choice(len(pts), 80000, replace=False)
        pts, cols, ground = pts[sel], cols[sel], ground[sel]
    with open(web / "cloud.bin", "wb") as f:
        f.write(pts.astype("<f4").tobytes())
        f.write((np.clip(cols, 0, 1) * 255).astype(np.uint8).tobytes())
        f.write(ground.astype(np.uint8).tobytes())

    for name in ("rgb", "depth"):
        _to_h264(tmp / f"{name}.mp4", web / f"{name}.mp4")
    _to_h264(tmp / "annotated.mp4", out / "annotated.mp4")
    shutil.rmtree(tmp, ignore_errors=True)

    latency = timer.summary()
    summary = {
        "video": Path(video).name,
        "frames_processed": processed,
        "duration_s": round(processed / out_fps, 2),
        "pipeline_fps": round(processed / wall, 2),
        "unique_tracks": int(df.track_id.nunique()),
        "detections_dropped_by_ego_mask": n_masked,
        "objects_by_class": {k: int(v) for k, v in df.groupby("cls_name").track_id.nunique().items()},
        "risk_observations": {k: int(v) for k, v in df.risk.value_counts().items()},
        "anomalous_tracks": [int(t) for t in feats.loc[feats.anomaly, "track_id"]],
        "closest_object_m": round(float(df.z_m.min()), 2),
        "latency_ms": latency,
        "ablation": ablation,
        "ground_plane": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ground_info.items()},
        "point_cloud_points": int(len(pts)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    meta = {"summary": summary, "config": cfg.to_dict(), "classes": CLASSES, "levels": LEVELS,
            "video": {"width": W, "height": H, "fps": out_fps, "frames": processed},
            "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy},
            "keyframe": {"index": key["idx"], "objects": [
                {"id": o["track_id"], "cls": o["cls_name"], "x": o["x_raw"], "y": o["y"], "z": o["z_raw"],
                 "w": o["size_w"], "h": o["size_h"], "risk": LEVELS.index(o["risk"])} for o in key["objects"]]},
            "cloud": {"n": int(len(pts))}}
    tracks = feats.replace([np.inf, -np.inf], None).to_dict(orient="records")
    (web / "meta.json").write_text(json.dumps(meta, default=float))
    (web / "frames.json").write_text(json.dumps(frames_web, separators=(",", ":")))
    (web / "tracks.json").write_text(json.dumps(tracks, default=float))
    log.info("Done: %d frames, %d tracks, %.2f FPS", processed, summary["unique_tracks"], summary["pipeline_fps"])
    return summary
