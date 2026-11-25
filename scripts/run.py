"""Command-line entry point.

    python scripts/run.py --video samples/drive.mp4
    python scripts/run.py --video samples/drive.mp4 --config configs/default.yaml --max-frames 150 --stride 3
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from u3p.config import Config  # noqa: E402
from u3p.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Urban 3D Perception pipeline")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs", help="CSV / PLY / annotated video")
    ap.add_argument("--web", default="docs/data", help="data bundle for the web dashboard")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--hfov", type=float, help="camera horizontal field of view (deg)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load(a.config)
    if a.max_frames:
        cfg.run.max_frames = a.max_frames
    if a.stride:
        cfg.run.stride = a.stride
    if a.hfov:
        cfg.camera.hfov_deg = a.hfov
    summary = run(a.video, cfg, a.out, a.web)
    write_results_md(summary, Path(a.out) / "results.md")
    print(json.dumps(summary, indent=2))
    print(f"\nResults tables for REPORT.md / README.md written to {Path(a.out) / 'results.md'}")


def write_results_md(s, path):
    """Markdown tables ready to paste into REPORT.md and README.md."""
    a, g, lat = s["ablation"], s["ground_plane"], s["latency_ms"]
    names = {"detection_tracking": "Detection + tracking (YOLOv8n + ByteTrack)",
             "depth": "Metric depth (Depth Anything V2, ViT-S)",
             "lifting_kalman_risk": "3D lifting + Kalman + risk"}
    lines = [
        f"**Clip:** {s['video']}, {s['frames_processed']} frames ({s['duration_s']} s), "
        f"{s['unique_tracks']} tracked objects, end-to-end {s['pipeline_fps']} FPS.",
        "",
        "| Stage | Mean latency (ms) | Share |",
        "|---|---:|---:|",
        *[f"| {names.get(k, k)} | {v['mean_ms']:.1f} | {100 * v['share']:.0f}% |" for k, v in lat.items()],
        "",
    ]
    if a:
        lines += [
            "| Distance estimate | Jitter (m) | Speed noise (m/s) |",
            "|---|---:|---:|",
            f"| Raw per-frame depth | {a['distance_jitter_raw_m']:.3f} | {a['velocity_std_raw_mps']:.2f} |",
            f"| Kalman filtered (ours) | {a['distance_jitter_kalman_m']:.3f} | {a['velocity_std_kalman_mps']:.2f} |",
            f"| **Reduction** | **{a['jitter_reduction_pct']}%** | |",
            "",
            f"Evaluated on {a['tracks_evaluated']} tracks with at least 6 observations.",
            "",
        ]
    lines += [
        f"RANSAC ground plane: estimated camera height **{g['camera_height_m']:.2f} m**, "
        f"{100 * g['ground_ratio']:.0f}% of points on the road plane.",
        "",
        f"Anomalous tracks (Isolation Forest): {s['anomalous_tracks'] or 'none'}.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
