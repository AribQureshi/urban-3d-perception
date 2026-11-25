"""Drawing helpers for the annotated video (the web dashboard draws its own overlays)."""
import cv2
import numpy as np

RISK_BGR = {"LOW": (110, 158, 47), "MEDIUM": (0, 164, 217), "HIGH": (27, 106, 224), "CRITICAL": (59, 40, 200)}


def colorize_depth(depth, max_depth=80.0):
    """Near = warm, far = cool (inverse depth gives more contrast up close)."""
    inv = 1.0 / np.clip(depth, 1.0, max_depth)
    norm = (inv - 1 / max_depth) / (1.0 - 1 / max_depth)
    return cv2.applyColorMap((np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def draw_objects(frame, objects):
    k = max(1.0, frame.shape[1] / 1280)          # scale strokes and text with resolution (4K -> 3x)
    fs, th_, lw = 0.5 * k, max(1, round(k)), max(2, round(2 * k))
    for o in objects:
        x1, y1, x2, y2 = o["box"].astype(int)
        col = RISK_BGR[o["risk"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, lw)
        ttc = f"  TTC {o['ttc']:.1f}s" if np.isfinite(o["ttc"]) else ""
        label = f"#{o['track_id']} {o['cls_name']} {o['z']:.1f}m{ttc}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
        pad = int(3 * k)
        ty = max(y1, th + 2 * pad)
        cv2.rectangle(frame, (x1, ty - th - 2 * pad), (x1 + tw + 2 * pad, ty), col, -1)
        cv2.putText(frame, label, (x1 + pad, ty - pad - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_,
                    cv2.LINE_AA)
    return frame


def draw_bev(objects, size, max_range=40.0):
    bev = np.full((size, size, 3), (40, 34, 28), np.uint8)
    s = size / max_range
    k = max(1.0, size / 720)                     # scale strokes and text with resolution
    fs, th_ = 0.4 * k, max(1, round(k))
    for d in range(10, int(max_range) + 1, 10):
        cv2.circle(bev, (size // 2, size), int(d * s), (85, 78, 70), th_, cv2.LINE_AA)
        cv2.putText(bev, f"{d} m", (size // 2 + int(4 * k), size - int(d * s) + int(14 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (150, 140, 130), th_, cv2.LINE_AA)
    e = int(7 * k)
    cv2.rectangle(bev, (size // 2 - e, size - 2 * e - 2), (size // 2 + e, size - 2), (255, 255, 255), -1)
    for o in objects:
        px, py = int(size / 2 + o["x"] * s), int(size - o["z"] * s)
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(bev, (px, py), int(6 * k), RISK_BGR[o["risk"]], -1, cv2.LINE_AA)
            cv2.putText(bev, str(o["track_id"]), (px + int(8 * k), py + int(4 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th_, cv2.LINE_AA)
    return bev


def shade_ego(frame, free_rows):
    """Darken the image bands covered by the ego-vehicle mask so it is visible in the output."""
    r0, r1 = free_rows
    for a, b in ((0, r0), (r1, frame.shape[0])):
        if b > a:
            frame[a:b] = (frame[a:b] * 0.45).astype(frame.dtype)
    if r1 < frame.shape[0]:
        cv2.line(frame, (0, r1), (frame.shape[1], r1), (255, 255, 255), max(2, frame.shape[1] // 640), cv2.LINE_AA)
    return frame
