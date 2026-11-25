"""Camera model, 2D-to-3D lifting, point clouds and RANSAC ground-plane fitting.
Pure NumPy so it is fast to unit-test."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    """Pinhole camera. Without calibration, focal length comes from an assumed
    horizontal field of view: fx = (W/2) / tan(hfov/2)."""
    width: int
    height: int
    hfov_deg: float = 70.0

    @property
    def fx(self) -> float:
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    def backproject(self, u, v, z):
        """Pixel (u, v) + depth z (m) -> camera-frame point (X right, Y down, Z forward)."""
        z = np.asarray(z, dtype=float)
        x = (np.asarray(u, dtype=float) - self.cx) * z / self.fx
        y = (np.asarray(v, dtype=float) - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)

    def project(self, pts):
        """Camera-frame points (N, 3) -> pixels (N, 2)."""
        pts = np.atleast_2d(pts)
        return np.stack([self.fx * pts[:, 0] / pts[:, 2] + self.cx,
                         self.fy * pts[:, 1] / pts[:, 2] + self.cy], axis=-1)


def box_depth(box, depth, inner=0.5):
    """Robust object distance: median depth over the central `inner` fraction of the
    box, which excludes background pixels that leak in at the box edges."""
    x1, y1, x2, y2 = box
    h, w = depth.shape
    m = (1 - inner) / 2
    ix1, ix2 = int(x1 + m * (x2 - x1)), int(np.ceil(x2 - m * (x2 - x1)))
    iy1, iy2 = int(y1 + m * (y2 - y1)), int(np.ceil(y2 - m * (y2 - y1)))
    ix1, iy1 = max(ix1, 0), max(iy1, 0)
    ix2, iy2 = min(max(ix2, ix1 + 1), w), min(max(iy2, iy1 + 1), h)
    patch = depth[iy1:iy2, ix1:ix2]
    return float(np.median(patch)) if patch.size else float(depth[min(int(y1), h - 1), min(int(x1), w - 1)])


def object_position(box, depth, cam: Camera):
    """3D centre of a detected object and its approximate metric width/height."""
    z = box_depth(box, depth)
    x1, y1, x2, y2 = box
    centre = cam.backproject((x1 + x2) / 2, (y1 + y2) / 2, z)
    size_w = (x2 - x1) * z / cam.fx
    size_h = (y2 - y1) * z / cam.fy
    return centre, float(size_w), float(size_h)


def ego_rows(img_h, top_frac=0.0, bottom_frac=0.0):
    """Image rows [r0, r1) that are NOT covered by the ego vehicle mask."""
    return int(round(img_h * top_frac)), int(round(img_h * (1 - bottom_frac)))


def masked_fraction(box, img_h, top_frac=0.0, bottom_frac=0.0):
    """Share of a box's height that falls inside the ego mask bands (0 = fully visible)."""
    _, y1, _, y2 = box
    if y2 <= y1:
        return 0.0
    r0, r1 = ego_rows(img_h, top_frac, bottom_frac)
    visible = max(0.0, min(y2, r1) - max(y1, r0))
    return float(1 - visible / (y2 - y1))


def depth_to_points(frame_bgr, depth, cam: Camera, stride=3, max_depth=80.0, rows=None):
    """Dense coloured point cloud from a depth map. Returns (N,3) points, (N,3) RGB in [0,1]
    and (N,) image row of each point (used to seed ground fitting). `rows` = (r0, r1)
    restricts the cloud to those image rows, e.g. to drop the ego vehicle."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth[vs, us]
    keep = (zs > 0.1) & (zs < max_depth)
    if rows is not None:
        keep &= (vs >= rows[0]) & (vs < rows[1])
    pts = cam.backproject(us[keep], vs[keep], zs[keep])
    cols = frame_bgr[vs[keep], us[keep]][:, ::-1].astype(np.float32) / 255.0
    return pts.astype(np.float32), cols, vs[keep]


def fit_plane_ransac(pts, iters=300, thresh=0.15, rng=None, max_tilt_deg=None):
    """RANSAC plane fit n.p + d = 0 (unit normal). Returns (normal, d, inlier_mask).
    With `max_tilt_deg`, only planes whose normal is within that angle of the camera's
    vertical (Y) axis are considered, so walls and vehicle backs cannot win."""
    rng = rng or np.random.default_rng(0)
    best_mask, best = None, None
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points")
    for _ in range(iters):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if max_tilt_deg is not None and abs(normal[1]) < np.cos(np.radians(max_tilt_deg)):
            continue
        d = -normal @ a
        mask = np.abs(pts @ normal + d) < thresh
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best = mask, (normal, d)
    if best_mask is None:
        raise ValueError("No plane candidate satisfied the tilt constraint")
    # least-squares refinement on inliers (SVD)
    inl = pts[best_mask]
    centroid = inl.mean(0)
    # full_matrices=False: otherwise U is N x N, which exhausts memory for large clouds
    normal = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]
    d = -normal @ centroid
    if normal[1] > 0:           # make the normal point "up" (camera Y axis points down)
        normal, d = -normal, -d
    mask = np.abs(pts @ normal + d) < thresh
    return normal, float(d), mask


def segment_ground(pts, rows, img_h, cfg_iters=300, cfg_thresh=0.15, seed_top_frac=0.55,
                   seed_bottom_row=None, max_tilt_deg=30.0):
    """Fit the road plane using points from the lower part of the image (above any ego
    mask), then label every point. Camera height above the road = |d| (a sanity check on
    depth scale: a car-mounted camera is typically 1.2-1.8 m high)."""
    bottom = img_h if seed_bottom_row is None else seed_bottom_row
    seed = pts[(rows > img_h * seed_top_frac) & (rows < bottom)]
    if len(seed) < 50:
        seed = pts
    normal, d, _ = fit_plane_ransac(seed, cfg_iters, cfg_thresh, max_tilt_deg=max_tilt_deg)
    ground = np.abs(pts @ normal + d) < cfg_thresh
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[1])))))
    return {"normal": normal.tolist(), "d": d, "camera_height_m": abs(d), "tilt_deg": tilt,
            "seed_points": int(len(seed)), "ground_ratio": float(ground.mean())}, ground


def save_ply(path, points, colors, labels=None):
    """Binary little-endian PLY (opens in MeshLab, CloudCompare, Open3D, Blender)."""
    n = len(points)
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if labels is not None:
        dtype.append(("ground", "u1"))
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points.T
    arr["red"], arr["green"], arr["blue"] = rgb.T
    if labels is not None:
        arr["ground"] = labels.astype(np.uint8)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue"]
    if labels is not None:
        header.append("property uchar ground")
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())
