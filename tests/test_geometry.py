import numpy as np
import pytest

from u3p.geometry import Camera, box_depth, fit_plane_ransac, object_position, segment_ground


def test_backproject_project_roundtrip():
    cam = Camera(1280, 720, 70)
    pts = np.array([[1.0, 0.5, 10.0], [-3.0, 1.2, 25.0]])
    uv = cam.project(pts)
    back = cam.backproject(uv[:, 0], uv[:, 1], pts[:, 2])
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_principal_point_is_on_optical_axis():
    cam = Camera(640, 480, 90)
    np.testing.assert_allclose(cam.backproject(320, 240, 7.0), [0, 0, 7.0])
    assert cam.fx == pytest.approx(320.0)  # tan(45 deg) = 1


def test_box_depth_ignores_background_at_edges():
    depth = np.full((100, 100), 50.0)
    depth[20:80, 20:80] = 10.0          # object
    depth[20:80, 20:30] = 50.0          # background leaking into the left edge of the box
    assert box_depth((20, 20, 80, 80), depth) == pytest.approx(10.0)


def test_object_size_scales_with_distance():
    cam = Camera(1000, 1000, 90)        # fx = 500
    depth = np.full((1000, 1000), 20.0)
    _, w, h = object_position((450, 450, 550, 550), depth, cam)
    assert w == pytest.approx(100 * 20 / 500)
    assert h == pytest.approx(4.0)


def test_ransac_recovers_ground_plane_with_outliers():
    rng = np.random.default_rng(1)
    n = 2000
    ground = np.c_[rng.uniform(-10, 10, n), np.full(n, 1.5) + rng.normal(0, 0.02, n), rng.uniform(2, 40, n)]
    clutter = rng.uniform([-10, -5, 2], [10, 1.2, 40], (600, 3))
    pts = np.vstack([ground, clutter])
    normal, d, mask = fit_plane_ransac(pts, iters=200, thresh=0.1)
    np.testing.assert_allclose(np.abs(normal), [0, 1, 0], atol=0.02)
    assert abs(d) == pytest.approx(1.5, abs=0.03)
    assert mask[:n].mean() > 0.95 and mask[n:].mean() < 0.1


def test_segment_ground_reports_camera_height():
    cam = Camera(320, 240, 70)
    v = np.arange(130, 240)
    u = np.arange(0, 320, 4)
    uu, vv = np.meshgrid(u, v)
    z = 1.4 * cam.fy / (vv - cam.cy)
    pts = cam.backproject(uu.ravel(), vv.ravel(), z.ravel())
    info, labels = segment_ground(pts, vv.ravel(), 240)
    assert info["camera_height_m"] == pytest.approx(1.4, abs=0.02)
    assert labels.mean() > 0.95


def test_ransac_handles_full_resolution_cloud():
    # A 4K frame at stride 3 gives ~900k points; the SVD refinement must not build an N x N matrix.
    rng = np.random.default_rng(2)
    n = 300_000
    pts = np.c_[rng.uniform(-10, 10, n), np.full(n, 1.5), rng.uniform(2, 40, n)].astype(np.float32)
    normal, d, mask = fit_plane_ransac(pts, iters=20, thresh=0.1)
    assert abs(d) == pytest.approx(1.5, abs=0.01)
    assert mask.mean() > 0.99


def test_masked_fraction_of_boxes():
    from u3p.geometry import masked_fraction
    # image 1000 px tall, bottom 30% masked (rows >= 700)
    assert masked_fraction((0, 100, 50, 600), 1000, bottom_frac=0.3) == 0.0           # fully visible
    assert masked_fraction((0, 750, 50, 950), 1000, bottom_frac=0.3) == 1.0           # hands on the wheel
    assert masked_fraction((0, 100, 50, 1000), 1000, bottom_frac=0.3) == pytest.approx(300 / 900)  # driver
    assert masked_fraction((0, 0, 50, 200), 1000, top_frac=0.1) == pytest.approx(0.5)


def test_tilt_constraint_rejects_dominant_wall():
    rng = np.random.default_rng(3)
    wall = np.c_[rng.uniform(-10, 10, 3000), rng.uniform(-3, 1.4, 3000), np.full(3000, 12.0)]  # faces camera
    road = np.c_[rng.uniform(-5, 5, 800), np.full(800, 1.4), rng.uniform(3, 30, 800)]
    pts = np.vstack([wall, road])
    _, d_free, _ = fit_plane_ransac(pts, iters=200, thresh=0.1)
    normal, d, _ = fit_plane_ransac(pts, iters=200, thresh=0.1, max_tilt_deg=30)
    assert abs(d_free) == pytest.approx(12.0, abs=0.1)        # unconstrained RANSAC picks the wall
    assert abs(d) == pytest.approx(1.4, abs=0.03)             # constrained RANSAC finds the road
    assert abs(normal[1]) > 0.99


def test_segment_ground_ignores_masked_rows():
    cam = Camera(320, 240, 70)
    v = np.arange(130, 240)
    u = np.arange(0, 320, 4)
    uu, vv = np.meshgrid(u, v)
    z = 1.4 * cam.fy / (vv - cam.cy)
    pts = cam.backproject(uu.ravel(), vv.ravel(), z.ravel())
    rows = vv.ravel()
    dash = rows >= 200                                          # "dashboard": a plane at 0.4 m below camera
    pts[dash] = cam.backproject(uu.ravel()[dash], rows[dash], 0.4 * cam.fy / (rows[dash] - cam.cy))
    info, _ = segment_ground(pts, rows, 240, seed_bottom_row=200)
    assert info["camera_height_m"] == pytest.approx(1.4, abs=0.02)
