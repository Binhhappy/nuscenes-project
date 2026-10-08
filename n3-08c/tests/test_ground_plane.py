"""Kiểm tra task_a/ground_plane.py: synthetic (biết đáp án) + data thật.

Chạy: python tests/test_ground_plane.py   (hoặc pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_a.ground_plane import (GroundModel, estimate_ground_ransac, get_drivable_area,  # noqa: E402
                                 get_map_area, layers_for_category, load_map_for_sample,
                                 project_z_to_ground, select_ground_candidates)
from utils import nuscenes_loader as nl  # noqa: E402
from utils.transforms import points_in_cuboid  # noqa: E402


def _synthetic(rng, slope=0.02, n_ground=20000, n_obj=5000):
    """Mặt đất z = -1.8 + slope*x (+ noise 2 cm) và 'xe' cao 0.3-1.8 m trên mặt đất."""
    xy = rng.uniform(-40, 40, (n_ground, 2))
    ground = np.c_[xy, -1.8 + slope * xy[:, 0] + rng.normal(0, 0.02, n_ground)]
    oxy = rng.uniform(-40, 40, (n_obj, 2))
    obj = np.c_[oxy, -1.8 + slope * oxy[:, 0] + rng.uniform(0.3, 1.8, n_obj)]
    pts = np.r_[ground, obj]
    return np.c_[pts, np.zeros((len(pts), 2))].astype(np.float32)


def test_ransac_synthetic(nusc=None):
    rng = np.random.default_rng(0)
    pts = _synthetic(rng, slope=0.0)
    a, b, c, d = estimate_ground_ransac(pts, rng=rng)
    assert c > 0.999 and abs(a) < 0.01 and abs(b) < 0.01
    assert abs(project_z_to_ground(0, 0, (a, b, c, d)) + 1.8) < 0.02


def test_local_model_follows_slope(nusc=None):
    """Dốc 4%: plane chỉ fit được phần z < -1.5; GroundModel phải bám theo dốc ở mọi nơi có điểm."""
    rng = np.random.default_rng(1)
    pts = _synthetic(rng, slope=0.04)
    model = GroundModel.from_points(pts, rng=rng, band=2.0)
    xs = np.linspace(-35, 35, 15)
    z, support = model.height_at(xs, np.zeros_like(xs))
    assert np.all(support >= model.min_support)
    assert np.abs(z - (-1.8 + 0.04 * xs)).max() < 0.08


def test_candidates_reject_objects(nusc=None):
    rng = np.random.default_rng(2)
    pts = _synthetic(rng, slope=0.0)
    plane = estimate_ground_ransac(pts, rng=rng)
    cand = select_ground_candidates(pts, plane)
    # Điểm vật thể thấp (< band) chỉ lọt ở cell không có điểm mặt đất nào -> phải rất hiếm,
    # và median trong height_at phải không bị ảnh hưởng.
    leak = np.mean(cand[:, 2] > -1.8 + 0.25)
    assert leak < 0.01, f"{leak:.2%} ground candidates là điểm vật thể"
    model = GroundModel(plane, cand)
    xy = rng.uniform(-35, 35, (500, 2))
    assert np.abs(model.height_at(xy[:, 0], xy[:, 1])[0] + 1.8).max() < 0.03


def test_no_support_falls_back_to_plane(nusc=None):
    rng = np.random.default_rng(3)
    model = GroundModel.from_points(_synthetic(rng, slope=0.0), rng=rng)
    z, n = model.height_at(200.0, 200.0)
    assert n[0] == 0 and abs(z[0] - project_z_to_ground(200, 200, model.plane)) < 1e-9
    assert not model.is_supported(200.0, 200.0) and model.is_supported(0.0, 0.0)


def test_real_ground_vs_annotations(nusc):
    """Đáy box xe thật nhìn thấy được (>= 20 điểm, r < 30 m) phải gần mặt đất ước lượng."""
    errs = []
    for s in nusc.sample[::20]:
        pts = nl.get_pointcloud(nusc, s["token"])
        model = GroundModel.from_points(pts)
        for a in nl.get_annotations(nusc, s["token"]):
            x, y, z = a["translation"]
            if (not a["category"].startswith("vehicle.car") or np.hypot(x, y) > 30
                    or points_in_cuboid(pts, a["translation"], a["size"], a["rotation"]).sum() < 20):
                continue
            errs.append(project_z_to_ground(x, y, model) - (z - a["size"][2] / 2))
    errs = np.abs(errs)
    assert len(errs) > 50
    assert np.median(errs) < 0.15 and np.percentile(errs, 90) < 0.45, \
        f"median {np.median(errs):.3f}, p90 {np.percentile(errs, 90):.3f}"


def test_map_areas(nusc):
    tok = nusc.sample[0]["token"]
    m = load_map_for_sample(nusc, tok)
    assert load_map_for_sample(nusc, tok) is m, "map không được cache"
    drivable = get_drivable_area(nusc, m, tok)
    walk = get_map_area(nusc, m, tok, layers_for_category("human.pedestrian.adult"))
    assert drivable.area > 500 and walk.area > 100
    assert drivable.intersection(walk).area < 0.5 * walk.area  # vỉa hè phần lớn không phải lòng đường
    # Xe có annotation (tâm) phần lớn nằm trên drivable area (cùng frame -> transform đúng)
    from shapely.geometry import Point
    cars = [a for a in nl.get_annotations(nusc, tok) if a["category"].startswith("vehicle.car")
            and np.hypot(*a["translation"][:2]) < 60]
    on = np.mean([drivable.buffer(1.0).contains(Point(a["translation"][:2])) for a in cars])
    assert on > 0.7, f"chỉ {on:.0%} xe nằm trên drivable area"


def test_layers_for_category(nusc=None):
    assert layers_for_category("vehicle.car") == ("drivable_area",)
    assert "walkway" in layers_for_category("human.pedestrian.adult")
    assert layers_for_category("animal") == ("drivable_area",)


if __name__ == "__main__":
    nusc = nl.load_nuscenes()
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(nusc)
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
