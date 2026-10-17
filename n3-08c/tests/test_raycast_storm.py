"""Kiểm tra vật thể VN ray-cast (task_a/raycast.py), giông bão (task_b/storm.py), khớp vật cản không lớp.

Chạy: pytest tests/test_raycast_storm.py   (không cần data nuScenes)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_a.raycast import SHAPES, beam_table, build_model, raycast  # noqa: E402
from task_b.storm import GROUND_Z, simulate_storm, wet_road_mask  # noqa: E402
from tools.detect import class_ok, detection_class  # noqa: E402

HDL32 = beam_table(32)
GROUND = -1.84


def _cast(name, r, yaw=0.0, seed=0):
    m = build_model(name, np.random.default_rng(1))
    return m, raycast(m, [r, 0, GROUND + m["size"][2] / 2], yaw, HDL32, np.random.default_rng(seed))


def test_every_shape_builds_and_sits_on_ground():
    for name in SHAPES:
        m, pts = _cast(name, 8.0)
        assert all(s > 0 for s in m["size"]), name
        assert len(pts) > 0, name
        assert pts[:, 2].min() > GROUND - 0.1, name               # không xuyên mặt đất
        assert pts[:, 2].max() < GROUND + m["size"][2] + 0.1, name


def test_density_falls_with_distance():
    n = [len(_cast("car", r)[1]) for r in (5, 10, 20, 40)]
    assert n[0] > n[1] > n[2] > n[3]


def test_only_side_facing_sensor_is_hit():
    """Xe nằm ngang trước sensor: chỉ mặt gần sensor có điểm (x của điểm < tâm xe)."""
    m, pts = _cast("car", 10.0, yaw=np.pi / 2)
    assert (pts[:, 0] < 10.0).mean() > 0.95


def test_points_on_beam_rings():
    _, pts = _cast("moto_multi", 6.0)
    el = np.arctan2(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1]))
    assert np.allclose(el, HDL32[pts[:, 4].astype(int)], atol=0.01)  # mỗi điểm nằm đúng góc ngước của tia


def test_insert_uses_sensor_spec_not_scan_rings():
    """Scan có cột ring hỏng (mọi ring cùng góc) không được làm bắn dồn tia: chó 13 m chỉ ~1-2 tia dọc."""
    from task_a.raycast import insert_raycast
    rng = np.random.default_rng(0)
    r = rng.uniform(3, 40, 20000)
    a = rng.uniform(-np.pi, np.pi, 20000)
    scan = np.c_[r * np.cos(a), r * np.sin(a), np.full(20000, GROUND), rng.uniform(0, 50, 20000),
                 rng.integers(0, 32, 20000)].astype(np.float32)
    scan[:, 4] = 25  # ring hỏng: mọi điểm cùng 1 ring
    out, info = insert_raycast(scan, build_model("dog"), (13.0, 0.0, np.pi / 2), (0.0, 0.0, 1.0, -GROUND), rng)
    assert 0 < info["n_visible"] <= 30


def test_puddles_optional_and_local():
    """Vũng nước tắt mặc định (giữ số đo cũ); bật thì chỉ mất điểm mặt đường gần xe."""
    from task_b.storm import puddle_mask
    rng = np.random.default_rng(0)
    r = rng.uniform(2, 50, 20000)
    a = rng.uniform(-np.pi, np.pi, 20000)
    pts = np.c_[r * np.cos(a), r * np.sin(a), np.full(20000, GROUND), np.ones(20000), np.zeros(20000)].astype(np.float32)
    a1 = simulate_storm(pts, np.random.default_rng(3))
    a2 = simulate_storm(pts, np.random.default_rng(3), puddles=0)
    assert np.array_equal(a1, a2)
    keep = puddle_mask(pts, np.random.default_rng(1), n=8)
    assert keep[r > 30].all() and 0 < (~keep).sum() < len(pts)


def test_camera_weather_shapes():
    from task_b.camera_weather import apply_camera_weather
    img = np.full((90, 160, 3), 100, np.uint8)
    for w, lv in (("rain", "heavy"), ("fog", "dense"), ("storm", "storm")):
        out = apply_camera_weather(img, w, lv, np.random.default_rng(0))
        assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)


def test_pcd_roundtrip(tmp_path):
    from utils.pcd_io import load_pcd, save_pcd
    pts = np.random.default_rng(0).normal(size=(500, 5)).astype(np.float32)
    save_pcd(pts, str(tmp_path / "a.pcd"))
    assert np.array_equal(load_pcd(str(tmp_path / "a.pcd")), pts)


def test_ply_roundtrip(tmp_path):
    from utils.pcd_io import load_ply, save_ply
    pts = np.random.default_rng(0).normal(size=(500, 5)).astype(np.float32)
    save_ply(pts, str(tmp_path / "a.ply"))
    assert np.array_equal(load_ply(str(tmp_path / "a.ply")), pts)
    head = open(tmp_path / "a.ply", "rb").read(200).decode("ascii", "ignore")
    assert "element vertex 500" in head and "property float intensity" in head and "property float ring" in head


def test_vn_categories_map_to_classes():
    assert detection_class("vn.moto_multi") == "motorcycle"
    assert detection_class("vn.ba_gac") == "obstacle:ba_gac"
    assert class_ok("obstacle:dog", "pedestrian") and class_ok("car", "car") and not class_ok("car", "truck")


def test_storm_removes_far_ground_and_adds_spray():
    rng = np.random.default_rng(0)
    r = rng.uniform(2, 50, 20000)
    a = rng.uniform(-np.pi, np.pi, 20000)
    pts = np.c_[r * np.cos(a), r * np.sin(a), np.full(20000, GROUND), rng.uniform(0, 50, 20000),
                rng.integers(0, 32, 20000)].astype(np.float32)
    keep = wet_road_mask(pts, rng)
    near, far = r < 8, r > 40
    assert keep[near].all() and keep[far].mean() < 0.3
    car = {"translation": [10.0, 0.0, -1.0], "size": [1.8, 4.4, 1.5], "rotation": [1.0, 0, 0, 0]}
    out, lab = simulate_storm(pts, np.random.default_rng(1), [car], return_labels=True)
    ground_kept = ((out[:, 2] < GROUND_Z) & (lab == 0)).sum()
    assert ground_kept < len(pts) * 0.8
    spray = out[(lab == 1) & (out[:, 0] < 10 - 2.2) & (out[:, 0] > 10 - 2.2 - 3.1) & (np.abs(out[:, 1]) < 0.95)]
    assert len(spray) > 20  # đám bụi nước sau đuôi xe
