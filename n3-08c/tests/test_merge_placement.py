"""Kiểm tra luật vật lý khi chèn object (task_a/merge.py, task_a/placement.py): synthetic + data thật.

Chạy: python tests/test_merge_placement.py   (hoặc pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box as shp_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import GroundModel  # noqa: E402
from task_a.merge import (assign_rings, insert_object, match_density, points_in_new_box,  # noqa: E402
                          resolve_occlusion, ring_elevations)
from task_a.placement import count_obstacle_points, sample_position_for_template  # noqa: E402
from utils.transforms import wrap_angle  # noqa: E402

TABLE = np.radians(np.linspace(-30.67, 10.67, config.LIDAR_RINGS))


def _with_rings(xyz, rng):
    pts = np.c_[xyz, rng.uniform(0, 255, len(xyz)), np.zeros(len(xyz))].astype(np.float32)
    pts[:, 4] = assign_rings(pts, TABLE)
    return pts


def _scene(rng):
    """Mặt đất z = -1.8 trong [2, 30] x [-10, 10] + tường ở x = 20 (mỗi tia trúng tường có 1 điểm, như LiDAR)."""
    ground = np.c_[rng.uniform(2, 30, 6000), rng.uniform(-10, 10, 6000), np.full(6000, -1.8)]
    az = (np.arange(config.LIDAR_AZIMUTH_BINS) + 0.5) / config.LIDAR_AZIMUTH_BINS * 2 * np.pi - np.pi
    az, el = np.meshgrid(az[np.abs(az) < np.arctan2(10, 20)], TABLE)
    wall = np.c_[np.full(az.size, 20.0), 20 * np.tan(az.ravel()), 20 / np.cos(az.ravel()) * np.tan(el.ravel())]
    wall = wall[(wall[:, 2] > -1.8) & (wall[:, 2] < 2.0)]
    return _with_rings(np.r_[ground, wall], rng)


def _rear_face_template(rng, n=3000, distance=10.0):
    face = np.c_[np.full(n, -2.25), rng.uniform(-0.9, 0.9, n), rng.uniform(-0.7, 0.8, n)]
    return {"points": np.c_[face, rng.uniform(0, 255, (n, 2))].astype(np.float32), "size": [1.9, 4.5, 1.6],
            "category": "vehicle.car", "ann_token": "t", "distance": distance, "view_angle": np.pi}


def test_ring_table_synthetic(nusc=None):
    pts = _scene(np.random.default_rng(0))
    # ring 0 gom cả điểm thấp hơn -30.67° (đất sát xe), ring trên cao không có điểm -> chỉ so ring giữa
    assert np.allclose(ring_elevations(pts)[5:25], TABLE[5:25], atol=np.radians(0.5))


def test_ring_assignment_real(nusc):
    """Gán ring theo góc ngước phải khớp ring thật của LIDAR_TOP."""
    from utils.nuscenes_loader import get_pointcloud

    pts = get_pointcloud(nusc, nusc.sample[0]["token"])
    table = ring_elevations(pts)
    assert np.all(np.diff(table) > 0)
    r = np.hypot(pts[:, 0], pts[:, 1])
    diff = assign_rings(pts, table) - pts[:, 4].astype(int)
    assert (diff[r > 10] == 0).mean() > 0.99
    # < 10 m: gốc phát tia lệch khỏi gốc sensor frame nên có thể lệch 1 ring (ô range image kề nhau)
    assert (np.abs(diff[r > 3]) <= 1).mean() > 0.99


def test_occlusion_object_hides_wall(nusc=None):
    rng = np.random.default_rng(1)
    target = _scene(rng)
    aug, info = insert_object(target, _rear_face_template(rng), (10.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.8), rng)

    def wall_behind(p):
        return ((p[:, 0] > 19.9) & (np.abs(p[:, 1]) < 0.5) & (p[:, 2] > -1.6) & (p[:, 2] < -1.0)).sum()
    assert wall_behind(target) > 5 and wall_behind(aug) == 0
    assert info["n_target_occluded"] > 0 and info["n_visible"] == info["n_template"]


def test_occlusion_wall_hides_object(nusc=None):
    """Object đặt sau tường (x = 25) phải bị che gần hết."""
    rng = np.random.default_rng(2)
    _, info = insert_object(_scene(rng), _rear_face_template(rng, distance=25.0), (25.0, 0.0, 0.0),
                            (0.0, 0.0, 1.0, 1.8), rng)
    assert info["n_visible"] < 0.05 * info["n_template"]


def test_resolve_occlusion_keeps_unrelated_cells(nusc=None):
    rng = np.random.default_rng(3)
    pts = _scene(rng)
    keep = resolve_occlusion(pts, np.zeros(len(pts), bool))
    assert keep.all()


def test_occlusion_near_object_covers_adjacent_ring(nusc=None):
    """Object gần (< OCCLUSION_NEAR_RANGE) che cả điểm nền ở ring kề (ring object có thể lệch 1); object xa thì không."""
    def cloud(r_obj):
        # 1 điểm object ring 10, điểm nền ở ring 9/10/11 cùng azimuth, xa 30 m
        pts = np.array([[r_obj, 0, 0, 0, 10], [30, 0, 0, 0, 9], [30, 0, 0, 0, 10], [30, 0, 0, 0, 11]], np.float32)
        return resolve_occlusion(pts, np.array([True, False, False, False]))
    assert cloud(5.0).tolist() == [True, False, False, False]
    assert cloud(20.0).tolist() == [True, True, False, True]


def test_required_attributes(nusc=None):
    from task_a.extract_object import has_required_attributes

    req = config.TEMPLATE_REQUIRED_ATTRIBUTES
    assert has_required_attributes({"category": "vehicle.motorcycle", "attributes": ["cycle.with_rider"]}, req)
    assert not has_required_attributes({"category": "vehicle.motorcycle", "attributes": ["cycle.without_rider"]}, req)
    assert not has_required_attributes({"category": "vehicle.bicycle"}, req)
    assert has_required_attributes({"category": "vehicle.car", "attributes": ["vehicle.parked"]}, req)


def test_bank_filters_riderless_cycles(nusc):
    from task_a.extract_object import TemplateBank

    bank = TemplateBank().load_or_build(nusc)
    pool = bank.filter(categories=["vehicle.motorcycle"], required_attributes=config.TEMPLATE_REQUIRED_ATTRIBUTES)
    assert pool and all("cycle.with_rider" in m["attributes"] for m in pool)
    assert len(pool) < len(bank.filter(categories=["vehicle.motorcycle"]))


def test_match_density(nusc=None):
    rng = np.random.default_rng(4)
    pts = np.zeros((10000, 5), np.float32)
    assert len(match_density(pts, 10.0, 8.0, rng)) == 10000           # kéo lại gần: giữ nguyên
    assert abs(len(match_density(pts, 10.0, 20.0, rng)) - 2500) < 150  # xa gấp đôi: ~1/4


def test_box_clearing_keeps_ground(nusc=None):
    rng = np.random.default_rng(5)
    pts = np.r_[_with_rings(np.c_[rng.uniform(8, 12, (500, 2)), np.full(500, -1.8)], rng),
                _with_rings(np.c_[rng.uniform(9, 11, (200, 2)), rng.uniform(-1.5, -0.5, 200)], rng)]
    box = {"translation": [10.0, 10.0, -1.0], "size": [3.0, 3.0, 1.6], "rotation": [1, 0, 0, 0]}
    m = points_in_new_box(pts, box)
    assert m[500:].all() and not m[:500].any()


def test_placement_keeps_distance_and_view(nusc=None):
    rng = np.random.default_rng(6)
    tpl = dict(_rear_face_template(rng, distance=12.0), view_angle=0.7)
    area = shp_box(-40, -40, 40, 40)
    ground = GroundModel.from_points(_with_rings(np.c_[rng.uniform(-40, 40, (20000, 2)),
                                                       np.full(20000, -1.8)], rng), rng=rng)
    for _ in range(20):
        x, y, yaw = sample_position_for_template(area, [], tpl, ground, rng=rng, r_scale_max=1.0)
        assert abs(np.hypot(x, y) - 12.0) <= 12.0 * 0.05 + 1e-6
        assert abs(wrap_angle(np.arctan2(y, x) - yaw - 0.7)) < 1e-6


def test_placement_rejects_obstacle_and_no_ground(nusc=None):
    rng = np.random.default_rng(7)
    tpl = dict(_rear_face_template(rng, distance=12.0), view_angle=0.0)
    area = shp_box(-40, -40, 40, 40)
    # Mặt đất chỉ có ở nửa x < 0 -> mọi vị trí x > 0 bị no_ground
    ground = GroundModel.from_points(_with_rings(np.c_[rng.uniform(-40, 0, (20000, 1)), rng.uniform(-40, 40, (20000, 1)),
                                                       np.full(20000, -1.8)], rng), rng=rng)
    stats = {}
    for _ in range(20):
        x, _, _ = sample_position_for_template(area, [], tpl, ground, rng=rng, stats=stats)
        assert x < config.GROUND["radius"]  # support lấy trong bán kính radius quanh (x, y)
    assert stats.get("no_ground", 0) > 0
    pole = np.c_[np.zeros((50, 2)) + [5.0, 0.0], np.linspace(-1.5, 1.0, 50)]
    assert count_obstacle_points(pole, (5.0, 0.0), [1.9, 4.5, 1.6], 0.0, -1.8) > config.MAX_OBSTACLE_POINTS


def test_real_insertion_is_physical(nusc):
    """Data thật: object chèn giữ khoảng cách, mặt có điểm quay về sensor, đứng trên mặt đất nhìn thấy."""
    from task_a.run_insertion import collect_templates, run_insertion

    tpls = collect_templates(nusc, nusc.sample[0]["token"], None, 3)
    r = run_insertion(nusc, nusc.sample[10]["token"], tpls, rng=np.random.default_rng(config.SEED))
    assert r["n_inserted"] >= 1
    for a, q in zip(r["annotations_sensor"], r["quality"]):
        assert q["source_distance"] * 0.94 <= q["distance"] <= q["source_distance"] * config.MAX_DISTANCE_SCALE
        assert q["ground_support"] >= config.GROUND["min_support"]
        assert q["facing_cos"] > -0.3   # nhìn ngang hông thì cos ~ 0
        assert config.MIN_VISIBLE_POINTS <= a["num_lidar_pts"] <= q["n_template"]


if __name__ == "__main__":
    from utils.nuscenes_loader import load_nuscenes

    _nusc = load_nuscenes()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(_nusc)
            print("PASS", name)
