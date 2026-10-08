"""Kiểm tra utils/ trên data thật, đối chiếu với nuScenes devkit làm chuẩn.

Chạy: python tests/test_utils.py   (hoặc pytest tests/ nếu đã cài pytest)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLBACKEND", "Agg")

from nuscenes.utils.data_classes import LidarPointCloud  # noqa: E402
from nuscenes.utils.geometry_utils import points_in_box  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

import config  # noqa: E402
from utils import nuscenes_loader as nl  # noqa: E402
from utils import transforms as tf  # noqa: E402

N_SAMPLES = 5  # số sample kiểm tra (rải đều trong dataset)


def _samples(nusc):
    idx = np.linspace(0, len(nusc.sample) - 1, N_SAMPLES).astype(int)
    return [nusc.sample[i]["token"] for i in idx]


def test_pointcloud_matches_devkit(nusc):
    for tok in _samples(nusc):
        pts = nl.get_pointcloud(nusc, tok)
        ref = LidarPointCloud.from_file(nusc.get_sample_data_path(nl.get_lidar_token(nusc, tok))).points
        assert pts.dtype == np.float32 and pts.shape[1] == 5
        assert np.array_equal(pts[:, :4].T, ref), "xyz/intensity khác devkit"
        assert set(np.unique(pts[:, 4])) <= set(range(32)), "ring ngoài [0, 31]"


def test_calib_pose_records(nusc):
    sd = nl.get_lidar_token(nusc, nusc.sample[0]["token"])
    calib, pose = nl.get_calibration(nusc, sd), nl.get_ego_pose(nusc, sd)
    for rec in (calib, pose):
        assert len(rec["translation"]) == 3 and len(rec["rotation"]) == 4
    assert 1.5 < calib["translation"][2] < 2.2, "LIDAR_TOP phải cao ~1.84 m so với ego"


def test_annotations_sensor_vs_global(nusc):
    """box_sensor_to_global(ann sensor) phải trùng ann global gốc."""
    for tok in _samples(nusc):
        sd = nl.get_lidar_token(nusc, tok)
        calib, pose = nl.get_calibration(nusc, sd), nl.get_ego_pose(nusc, sd)
        glob = {a["token"]: a for a in nl.get_annotations(nusc, tok, frame="global")}
        sens = nl.get_annotations(nusc, tok, frame="sensor")
        assert len(glob) == len(sens) == len(nusc.get("sample", tok)["anns"])
        for a in sens:
            g, b = glob[a["token"]], tf.box_sensor_to_global(a, calib, pose)
            assert np.allclose(b["translation"], g["translation"], atol=1e-6)
            assert Quaternion.absolute_distance(Quaternion(b["rotation"]), Quaternion(g["rotation"])) < 1e-6
            assert np.allclose(a["size"], g["size"]) and a["category"] == g["category"]


def test_sensor_global_roundtrip_and_devkit(nusc):
    tok = nusc.sample[0]["token"]
    sd = nl.get_lidar_token(nusc, tok)
    calib, pose = nl.get_calibration(nusc, sd), nl.get_ego_pose(nusc, sd)
    pts = nl.get_pointcloud(nusc, tok)

    g = tf.sensor_to_global(pts, calib, pose)
    # Devkit: rotate/translate tuần tự sensor -> ego -> global
    pc = LidarPointCloud(pts[:, :4].T.astype(np.float64).copy())
    pc.rotate(Quaternion(calib["rotation"]).rotation_matrix)
    pc.translate(np.array(calib["translation"]))
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
    pc.translate(np.array(pose["translation"]))
    assert np.allclose(g[:, :3], pc.points[:3].T, atol=1e-3)
    assert np.array_equal(g[:, 3:], pts[:, 3:]), "cột intensity/ring bị đổi"

    back = tf.global_to_sensor(g, calib, pose)
    assert np.abs(back[:, :3] - pts[:, :3]).max() < 1e-3


def test_points_in_cuboid_matches_devkit(nusc):
    from nuscenes.utils.data_classes import Box

    for tok in _samples(nusc):
        pts = nl.get_pointcloud(nusc, tok)
        for a in nl.get_annotations(nusc, tok):
            box = Box(a["translation"], a["size"], Quaternion(a["rotation"]))
            ref = points_in_box(box, pts[:, :3].T)
            got = tf.points_in_cuboid(pts, a["translation"], a["size"], a["rotation"])
            assert np.array_equal(got, ref), f"lệch {np.sum(got != ref)} điểm ở {a['token']}"


def test_local_frame_roundtrip(nusc):
    tok = nusc.sample[0]["token"]
    pts = nl.get_pointcloud(nusc, tok)
    a = max(nl.get_annotations(nusc, tok),
            key=lambda a: tf.points_in_cuboid(pts, a["translation"], a["size"], a["rotation"]).sum())
    m = tf.points_in_cuboid(pts, a["translation"], a["size"], a["rotation"])
    local = tf.to_local_frame(pts[m], a["translation"], a["rotation"])
    w, l, h = a["size"]
    assert np.all(np.abs(local[:, :3]) <= np.array([l, w, h]) / 2 + 1e-4), "local ngoài box"
    back = tf.from_local_frame(local, a["translation"], a["rotation"])
    assert np.abs(back[:, :3] - pts[m, :3]).max() < 1e-4
    # Dời sang pose mới: khoảng cách giữa các điểm giữ nguyên (phép biến đổi cứng)
    moved = tf.from_local_frame(local, [5.0, -3.0, 0.2], tf.yaw_to_quaternion(1.0))
    d0 = np.linalg.norm(local[:, :3] - local[0, :3], axis=1)
    d1 = np.linalg.norm(moved[:, :3] - moved[0, :3], axis=1)
    assert np.allclose(d0, d1, atol=1e-4)


def test_yaw_quaternion():
    for yaw in np.linspace(-3.1, 3.1, 13):
        assert abs(tf.quaternion_to_yaw(tf.yaw_to_quaternion(yaw)) - yaw) < 1e-9


def test_visualization_bev(nusc, tmp_dir=os.path.join(config.VIS_DIR, "_test")):
    from utils.visualization import label_colors, plot_bev

    tok = nusc.sample[0]["token"]
    pts, anns = nl.get_pointcloud(nusc, tok), nl.get_annotations(nusc, tok)
    out = os.path.join(tmp_dir, "bev.png")
    plot_bev(pts, anns, anns[:1], colors=label_colors(np.zeros(len(pts))), save_path=out)
    assert os.path.getsize(out) > 10_000


if __name__ == "__main__":
    nusc = nl.load_nuscenes()
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(nusc) if fn.__code__.co_argcount else fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
