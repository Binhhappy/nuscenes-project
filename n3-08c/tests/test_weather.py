"""Luật vật lý của mô phỏng mưa / sương (task_b/rain.py, fog.py) bản v2, lề an toàn khi đặt object và
mặt đất đồng thuận (task_a/placement.py, ground_plane.py). Synthetic, không cần dataset.

Chạy: pytest tests/test_weather.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import ConsensusGround, GroundModel  # noqa: E402
from task_a.placement import check_collision_bev  # noqa: E402
from task_b.fog import simulate_fog  # noqa: E402
from task_b.rain import simulate_rain  # noqa: E402
from utils.transforms import yaw_to_quaternion  # noqa: E402


def _cloud(rng, n=40000, r_max=60.0):
    r = rng.uniform(1.0, r_max, n)
    az = rng.uniform(-np.pi, np.pi, n)
    el = np.radians(rng.uniform(-30.0, 10.0, n))
    return np.c_[r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el),
                 rng.uniform(0, 100, n), rng.integers(0, 32, n)].astype(np.float32)


def _unit(p):
    return p[:, :3] / np.linalg.norm(p[:, :3], axis=1, keepdims=True)


def test_rain_v2_dropout_grows_with_range():
    pts = _cloud(np.random.default_rng(0))
    out, lab = simulate_rain(pts, "heavy", np.random.default_rng(1), True, model="v2")
    kept = out[lab == 0]
    r_in, r_kept = np.linalg.norm(pts[:, :3], axis=1), np.linalg.norm(kept[:, :3], axis=1)
    near = (r_kept < 10).sum() / (r_in < 10).sum()
    far = (r_kept > 40).sum() / (r_in > 40).sum()
    assert near > far + 0.3, (near, far)


def test_rain_v2_noise_on_real_beams_and_replaces_them():
    pts = _cloud(np.random.default_rng(0), n=4000)
    out, lab = simulate_rain(pts, "heavy", np.random.default_rng(1), True, model="v2")
    noise, kept = out[lab == 1], out[lab == 0]
    assert len(noise) == int(config.RAIN_LEVELS["heavy"]["noise_ratio"] * len(pts))
    assert np.linalg.norm(noise[:, :3], axis=1).max() <= 10.0 + 1e-4
    # mỗi điểm nhiễu nằm trên tia (cùng hướng, cùng ring) của 1 điểm gốc; điểm gốc đó không còn trong output
    same_beam = (_unit(noise) @ _unit(pts).T > 1 - 1e-6) & (noise[:, 4:5] == pts[None, :, 4])
    assert same_beam.any(1).all()
    replaced = {tuple(p[:3]) for p in pts[same_beam.any(0)]}
    assert not replaced & {tuple(p[:3]) for p in kept}


def test_fog_v2_backscatter_replaces_beam():
    pts = _cloud(np.random.default_rng(0))
    v1, l1 = simulate_fog(pts, 20.0, np.random.default_rng(2), True, model="v1")
    v2, l2 = simulate_fog(pts, 20.0, np.random.default_rng(2), True, model="v2")
    assert l1.sum() == l2.sum()  # cùng số tia backscatter
    assert len(v2) < len(v1)     # v2 bỏ điểm gốc của các tia đó, v1 giữ cả 2
    assert len(v2) <= len(pts)   # mỗi tia tối đa 1 điểm


def test_collision_margin():
    a = {"translation": [0.0, 0.0, 0.0], "size": [2.0, 4.0, 1.5], "rotation": yaw_to_quaternion(0.0)}
    b = {"translation": [4.2, 0.0, 0.0], "size": [2.0, 4.0, 1.5], "rotation": yaw_to_quaternion(0.0)}  # cách 0.2 m
    assert not check_collision_bev(a, [b])
    assert check_collision_bev(a, [b], margin=0.3)
    assert not check_collision_bev(a, [b], margin=0.1)


def test_consensus_ground_rejects_unstable_area():
    rng = np.random.default_rng(0)
    flat = np.c_[rng.uniform(-30, 30, (20000, 2)), rng.normal(-1.8, 0.02, 20000)]
    pit = np.c_[rng.uniform(20, 24, (300, 1)), rng.uniform(-2, 2, (300, 1)), np.full((300, 1), -2.35)]
    pts = np.c_[np.r_[flat, pit], np.zeros((20300, 2))].astype(np.float32)
    base = GroundModel.from_points(pts, rng=np.random.default_rng(1))
    cons = ConsensusGround([base, GroundModel.from_points(pts, rng=np.random.default_rng(2), band=0.4)])
    assert cons.is_supported(0.0, 10.0)
    assert base.is_supported(22.0, 0.0) and not cons.is_supported(22.0, 0.0)  # model band hẹp không thấy hố
