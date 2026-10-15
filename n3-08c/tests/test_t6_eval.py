"""Kiểm tra logic khớp prediction - GT của T6 (tools/detect.py, tools/t6_eval.py). Không cần GPU / torch.

Chạy: pytest tests/test_t6_eval.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.detect import gt_boxes, match  # noqa: E402
from tools.t6_eval import object_records, summarize  # noqa: E402
from utils.transforms import yaw_to_quaternion  # noqa: E402


def _ann(x, y, cat="vehicle.motorcycle", yaw=0.0, vel=(5.0, 0.0)):
    return {"translation": [x, y, -1.0], "size": [0.8, 2.0, 1.5], "rotation": yaw_to_quaternion(yaw),
            "category": cat, "num_lidar_pts": 30, "velocity_sensor": list(vel)}


def _pred(rows):
    """rows: (x, y, heading, vx, vy, score, label)."""
    boxes = np.array([[x, y, -1, 2, 0.8, 1.5, h, vx, vy] for x, y, h, vx, vy, _, _ in rows], float).reshape(-1, 9)
    return {"boxes": boxes, "scores": np.array([r[5] for r in rows], float),
            "labels": np.array([r[6] for r in rows], dtype=object)}


def test_gt_boxes_drops_unknown_category():
    gts = gt_boxes([_ann(5, 0), _ann(5, 0, cat="animal")])
    assert [g["cls"] for g in gts] == ["motorcycle"]


def test_match_class_distance_and_score():
    gts = gt_boxes([_ann(10, 0), _ann(20, 0, cat="vehicle.car")])
    pred = _pred([(10.5, 0, 0, 5, 0, 0.9, "motorcycle"),   # khớp GT 0
                  (20.2, 0, 0, 5, 0, 0.9, "truck"),        # sai lớp: không khớp GT 1
                  (20.1, 0, 0, 5, 0, 0.1, "car")])         # dưới ngưỡng score
    assert match(gts, pred) == [0, None]


def test_match_one_prediction_per_gt():
    gts = gt_boxes([_ann(10, 0), _ann(10, 1.0)])
    pred = _pred([(10, 0.5, 0, 5, 0, 0.9, "motorcycle")])  # 1 box cho 2 xe sát nhau
    assert sorted(match(gts, pred), key=lambda k: k is None) == [0, None]


def test_records_miss_types_and_flips():
    gts = gt_boxes([_ann(10, 0), _ann(10, 1.0), _ann(30, 0), _ann(40, 5),
                    _ann(15, 0, cat="vehicle.car", yaw=np.pi, vel=(-8.0, 0.0))])
    pred = _pred([(10, 0.4, 0, 5, 0, 0.9, "motorcycle"),    # khớp 1 trong 2 xe sát nhau -> xe kia "merged"
                  (30, 0, 0, 0, 0, 0.8, "bicycle"),         # xe máy bị nhận là xe đạp -> "confused"
                  (15, 0, 0.1, 8, 0, 0.9, "car")])          # xe ngược chiều, đoán hướng + vận tốc cùng chiều ego
    recs = object_records(gts, pred, "test", "tok", 0)
    assert [r["detected"] for r in recs] == [True, False, False, False, True]
    assert [r["miss"] for r in recs] == [None, "merged", "confused:bicycle", "missed", None]
    assert recs[4]["heading_flip"] is True and recs[4]["vel_flip"] is True
    assert recs[0]["heading_flip"] is False


def test_summarize_recall_per_group_class():
    recs = [{"group": "cut_in", "seed": s, "cls": "motorcycle", "dist": 5.0, "npts": 30, "detected": d,
             "miss": None if d else "missed", "heading_flip": None, "vel_flip": None}
            for s, d in [(0, True), (0, False), (1, True), (1, True)]]
    (row,) = summarize(recs)
    assert row["n"] == 4 and row["recall"] == 0.75 and row["missed"] == 0.25
    assert abs(row["recall_std_seeds"] - 0.25) < 1e-9  # seed 0: 50%, seed 1: 100%
