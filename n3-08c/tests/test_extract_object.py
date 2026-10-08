"""Kiểm tra task_a/extract_object.py trên data thật.

Chạy: python tests/test_extract_object.py   (hoặc pytest tests/)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_a.extract_object import (TemplateBank, crop_template, extract_all_templates,  # noqa: E402
                                   extract_object_template, load_template, save_template)
from utils import nuscenes_loader as nl  # noqa: E402
from utils import transforms as tf  # noqa: E402


def _densest(nusc, tok):
    pts = nl.get_pointcloud(nusc, tok)
    anns = nl.get_annotations(nusc, tok)
    a = max(anns, key=lambda a: tf.points_in_cuboid(pts, a["translation"], a["size"], a["rotation"]).sum())
    return pts, a


def test_template_inside_box_and_roundtrip(nusc):
    tok = nusc.sample[0]["token"]
    pts, a = _densest(nusc, tok)
    t = crop_template(pts, a, tok, ground_margin=0.0)
    w, l, h = t["size"]
    assert np.all(np.abs(t["points"][:, :3]) <= np.array([l, w, h]) / 2 + 1e-4)
    assert t["num_points"] == tf.points_in_cuboid(pts, a["translation"], a["size"], a["rotation"]).sum()
    # Đặt template về đúng pose gốc phải ra đúng các điểm gốc (kể cả intensity, ring)
    back = tf.from_local_frame(t["points"], a["translation"], a["rotation"])
    m = tf.points_in_cuboid(pts, a["translation"], a["size"], a["rotation"])
    assert np.abs(back - pts[m]).max() < 1e-3


def test_ground_margin_strips_bottom(nusc):
    tok = nusc.sample[0]["token"]
    pts, a = _densest(nusc, tok)
    full, stripped = crop_template(pts, a, tok, ground_margin=0.0), crop_template(pts, a, tok, ground_margin=0.1)
    h = a["size"][2]
    assert stripped["points"][:, 2].min() >= -h / 2 + 0.1 - 1e-6
    n_bottom = (full["points"][:, 2] < -h / 2 + 0.1).sum()
    assert stripped["num_points"] == full["num_points"] - n_bottom


def test_view_angle(nusc):
    """view_angle = azimuth - yaw: object nằm ngay trước sensor, đầu hướng về sensor -> view_angle = 0."""
    pts = np.zeros((0, 5), np.float32)
    ann = {"token": "x", "category": "vehicle.car", "size": [2, 4, 1.5],
           "translation": [10.0, 0.0, -1.0], "rotation": tf.yaw_to_quaternion(0.0)}
    assert abs(crop_template(pts, ann)["view_angle"]) < 1e-9  # sensor ở phía sau object
    ann["rotation"] = tf.yaw_to_quaternion(np.pi)
    assert abs(abs(crop_template(pts, ann)["view_angle"]) - np.pi) < 1e-9  # sensor ở phía trước


def test_extract_all_matches_single(nusc):
    tok = nusc.sample[5]["token"]
    all_t = extract_all_templates(nusc, tok, categories=["vehicle"], min_points=30)
    assert all_t and all(t["category"].startswith("vehicle") and t["num_points"] >= 30 for t in all_t)
    one = extract_object_template(nusc, tok, all_t[0]["ann_token"])
    assert np.array_equal(one["points"], all_t[0]["points"])


def test_save_load_roundtrip(nusc):
    t = extract_all_templates(nusc, nusc.sample[0]["token"])[0]
    with tempfile.TemporaryDirectory() as d:
        back = load_template(save_template(t, d))
    assert np.array_equal(back["points"], t["points"])
    assert {k: back[k] for k in back if k != "points"} == {k: t[k] for k in t if k != "points"}


def test_bank_build_filter_sample(nusc):
    scenes = {s["token"]: s for s in nusc.scene}
    tokens = [s["token"] for s in nusc.sample[::50]]
    with tempfile.TemporaryDirectory() as d:
        bank = TemplateBank(d).build(nusc, sample_tokens=tokens, progress=False)
        assert len(bank) > 0 and os.path.isfile(bank.index_path)
        reloaded = TemplateBank(d).load()
        assert reloaded.meta == bank.meta

        scene0 = nusc.get("sample", tokens[0])["scene_token"]
        assert scene0 in scenes
        pool = bank.filter(exclude_scenes=[scene0], categories=["vehicle"], min_points=40)
        assert pool and all(m["scene_token"] != scene0 and m["num_points"] >= 40 for m in pool)

        rng = np.random.default_rng(0)
        picked = bank.sample(200, rng, balance=True)
        cats = {m["category"] for m in bank.meta}
        assert all(p["points"].shape == (p["num_points"], 5) for p in picked)
        # balance: mỗi category được chọn xấp xỉ đều (200 / số category)
        counts = np.array([sum(p["category"] == c for p in picked) for c in cats])
        assert counts.min() > 0.3 * 200 / len(cats)


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
