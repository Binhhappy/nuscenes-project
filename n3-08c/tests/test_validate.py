"""Kiểm tra bộ đánh giá scene hợp lệ (evaluate.validate_scene) trên data thật: scene chèn đúng phải đạt,
scene bị sửa sai (chồng box, ra ngoài đường, lơ lửng, template cùng scene) phải bị bắt.

Chạy: pytest tests/test_validate.py
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from evaluate import validate_scene  # noqa: E402
from task_a.extract_object import TemplateBank  # noqa: E402
from task_a.run_insertion import run_insertion  # noqa: E402
from utils.nuscenes_loader import get_annotations  # noqa: E402


def _inserted(nusc):
    """Chèn 3 xe từ scene khác vào 1 sample; trả về result dict đã có >= 1 object."""
    rng = np.random.default_rng(config.SEED)
    bank = TemplateBank().load_or_build(nusc)
    for s in nusc.sample[5::40]:
        templates = bank.sample(3, rng, categories=["vehicle"], min_points=50, exclude_scenes=[s["scene_token"]],
                                required_attributes=config.TEMPLATE_REQUIRED_ATTRIBUTES)
        res = run_insertion(nusc, s["token"], templates, rng=rng)
        if res["n_inserted"]:
            return res
    raise RuntimeError("không chèn được object nào")


def _fresh(nusc, cache={}):
    if "res" not in cache:
        cache["res"] = _inserted(nusc)
    return copy.deepcopy(cache["res"])


def test_valid_insertion_passes(nusc):
    c = validate_scene(nusc, _fresh(nusc))
    assert c["valid"] and c["cross_scene"], c
    assert c["ground_err_max"] < 0.3


def test_overlap_detected(nusc):
    res = _fresh(nusc)
    existing = get_annotations(nusc, res["target_sample_token"])
    res["annotations_sensor"][0]["translation"][:2] = existing[0]["translation"][:2]
    c = validate_scene(nusc, res)
    assert not c["valid"] and "overlap" in c["reasons"]


def test_off_area_detected(nusc):
    res = _fresh(nusc)
    res["annotations_sensor"][0]["translation"][:2] = [500.0, 500.0]  # ngoài bán kính map đã cắt
    c = validate_scene(nusc, res)
    assert not c["valid"] and "off_area" in c["reasons"]


def test_floating_box_detected(nusc):
    res = _fresh(nusc)
    res["annotations_sensor"][0]["translation"][2] += 1.0
    c = validate_scene(nusc, res)
    assert not c["valid"] and "off_ground" in c["reasons"]


def test_same_scene_source_flagged(nusc):
    res = _fresh(nusc)
    res["source_ann_tokens"] = [get_annotations(nusc, res["target_sample_token"])[0]["token"]]
    assert not validate_scene(nusc, res)["cross_scene"]


def test_empty_scene_invalid(nusc):
    res = _fresh(nusc)
    res["annotations_sensor"], res["n_inserted"] = [], 0
    c = validate_scene(nusc, res)
    assert not c["valid"] and c["reasons"] == ["empty"]
