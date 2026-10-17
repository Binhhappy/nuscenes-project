"""Tiêu chí chọn ảnh nguồn (utils/camera_paste.pick_source) và dán ảnh object lên camera trên data thật.

Chạy: pytest tests/test_camera_paste.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.camera_paste import MAX_VIEW_DIFF, pick_source  # noqa: E402


def _cand(**kw):
    c = {"az": 0.0, "offset": 0, "area": 10000.0, "occ": 0.0, "sharp": 1.0, "lum": 100.0, "cond": frozenset()}
    c.update(kw)
    return c


def test_pick_prefers_exact_frame_over_bigger_neighbor():
    best, _ = pick_source([_cand(offset=1, area=90000.0), _cand(offset=0)], 0.0)
    assert best["offset"] == 0


def test_pick_prefers_sharper_and_unoccluded():
    best, _ = pick_source([_cand(sharp=0.3), _cand(occ=0.05, sharp=0.9, area=12000.0)], 0.0)
    assert best["sharp"] == 0.9


def test_pick_gates():
    assert pick_source([_cand(az=np.radians(MAX_VIEW_DIFF + 5))], 0.0) == (None, "view_mismatch")
    assert pick_source([_cand(lum=30.0)], 0.0, target_lum=120.0, relax=False) == (None, "lighting_mismatch")
    assert pick_source([_cand(cond=frozenset({"night"}))], 0.0, target_cond=frozenset(),
                       relax=False) == (None, "condition_mismatch")


def test_relaxed_fallback_prefers_matching_condition():
    night, day = _cand(cond=frozenset({"night"}), area=90000.0), _cand(area=5000.0)
    best, _ = pick_source([night, day], 0.0, target_cond=frozenset())
    assert best["area"] == 5000.0 and not best["relaxed"]  # có ảnh cùng điều kiện thì dùng, dù nhỏ hơn
    best, _ = pick_source([night], 0.0, target_cond=frozenset())
    assert best["relaxed"]  # chỉ có ảnh đêm: vẫn dán (cổng mềm)
    best, why = pick_source([_cand(cond=frozenset({"night"}), az=1.0)], 0.0, target_cond=frozenset())
    assert best is None and why == "view_mismatch"  # cổng góc nhìn vẫn cứng


def test_asset_paste_for_raycast_object(nusc, tmp_path, monkeypatch):
    """Vật thể VN ray-cast (không có ảnh gốc) được dán từ ảnh PNG tải về: đúng mặt, nằm trong khung box chiếu."""
    import cv2
    from PIL import Image

    import utils.camera_paste as cp
    from task_a.extract_object import TemplateBank
    from task_a.vn_scenarios import VN_SCENARIOS, run_vn_scenario
    from utils.camera_proj import CAMS, lidar_to_cam, project
    from utils.visualization import to_box

    colors = {"front": (255, 0, 0), "rear": (0, 0, 255), "side": (0, 255, 0)}
    for view, rgb in colors.items():  # ảnh giả: khối màu đặc theo mặt, nền trong suốt
        im = np.zeros((120, 200, 4), np.uint8)
        im[10:110, 20:180] = (*rgb[::-1], 255)
        (tmp_path / "ba_gac").mkdir(exist_ok=True)
        cv2.imwrite(str(tmp_path / "ba_gac" / f"{view}.png"), im)
    monkeypatch.setattr(cp, "ASSET_DIR", tmp_path)

    tok = nusc.sample[10]["token"]
    res = run_vn_scenario(nusc, tok, VN_SCENARIOS["ba_gac_slow"], TemplateBank().load_or_build(nusc),
                          np.random.default_rng(0), 1000)
    anns = [a for a in res["annotations_sensor"] if a["category"] == "vn.ba_gac"]
    assert anns
    n = 0
    for cam in CAMS:
        T, K, path = lidar_to_cam(nusc, tok, cam)
        img = np.asarray(Image.open(path).convert("RGB"))
        out, status, src = cp.paste_objects(nusc, img, T, K, anns, [], {})
        for a, st, s in zip(anns, status, src):
            if st != "pasted_asset":
                continue
            n += 1
            uv, _, _ = project(T, K, to_box(a).corners().T, cp.W, cp.H)
            x0, y0 = np.clip(uv.min(0), 0, None).astype(int)
            x1, y1 = np.ceil(uv.max(0)).astype(int)
            changed = np.abs(out.astype(int) - img).sum(2) > 30
            inside = changed[max(y0 - 2, 0):y1 + 3, max(x0 - 60, 0):x1 + 60].sum()  # + lề cho tỉ lệ ảnh / bóng
            assert inside > 0.9 * changed.sum()
            az = cp._view_azimuth(to_box(a).center, to_box(a).rotation_matrix, np.linalg.inv(T)[:3, 3])
            assert s["view"] == cp._asset_view(az, {v: None for v in colors})
    assert n > 0


def test_asset_view_nearest_and_variants(tmp_path, monkeypatch):
    """5 mặt: chọn mặt có góc tâm gần nhất trong số mặt có ảnh; biến thể <mặt>_*.png được nạp chung."""
    import cv2

    import utils.camera_paste as cp
    views = {v: None for v in cp.ASSET_VIEWS}
    assert cp._asset_view(np.radians(10), views) == "front"
    assert cp._asset_view(np.radians(-50), views) == "oblique_front"
    assert cp._asset_view(np.radians(95), views) == "side"
    assert cp._asset_view(np.radians(140), views) == "oblique_rear"
    assert cp._asset_view(np.radians(-175), views) == "rear"
    assert cp._asset_view(np.radians(140), {"side": None, "rear": None}) == "rear"
    assert cp._asset_view(0.0, {}) is None

    im = np.zeros((20, 30, 4), np.uint8)
    im[2:18, 3:27, 3] = 255
    (tmp_path / "dog").mkdir()
    for name in ("side.png", "side_2.png", "oblique_front.png", "front_b.png"):
        cv2.imwrite(str(tmp_path / "dog" / name), im)
    monkeypatch.setattr(cp, "ASSET_DIR", tmp_path)
    got = {v: [n for n, _ in lst] for v, lst in cp.load_asset("dog").items()}
    assert got == {"side": ["side.png", "side_2.png"], "oblique_front": ["oblique_front.png"], "front": ["front_b.png"]}


def test_paste_changes_only_object_region(nusc):
    from task_a.run_insertion import collect_from_bank, run_insertion
    from utils.camera_paste import paste_objects, scene_conditions
    from utils.camera_proj import CAMS, lidar_to_cam
    from PIL import Image

    tok = nusc.sample[10]["token"]
    rng = np.random.default_rng(42)
    res = run_insertion(nusc, tok, collect_from_bank(nusc, tok, 6, ["vehicle.car", "vehicle.truck"], rng=rng), rng=rng)
    pasted = 0
    for cam in CAMS:
        T, K, path = lidar_to_cam(nusc, tok, cam)
        img = np.asarray(Image.open(path).convert("RGB"))
        out, status, src = paste_objects(nusc, img, T, K, res["annotations_sensor"], res["existing_annotations"], {},
                                         scene_conditions(nusc, tok))
        assert out.shape == img.shape and len(status) == len(res["annotations_sensor"])
        changed = np.abs(out.astype(int) - img).sum(2) > 0
        if not any(st.startswith("pasted") for st in status):
            assert not changed.any()
        else:
            pasted += 1
            assert 0 < changed.mean() < 0.5
            assert all(s["occ"] <= 0.1 and s["view_diff"] <= MAX_VIEW_DIFF for s in src if s)
    assert pasted > 0
