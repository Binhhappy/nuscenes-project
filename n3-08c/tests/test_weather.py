"""Kiểm tra task_b (rain, wet_road, snow, run_weather) trên point cloud tổng hợp — KHÔNG cần dataset.

Chạy: python tests/test_weather.py   (hoặc pytest tests/test_weather.py)
"""
import inspect
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from evaluate import evaluate_weather  # noqa: E402
from task_b.rain import simulate_rain  # noqa: E402
from task_b.run_weather import LEVELS, apply_weather, run_weather  # noqa: E402
from task_b.snow import simulate_snow  # noqa: E402
from task_b.wet_road import ground_mask, in_puddles, simulate_puddles  # noqa: E402
from utils.nuscenes_loader import load_bin  # noqa: E402
from utils.visualization import label_colors  # noqa: E402

N_GROUND, N_OBJ = 20000, 5000
ALL_LABELS = set(config.LABEL_NAMES)
SIMULATORS = {"rain": simulate_rain, "wet_road": simulate_puddles, "snow": simulate_snow}


def _synthetic(seed=0, slope=0.0):
    """N_GROUND điểm mặt đường z = -1.8 + slope*x (noise 2 cm) rồi tới N_OBJ điểm vật thể cao 0.5-1.8 m."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-40, 40, (N_GROUND + N_OBJ, 2))
    z = -1.8 + slope * xy[:, 0] + np.r_[rng.normal(0, 0.02, N_GROUND), rng.uniform(0.5, 1.8, N_OBJ)]
    return np.c_[xy, z, rng.uniform(1, 255, len(xy)), rng.integers(0, 32, len(xy))].astype(np.float32)


def _raises(exc, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def test_output_format():
    pts = _synthetic()
    for name, fn in SIMULATORS.items():
        for level in LEVELS[name]:
            out, lab = fn(pts, level, np.random.default_rng(1), return_labels=True)
            assert out.ndim == 2 and out.shape[1] == 5 and out.dtype == np.float32, (name, level, out.shape)
            assert lab.shape == (len(out),), (name, level)
            assert set(np.unique(lab).tolist()) <= ALL_LABELS, (name, level)
            assert np.isfinite(out).all(), (name, level)
            only = fn(pts, level, np.random.default_rng(1))  # không return_labels -> cùng points
            assert np.array_equal(only, out), (name, level)
        assert _raises(ValueError, fn, pts, "khong_co_level"), name


def test_same_seed_same_result():
    pts = _synthetic()
    for name, fn in SIMULATORS.items():
        a, la = fn(pts, "moderate", np.random.default_rng(3), return_labels=True)
        b, lb = fn(pts, "moderate", np.random.default_rng(3), return_labels=True)
        c, _ = fn(pts, "moderate", np.random.default_rng(4), return_labels=True)
        assert np.array_equal(a, b) and np.array_equal(la, lb), name
        assert a.shape != c.shape or not np.array_equal(a, c), f"{name}: seed khác mà kết quả giống hệt"


def test_input_not_modified():
    pts = _synthetic()
    ref = pts.copy()
    for fn in SIMULATORS.values():
        fn(pts, "heavy", np.random.default_rng(0))
    assert np.array_equal(pts, ref)


def test_heavier_level_loses_more():
    """Level nặng hơn: còn ít điểm gốc hơn; với mưa và tuyết thì tổng số điểm và intensity TB cũng giảm."""
    pts = _synthetic()
    for name, fn in SIMULATORS.items():
        n_orig, n_total, mean_i = [len(pts)], [len(pts)], [float(pts[:, 3].mean())]
        for level in LEVELS[name]:
            out, lab = fn(pts, level, np.random.default_rng(5), return_labels=True)
            n_orig.append(int((lab != config.LABEL_NOISE).sum()))
            n_total.append(len(out))
            mean_i.append(float(out[:, 3].mean()))
        assert all(a > b for a, b in zip(n_orig, n_orig[1:])), (name, n_orig)
        assert all(a > b for a, b in zip(n_total, n_total[1:])), (name, n_total)
        if name != "wet_road":  # đường ướt chỉ đổi điểm trong vũng, intensity TB toàn cảnh gần như không đổi
            assert all(a > b for a, b in zip(mean_i, mean_i[1:])), (name, mean_i)


def test_ground_mask():
    for slope in (0.0, 0.04):
        pts = _synthetic(slope=slope)
        m = ground_mask(pts)
        assert m[:N_GROUND].mean() > 0.99, f"slope {slope}: chỉ nhận {m[:N_GROUND].mean():.1%} điểm mặt đường"
        assert m[N_GROUND:].mean() < 0.01, f"slope {slope}: nhận nhầm {m[N_GROUND:].mean():.1%} điểm vật thể"
    # Fallback: không có điểm nào dưới z_threshold -> không fit được plane, không có mặt đường
    high = _synthetic()
    high[:, 2] += 5.0
    assert not ground_mask(high).any()
    out, lab = simulate_puddles(high, "heavy", np.random.default_rng(0), return_labels=True)
    assert np.array_equal(out, high) and not lab.any()


def test_puddle_only_changes_ground_points_inside():
    pts = _synthetic()
    puddle = np.array([[10.0, 0.0, 6.0, 5.0]])
    p = config.WET_ROAD_LEVELS["moderate"]
    is_ground = np.arange(len(pts)) < N_GROUND
    wet = is_ground & in_puddles(pts[:, :2], puddle)
    assert wet.sum() > 100

    out, lab = simulate_puddles(pts, "moderate", np.random.default_rng(2), return_labels=True, puddles=puddle)
    # Điểm ngoài vũng + điểm vật thể phía trên vũng: giữ nguyên, đúng thứ tự
    assert np.array_equal(out[lab == config.LABEL_ORIGINAL], pts[~wet])
    assert not (lab == config.LABEL_NOISE).any()
    # Điểm mặt đường trong vũng: phần lớn mất, phần còn lại giảm intensity
    survived = out[lab == config.LABEL_PUDDLE]
    frac = len(survived) / wet.sum()
    assert abs(frac - (1 - p["dropout"])) < 0.1, f"tỉ lệ sống sót {frac:.2f}"
    assert in_puddles(survived[:, :2], puddle).all()
    src = {tuple(r[:3]): r[3] for r in pts[wet]}
    expect = np.array([src[tuple(r[:3])] for r in survived]) * p["intensity_scale"]
    assert np.allclose(survived[:, 3], expect, rtol=1e-5)


def test_labels_are_carried_through():
    """labels đầu vào: điểm sống sót giữ nhãn cũ, điểm sinh thêm = LABEL_NOISE."""
    pts = _synthetic()
    lab_in = np.zeros(len(pts), np.int8)
    lab_in[::7] = config.LABEL_PUDDLE
    marked = {tuple(r[:3]) for r in pts[::7]}  # tuyết không dịch chuyển điểm còn sống
    for fn in (simulate_rain, simulate_snow):
        out, lab = fn(pts, "moderate", np.random.default_rng(6), return_labels=True, labels=lab_in)
        assert lab.shape == (len(out),)
        n_puddle = int((lab == config.LABEL_PUDDLE).sum())
        assert 0 < n_puddle < int((lab_in == config.LABEL_PUDDLE).sum())
        assert (lab == config.LABEL_NOISE).any()
    out, lab = simulate_snow(pts, "moderate", np.random.default_rng(6), return_labels=True, labels=lab_in)
    assert all(tuple(r[:3]) in marked for r in out[lab == config.LABEL_PUDDLE])


def test_combined_rain_wet_road():
    pts = _synthetic()
    out, lab = apply_weather(pts, "rain+wet_road", "heavy", np.random.default_rng(7))
    assert set(np.unique(lab).tolist()) == ALL_LABELS
    assert lab.shape == (len(out),)
    # Đúng bằng áp đường ướt trước rồi mưa, nhãn truyền qua
    rng = np.random.default_rng(7)
    p1, l1 = simulate_puddles(pts, "heavy", rng, return_labels=True)
    p2, l2 = simulate_rain(p1, "heavy", rng, return_labels=True, labels=l1)
    assert np.array_equal(out, p2) and np.array_equal(lab, l2)
    # Mất nhiều điểm gốc hơn so với chỉ mưa hoặc chỉ đường ướt
    n_combo = (lab != config.LABEL_NOISE).sum()
    for single in ("rain", "wet_road"):
        _, ls = apply_weather(pts, single, "heavy", np.random.default_rng(7))
        assert n_combo < (ls != config.LABEL_NOISE).sum(), single


def test_apply_weather_rejects_unknown():
    pts = _synthetic()
    assert _raises(ValueError, apply_weather, pts, "hail", "moderate")
    assert _raises(ValueError, apply_weather, pts, "fog", "heavy")
    assert _raises(ValueError, apply_weather, pts, "snow", "dense")


def test_run_weather_and_evaluate(tmp_path):
    pts = _synthetic()
    out_dir = str(tmp_path)
    for weather, levels in LEVELS.items():
        res = run_weather(pts, "synthetic", weather, levels[-1], np.random.default_rng(8), out_dir=out_dir)
        assert os.path.basename(res["dir"]) == f"{weather}_{levels[-1]}"
        saved = load_bin(os.path.join(res["dir"], "synthetic.bin"))
        labels = np.load(os.path.join(res["dir"], "synthetic_labels.npy"))
        assert np.array_equal(saved, res["points"]) and np.array_equal(labels, res["labels"])
        assert len(labels) == len(saved)

        rep = evaluate_weather(os.path.join(out_dir, "original"), res["dir"])
        assert rep["n_scenes"] == 1
        assert abs(rep["point_count_change"]["mean_ratio"] - len(saved) / len(pts)) < 1e-9
        assert set(rep["label_ratio"]) == set(config.LABEL_NAMES.values())
        assert abs(sum(rep["label_ratio"].values()) - 1.0) < 1e-9
    assert np.array_equal(load_bin(os.path.join(out_dir, "original", "synthetic.bin")), pts)


def test_label_colors():
    c = label_colors(np.array([config.LABEL_ORIGINAL, config.LABEL_NOISE, config.LABEL_PUDDLE, 9]))
    assert c.shape == (4, 3)
    assert len({tuple(r) for r in c[:3]}) == 3, "3 nhãn phải có 3 màu khác nhau"
    assert tuple(c[3]) == tuple(c[1])  # nhãn lạ tô như noise


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
