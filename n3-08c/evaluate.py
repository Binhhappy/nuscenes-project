"""Đánh giá: Task A (tỉ lệ scene hợp lệ, scene/giờ, số object, chất lượng vật lý) và Task B (thay đổi point count, intensity).

Chỉ số theo đề bài:
    valid_scene_rate   tỉ lệ scene sinh ra hợp lệ, kiểm tra lại độc lập trên file đầu ra (validate_scene):
                       có object chèn, không chồng box (BEV), đúng mặt đường / vỉa hè, đáy box chạm mặt đất
    scenes_per_hour    wall-clock toàn batch (lấy template + chèn + thời tiết + ghi file), từ batch_timing.json

Chạy:
    python evaluate.py                       # đánh giá kết quả có sẵn trong output/
    python evaluate.py --run-insertion 100 --augmented-dir ../output/augmented_eval   # 100 sample, nguồn khác scene
    python evaluate.py --run-insertion 100 --weather rain heavy --augmented-dir ../output/augmented_eval_rain
    python evaluate.py --run-insertion 20 --naive --augmented-dir ../output/augmented_naive   # bản cũ để so
"""
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from utils.nuscenes_loader import load_bin  # noqa: E402


def evaluate_insertion(results_dir: str = config.AUGMENTED_DIR) -> dict:
    """Đọc *.json do task_a/run_insertion.py ghi ra.

    Returns dict: valid_rate (inserted / requested), total_objects, scenes_per_hour, n_scenes, per_category,
    reject_stats (lý do loại vị trí), quality (insertion_quality, None với kết quả của bản cũ).
    """
    files = _result_files(results_dir)
    if not files:
        raise FileNotFoundError(f"Không có kết quả insertion trong {results_dir}")
    requested = inserted = 0
    elapsed = 0.0
    per_cat, rejects, objects = {}, {}, []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        requested += r["n_requested"]
        inserted += r["n_inserted"]
        elapsed += r["elapsed_s"]
        for a in r["annotations_sensor"]:
            per_cat[a["category"]] = per_cat.get(a["category"], 0) + 1
        for k, v in r.get("reject_stats", {}).items():
            rejects[k] = rejects.get(k, 0) + v
        objects += [(a, q) for a, q in zip(r["annotations_sensor"], r.get("quality", []))]
    core_per_hour = len(files) / (elapsed / 3600) if elapsed > 0 else float("inf")
    timing = _load_timing(results_dir)
    return {
        "n_scenes": len(files),
        "valid_rate": inserted / requested if requested else 0.0,  # theo object, xem validity cho theo scene
        "total_objects": inserted,
        # wall-clock toàn batch nếu có batch_timing.json, nếu không thì chỉ thời gian trong run_insertion
        "scenes_per_hour": timing["scenes_per_hour"] if timing else core_per_hour,
        "scenes_per_hour_core": core_per_hour,
        "timing": timing,
        "per_category": per_cat,
        "reject_stats": rejects,
        "quality": insertion_quality(objects) if objects else None,
    }


TIMING_NAME = "batch_timing.json"
GROUND_TOL = 0.3  # (m) |đáy box - mặt đất cục bộ| tối đa để coi là chạm mặt đất


def _result_files(results_dir: str):
    return sorted(f for f in glob.glob(os.path.join(results_dir, "*.json")) if os.path.basename(f) != TIMING_NAME)


def _load_timing(results_dir: str):
    path = os.path.join(results_dir, TIMING_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_scene(nusc, result: dict, ground_tol: float = GROUND_TOL, cache: dict = None) -> dict:
    """Kiểm tra lại 1 scene sinh ra, độc lập với placement (không tin việc placement đã loại vị trí sai).

    Dựng lại từ nuScenes gốc: box có sẵn của sample đích, map layer theo category, mặt đất (GroundModel
    từ point cloud gốc, seed khác lúc chèn). Mỗi object chèn phải:
        no overlap   footprint BEV không giao box có sẵn và object chèn trước (diện tích giao > 1e-6 m2)
        on area      footprint nằm trong layer hợp lệ (xe: drivable_area, người: walkway / ped_crossing)
        on ground    LiDAR thấy mặt đất tại tâm và |đáy box - mặt đất| <= ground_tol
    Scene hợp lệ (valid): có >= 1 object chèn và mọi object đạt. complete: valid và chèn đủ số yêu cầu.
    cross_scene: không template nào lấy từ chính scene đích (đề bài: chèn vào scene khác).
    """
    from task_a.ground_plane import GroundModel, get_map_area, layers_for_category, load_map_for_sample
    from task_a.placement import box_footprint
    from utils.nuscenes_loader import get_annotations, get_pointcloud

    cache = {} if cache is None else cache
    tok = result["target_sample_token"]
    new = result["annotations_sensor"]
    target_scene = nusc.get("sample", tok)["scene_token"]
    src_scenes = {nusc.get("sample", nusc.get("sample_annotation", a)["sample_token"])["scene_token"]
                  for a in result.get("source_ann_tokens", []) if a}

    placed = [box_footprint(b) for b in get_annotations(nusc, tok)]
    n_overlap = 0
    for a in new:
        fp = box_footprint(a)
        n_overlap += any(fp.intersection(o).area > 1e-6 for o in placed)
        placed.append(fp)

    nusc_map = load_map_for_sample(nusc, tok)
    n_off_area = 0
    for a in new:
        layers = layers_for_category(a["category"])
        if (tok, layers) not in cache:
            cache[(tok, layers)] = get_map_area(nusc, nusc_map, tok, layers).buffer(1e-3)
        n_off_area += not cache[(tok, layers)].contains(box_footprint(a))

    ground_err, n_off_ground = [], 0
    if new:
        ground = GroundModel.from_points(get_pointcloud(nusc, tok), rng=np.random.default_rng(0))
        xy = np.array([a["translation"][:2] for a in new])
        z, support = ground.height_at(xy[:, 0], xy[:, 1])
        bottom = np.array([a["translation"][2] - a["size"][2] / 2 for a in new])
        ground_err = np.abs(bottom - z)
        n_off_ground = int(((ground_err > ground_tol) | (support < config.GROUND["min_support"])).sum())

    reasons = [k for k, bad in (("empty", not new), ("overlap", n_overlap), ("off_area", n_off_area),
                                ("off_ground", n_off_ground)) if bad]
    return {
        "target_sample_token": tok,
        "valid": not reasons,
        "complete": not reasons and result["n_inserted"] >= result["n_requested"],
        "cross_scene": target_scene not in src_scenes,
        "reasons": reasons,
        "n_inserted": len(new),
        "n_overlap": int(n_overlap),
        "n_off_area": int(n_off_area),
        "n_off_ground": n_off_ground,
        "ground_err_max": float(np.max(ground_err)) if len(ground_err) else None,
    }


def evaluate_validity(nusc, results_dir: str = config.AUGMENTED_DIR) -> dict:
    """Tỉ lệ scene hợp lệ (validate_scene) trên mọi *.json do run_insertion ghi ra."""
    files = _result_files(results_dir)
    if not files:
        raise FileNotFoundError(f"Không có kết quả insertion trong {results_dir}")
    cache, checks = {}, []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            checks.append(validate_scene(nusc, json.load(fh), cache=cache))
    n = len(checks)
    fails = {}
    for c in checks:
        for r in c["reasons"]:
            fails[r] = fails.get(r, 0) + 1
    errs = [c["ground_err_max"] for c in checks if c["ground_err_max"] is not None]
    return {
        "n_scenes": n,
        "valid_scene_rate": sum(c["valid"] for c in checks) / n,
        "complete_scene_rate": sum(c["complete"] for c in checks) / n,
        "cross_scene_rate": sum(c["cross_scene"] for c in checks) / n,
        "fail_reasons": fails,  # 1 scene có thể có nhiều lý do
        "objects": {k: sum(c[k] for c in checks) for k in ("n_inserted", "n_overlap", "n_off_area", "n_off_ground")},
        "ground_err_max": float(max(errs)) if errs else None,
        "invalid_scenes": [{"token": c["target_sample_token"], "reasons": c["reasons"]}
                           for c in checks if not c["valid"]],
    }


def _real_points_lookup(bank_path: str = os.path.join(config.TEMPLATE_DIR, "bank.json"), window: float = 2.5):
    """Hàm (category, r) -> median số điểm của object THẬT cùng category ở khoảng cách r ± window (m).

    Dùng template bank (điểm của object thật trong nuScenes). None nếu không có bank hoặc < 5 mẫu.
    """
    if not os.path.isfile(bank_path):
        return lambda cat, r: None
    with open(bank_path, encoding="utf-8") as f:
        bank = json.load(f)
    by_cat = {}
    for m in bank:
        by_cat.setdefault(m["category"], []).append((m["distance"], m["num_points"]))
    by_cat = {c: np.array(v) for c, v in by_cat.items()}

    def lookup(cat, r):
        v = by_cat.get(cat)
        if v is None:
            return None
        near = v[np.abs(v[:, 0] - r) <= window, 1]
        return float(np.median(near)) if len(near) >= 5 else None
    return lookup


def insertion_quality(objects) -> dict:
    """Chỉ số vật lý của object chèn (cần trường quality do run_insertion ghi).

    facing_ok_rate          tỉ lệ object có mặt chứa điểm quay về sensor (facing_cos > 0)
    ground_supported_rate   tỉ lệ object đặt ở chỗ LiDAR thấy mặt đất (ground_support >= min_support)
    density_ratio_median    median (số điểm object chèn / median số điểm object thật cùng lớp, cùng khoảng cách);
                            ~1 là đúng, >> 1 là quá dày (lỗi bản cũ)
    visible_ratio_mean      điểm còn thấy sau che khuất / điểm template
    """
    lookup = _real_points_lookup()
    facing = np.array([q["facing_cos"] for _, q in objects], dtype=float)
    support = np.array([q["ground_support"] for _, q in objects])
    ratios = []
    for a, q in objects:
        real = lookup(a["category"], q["distance"])
        if real:
            ratios.append(a["num_lidar_pts"] / real)
    visible = [a["num_lidar_pts"] / q["n_template"] for a, q in objects if q.get("n_template")]
    return {
        "n_objects": len(objects),
        "facing_ok_rate": float(np.nanmean(facing > 0)),
        "ground_supported_rate": float(np.mean(support >= config.GROUND["min_support"])),
        "density_ratio_median": float(np.median(ratios)) if ratios else None,
        "density_ratio_n": len(ratios),
        "visible_ratio_mean": float(np.mean(visible)) if visible else None,
    }


def evaluate_weather(original_dir: str, augmented_dir: str) -> dict:
    """So sánh từng cặp <name>.bin giữa original_dir và augmented_dir.

    Returns dict: point_count_change (mean tỉ lệ, mean tuyệt đối), intensity_stats (mean/std trước, sau).
    """
    pairs = [(os.path.join(original_dir, os.path.basename(p)), p)
             for p in sorted(glob.glob(os.path.join(augmented_dir, "*.bin")))]
    pairs = [(o, a) for o, a in pairs if os.path.isfile(o)]
    if not pairs:
        raise FileNotFoundError(f"Không có cặp .bin khớp tên giữa {original_dir} và {augmented_dir}")
    ratios, diffs, i_orig, i_aug = [], [], [], []
    bands = {f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}m+": (lo, hi) for lo, hi in RANGE_BANDS}
    n_orig, n_real, n_noise = (dict.fromkeys(bands, 0) for _ in range(3))
    for o, a in pairs:
        po, pa = load_bin(o), load_bin(a)
        ratios.append(len(pa) / len(po))
        diffs.append(len(pa) - len(po))
        i_orig.append(po[:, 3])
        i_aug.append(pa[:, 3])
        lab_path = a[:-4] + "_labels.npy"  # 0 = điểm gốc còn lại, 1 = nhiễu / backscatter (run_weather.py)
        lab = np.load(lab_path) if os.path.isfile(lab_path) else np.zeros(len(pa), np.int8)
        r_o, r_a = np.hypot(po[:, 0], po[:, 1]), np.hypot(pa[:, 0], pa[:, 1])
        for k, (lo, hi) in bands.items():
            n_orig[k] += int(((r_o >= lo) & (r_o < hi)).sum())
            in_a = (r_a >= lo) & (r_a < hi)
            n_real[k] += int((in_a & (lab == 0)).sum())
            n_noise[k] += int((in_a & (lab == 1)).sum())
    i_orig, i_aug = np.concatenate(i_orig), np.concatenate(i_aug)
    return {
        "n_scenes": len(pairs),
        "point_count_change": {"mean_ratio": float(np.mean(ratios)), "mean_diff": float(np.mean(diffs))},
        "intensity_stats": {
            "original": {"mean": float(i_orig.mean()), "std": float(i_orig.std())},
            "augmented": {"mean": float(i_aug.mean()), "std": float(i_aug.std())},
        },
        # theo dải khoảng cách BEV: kept = điểm gốc còn lại / điểm gốc; noise_share = điểm nhiễu / mọi điểm trong dải
        "by_range": {k: {"kept_ratio": n_real[k] / n_orig[k] if n_orig[k] else None,
                         "noise_share": n_noise[k] / (n_real[k] + n_noise[k]) if n_real[k] + n_noise[k] else None}
                     for k in bands},
    }


RANGE_BANDS = ((0.0, 10.0), (10.0, 30.0), (30.0, np.inf))


def run_insertion_batch(n_samples: int, num_objects: int = 3, physical: bool = True,
                        out_dir: str = config.AUGMENTED_DIR, categories=("vehicle",), weather=None,
                        seed: int = config.SEED) -> dict:
    """Chạy Task A trên n_samples sample rải đều toàn dataset để có dữ liệu đánh giá.

    Template lấy từ TemplateBank, loại scene đích (đề bài: chèn vật thể vào scene khác) và xe 2 bánh
    không người lái. weather = (loại, level), vd ("rain", "heavy"): áp thời tiết lên scene sau khi chèn.
    Ghi <out_dir>/batch_timing.json: wall-clock từng bước (lấy template, chèn, thời tiết, ghi file) và
    scenes_per_hour toàn batch. Không tính load nuScenes + bank (làm 1 lần, ghi riêng setup_s).
    """
    from tqdm import tqdm

    from task_a.extract_object import TemplateBank
    from task_a.run_insertion import run_insertion, save_result
    from task_b.run_weather import apply_weather
    from utils.nuscenes_loader import load_nuscenes

    if _result_files(out_dir):
        raise FileExistsError(f"{out_dir} đã có kết quả cũ, chọn --augmented-dir khác để số đo không bị trộn")
    t = time.perf_counter()
    nusc = load_nuscenes()
    bank = TemplateBank().load_or_build(nusc)
    setup_s = time.perf_counter() - t

    rng = np.random.default_rng(seed)
    idx = np.unique(np.linspace(0, len(nusc.sample) - 1, min(n_samples, len(nusc.sample))).round().astype(int))
    stages = dict.fromkeys(("template_s", "insert_s", "weather_s", "save_s"), 0.0)
    t_batch = time.perf_counter()
    for i in tqdm(idx, desc="insertion"):
        s = nusc.sample[i]
        t = time.perf_counter()
        templates = bank.sample(num_objects, rng, categories=list(categories), min_points=50,
                                exclude_scenes=[s["scene_token"]],
                                required_attributes=config.TEMPLATE_REQUIRED_ATTRIBUTES)
        t, stages["template_s"] = time.perf_counter(), stages["template_s"] + time.perf_counter() - t
        res = run_insertion(nusc, s["token"], templates, rng=rng, physical=physical)
        t, stages["insert_s"] = time.perf_counter(), stages["insert_s"] + time.perf_counter() - t
        if weather:
            boxes = [b for b in res["existing_annotations"] + res["annotations_sensor"]
                     if b["category"].startswith("vehicle")]
            res["points"], _ = apply_weather(res["points"], weather[0], weather[1], rng, vehicle_boxes=boxes)
            res["weather"] = f"{weather[0]}_{weather[1]}"
            t, stages["weather_s"] = time.perf_counter(), stages["weather_s"] + time.perf_counter() - t
        save_result(res, out_dir)
        stages["save_s"] += time.perf_counter() - t
    total = time.perf_counter() - t_batch

    n = len(idx)
    timing = {
        "n_scenes": n, "num_objects": num_objects, "categories": list(categories),
        "weather": f"{weather[0]}_{weather[1]}" if weather else None,
        "setup_s": setup_s, "total_s": total, **stages,
        "scenes_per_hour": n / (total / 3600),
        "scenes_per_hour_insertion_only": n / ((stages["template_s"] + stages["insert_s"]) / 3600),
        "scenes_per_hour_weather_only": n / (stages["weather_s"] / 3600) if stages["weather_s"] else None,
        "machine": {"cpu": platform.processor(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
                    "os": platform.platform()},
    }
    with open(os.path.join(out_dir, TIMING_NAME), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=1, ensure_ascii=False)
    return timing


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-insertion", type=int, default=0, metavar="N", help="chạy insertion trên N sample trước")
    ap.add_argument("--naive", action="store_true", help="--run-insertion bằng pipeline cũ (để so sánh)")
    ap.add_argument("--num-objects", type=int, default=3, help="số object yêu cầu mỗi scene (--run-insertion)")
    ap.add_argument("--categories", nargs="*", default=["vehicle"], help="prefix category template (--run-insertion)")
    ap.add_argument("--weather", nargs=2, metavar=("TYPE", "LEVEL"),
                    help="áp thời tiết sau khi chèn (--run-insertion), vd: rain heavy, fog dense, storm storm")
    ap.add_argument("--augmented-dir", default=config.AUGMENTED_DIR)
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "eval_report.json"))
    args = ap.parse_args()

    if args.run_insertion:
        run_insertion_batch(args.run_insertion, args.num_objects, physical=not args.naive,
                            out_dir=args.augmented_dir, categories=args.categories, weather=args.weather)

    report = {}
    try:
        report["insertion"] = evaluate_insertion(args.augmented_dir)
        from utils.nuscenes_loader import load_nuscenes
        report["validity"] = evaluate_validity(load_nuscenes(), args.augmented_dir)
    except FileNotFoundError as e:
        print("[insertion]", e)
    orig = os.path.join(config.WEATHER_DIR, "original")
    for d in sorted(glob.glob(os.path.join(config.WEATHER_DIR, "*_*"))):
        try:
            report[f"weather/{os.path.basename(d)}"] = evaluate_weather(orig, d)
        except FileNotFoundError as e:
            print("[weather]", e)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("saved", out)
