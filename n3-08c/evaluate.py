"""Đánh giá: Task A (valid insertion rate, throughput, số object, chất lượng vật lý) và Task B (thay đổi point count, intensity).

Chạy:
    python evaluate.py                       # đánh giá kết quả có sẵn trong output/
    python evaluate.py --run-insertion 20    # chạy insertion trên 20 sample trước rồi đánh giá
    python evaluate.py --run-insertion 20 --naive --augmented-dir ../output/augmented_naive   # bản cũ để so
"""
import glob
import json
import os
import sys
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
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
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
    return {
        "n_scenes": len(files),
        "valid_rate": inserted / requested if requested else 0.0,
        "total_objects": inserted,
        "scenes_per_hour": len(files) / (elapsed / 3600) if elapsed > 0 else float("inf"),
        "per_category": per_cat,
        "reject_stats": rejects,
        "quality": insertion_quality(objects) if objects else None,
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
    for o, a in pairs:
        po, pa = load_bin(o), load_bin(a)
        ratios.append(len(pa) / len(po))
        diffs.append(len(pa) - len(po))
        i_orig.append(po[:, 3])
        i_aug.append(pa[:, 3])
    i_orig, i_aug = np.concatenate(i_orig), np.concatenate(i_aug)
    # TODO: thống kê theo dải khoảng cách (0-10, 10-30, 30+ m) và tỉ lệ noise từ *_labels.npy.
    return {
        "n_scenes": len(pairs),
        "point_count_change": {"mean_ratio": float(np.mean(ratios)), "mean_diff": float(np.mean(diffs))},
        "intensity_stats": {
            "original": {"mean": float(i_orig.mean()), "std": float(i_orig.std())},
            "augmented": {"mean": float(i_aug.mean()), "std": float(i_aug.std())},
        },
    }


def run_insertion_batch(n_samples: int, num_objects: int = 3, physical: bool = True,
                        out_dir: str = config.AUGMENTED_DIR) -> None:
    """Chạy Task A trên n_samples sample đầu (source = sample kế tiếp) để có dữ liệu đánh giá."""
    from tqdm import tqdm

    from task_a.run_insertion import collect_templates, run_insertion, save_result
    from utils.nuscenes_loader import load_nuscenes

    nusc = load_nuscenes()
    rng = np.random.default_rng(config.SEED)
    samples = nusc.sample[:n_samples]
    for i, s in enumerate(tqdm(samples, desc="insertion")):
        source = nusc.sample[(i + 1) % len(nusc.sample)]["token"]
        templates = collect_templates(nusc, source, None, num_objects, rng=rng)
        if templates:
            save_result(run_insertion(nusc, s["token"], templates, rng=rng, physical=physical), out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-insertion", type=int, default=0, metavar="N", help="chạy insertion trên N sample trước")
    ap.add_argument("--naive", action="store_true", help="--run-insertion bằng pipeline cũ (để so sánh)")
    ap.add_argument("--augmented-dir", default=config.AUGMENTED_DIR)
    args = ap.parse_args()

    if args.run_insertion:
        run_insertion_batch(args.run_insertion, physical=not args.naive, out_dir=args.augmented_dir)

    report = {}
    try:
        report["insertion"] = evaluate_insertion(args.augmented_dir)
    except FileNotFoundError as e:
        print("[insertion]", e)
    orig = os.path.join(config.WEATHER_DIR, "original")
    for d in sorted(glob.glob(os.path.join(config.WEATHER_DIR, "*_*"))):
        try:
            report[f"weather/{os.path.basename(d)}"] = evaluate_weather(orig, d)
        except FileNotFoundError as e:
            print("[weather]", e)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("saved", out)
