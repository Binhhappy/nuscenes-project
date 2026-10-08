"""Đánh giá: Task A (valid insertion rate, throughput, số object) và Task B (thay đổi point count, intensity).

Chạy:
    python evaluate.py                       # đánh giá kết quả có sẵn trong output/
    python evaluate.py --run-insertion 20    # chạy insertion trên 20 sample trước rồi đánh giá
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

    Returns dict: valid_rate (inserted / requested), total_objects, scenes_per_hour, n_scenes, per_category.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"Không có kết quả insertion trong {results_dir}")
    requested = inserted = 0
    elapsed = 0.0
    per_cat = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        requested += r["n_requested"]
        inserted += r["n_inserted"]
        elapsed += r["elapsed_s"]
        for a in r["annotations_sensor"]:
            per_cat[a["category"]] = per_cat.get(a["category"], 0) + 1
    # TODO: thêm lý do reject (off_road / collision) khi placement ghi lại thống kê.
    # TODO: chất lượng — so sánh phân bố points/box của object chèn với object thật cùng khoảng cách.
    return {
        "n_scenes": len(files),
        "valid_rate": inserted / requested if requested else 0.0,
        "total_objects": inserted,
        "scenes_per_hour": len(files) / (elapsed / 3600) if elapsed > 0 else float("inf"),
        "per_category": per_cat,
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


def run_insertion_batch(n_samples: int, num_objects: int = 3) -> None:
    """Chạy Task A trên n_samples sample đầu (source = sample kế tiếp) để có dữ liệu đánh giá."""
    from tqdm import tqdm

    from task_a.run_insertion import collect_templates, run_insertion, save_result
    from utils.nuscenes_loader import load_nuscenes

    nusc = load_nuscenes()
    rng = np.random.default_rng(config.SEED)
    samples = nusc.sample[:n_samples]
    for i, s in enumerate(tqdm(samples, desc="insertion")):
        source = nusc.sample[(i + 1) % len(nusc.sample)]["token"]
        templates = collect_templates(nusc, source, None, num_objects)
        if templates:
            save_result(run_insertion(nusc, s["token"], templates, rng=rng))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-insertion", type=int, default=0, metavar="N", help="chạy insertion trên N sample trước")
    args = ap.parse_args()

    if args.run_insertion:
        run_insertion_batch(args.run_insertion)

    report = {}
    try:
        report["insertion"] = evaluate_insertion()
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
