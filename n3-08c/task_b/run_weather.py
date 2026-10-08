"""Pipeline Task B: input scene -> chọn weather type + intensity -> output.

Output (config.WEATHER_DIR):
    original/<name>.bin               bản gốc (để evaluate_weather so sánh)
    <weather>_<level>/<name>.bin      point cloud sau mô phỏng
    <weather>_<level>/<name>_labels.npy  0 = điểm gốc, 1 = noise/backscatter

Chạy:
    python task_b/run_weather.py                                  # sample[0], rain moderate
    python task_b/run_weather.py --weather fog --level dense
    python task_b/run_weather.py --input output/augmented/<tok>.bin --weather rain --level heavy
    python task_b/run_weather.py --all-levels                     # mọi weather x level + ảnh so sánh
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_b.fog import simulate_fog  # noqa: E402
from task_b.rain import simulate_rain  # noqa: E402
from utils.nuscenes_loader import get_pointcloud, load_bin, load_nuscenes, save_bin  # noqa: E402

LEVELS = {"rain": list(config.RAIN_LEVELS), "fog": list(config.FOG_VISIBILITY)}


def apply_weather(points: np.ndarray, weather: str, level: str,
                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Áp weather ('rain' | 'fog') với level -> (points, labels)."""
    if weather == "rain":
        return simulate_rain(points, level, rng, return_labels=True)
    if weather == "fog":
        return simulate_fog(points, config.FOG_VISIBILITY[level], rng, return_labels=True)
    raise ValueError(f"weather phải là 'rain' hoặc 'fog', nhận {weather!r}")
    # TODO: thêm snow (nhiễu dày gần sensor + dropout), kết hợp rain + fog.


def run_weather(points: np.ndarray, name: str, weather: str, level: str,
                rng: Optional[np.random.Generator] = None, out_dir: str = config.WEATHER_DIR) -> dict:
    """Áp weather và lưu kết quả + bản gốc."""
    out, labels = apply_weather(points, weather, level, rng)
    save_bin(os.path.join(out_dir, "original", f"{name}.bin"), points)
    sub = os.path.join(out_dir, f"{weather}_{level}")
    save_bin(os.path.join(sub, f"{name}.bin"), out)
    np.save(os.path.join(sub, f"{name}_labels.npy"), labels)
    return {"points": out, "labels": labels, "dir": sub}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="sample token (mặc định sample[0])")
    ap.add_argument("--input", help="file .bin (vd output của Task A) thay cho --sample")
    ap.add_argument("--weather", choices=["rain", "fog"], default="rain")
    ap.add_argument("--level", default="moderate", help="rain: light/moderate/heavy; fog: light/moderate/dense")
    ap.add_argument("--all-levels", action="store_true")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--vis", action="store_true", help="mở Open3D cho kết quả")
    args = ap.parse_args()

    if args.input:
        pts, name = load_bin(args.input), Path(args.input).stem
    else:
        nusc = load_nuscenes()
        name = args.sample or nusc.sample[0]["token"]
        pts = get_pointcloud(nusc, name)

    rng = np.random.default_rng(args.seed)
    jobs = [(w, lv) for w in LEVELS for lv in LEVELS[w]] if args.all_levels else [(args.weather, args.level)]
    grid = [{"title": "original", "points": pts, "labels": np.zeros(len(pts))}]
    for weather, level in jobs:
        res = run_weather(pts, name, weather, level, rng)
        print(f"{weather}_{level:9s}: {len(pts)} -> {len(res['points'])} pts "
              f"(noise {int(res['labels'].sum())}) -> {res['dir']}")
        grid.append({"title": f"{weather} {level}", "points": res["points"], "labels": res["labels"]})

    import matplotlib.pyplot as plt

    from utils.visualization import label_colors, plot_bev, show_open3d
    fig, axes = plt.subplots(1, len(grid), figsize=(5 * len(grid), 5), squeeze=False)
    for ax, g in zip(axes[0], grid):
        plot_bev(g["points"], colors=label_colors(g["labels"]), title=g["title"], ax=ax)
    png = os.path.join(config.VIS_DIR, f"weather_{name}.png")
    os.makedirs(config.VIS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print("saved", png)
    if args.vis:
        show_open3d(grid[-1]["points"], colors=label_colors(grid[-1]["labels"]), window_name=grid[-1]["title"])
