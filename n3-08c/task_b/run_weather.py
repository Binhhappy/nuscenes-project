"""Pipeline Task B: input scene -> chọn weather type + intensity -> output.

Weather: rain | fog | wet_road | snow | rain+wet_road (đường ướt trước, rồi mưa).

Output (config.WEATHER_DIR):
    original/<name>.bin               bản gốc (để evaluate_weather so sánh)
    <weather>_<level>/<name>.bin      point cloud sau mô phỏng
    <weather>_<level>/<name>_labels.npy  nhãn điểm: config.LABEL_ORIGINAL / LABEL_NOISE / LABEL_PUDDLE

Chạy:
    python task_b/run_weather.py                                  # sample[0], rain moderate
    python task_b/run_weather.py --weather fog --level dense
    python task_b/run_weather.py --weather rain+wet_road --level heavy
    python task_b/run_weather.py --input output/augmented/<tok>.bin --weather rain --level heavy
    python task_b/run_weather.py --all-levels                     # mọi weather x level + ảnh so sánh
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_b.fog import simulate_fog  # noqa: E402
from task_b.rain import simulate_rain  # noqa: E402
from task_b.snow import simulate_snow  # noqa: E402
from task_b.wet_road import simulate_puddles  # noqa: E402
from utils.nuscenes_loader import get_pointcloud, load_bin, load_nuscenes, save_bin  # noqa: E402

# Weather kết hợp: các bước áp lần lượt, cùng 1 level. Đường ướt trước vì nó cần point cloud chưa
# nhiễu để tìm mặt đường; mưa sau đó dropout / thêm noise trên kết quả.
COMBOS = {"rain+wet_road": ("wet_road", "rain")}
# Thứ tự rain, fog đứng đầu để kết quả --all-levels của 2 loại này không đổi so với trước.
LEVELS = {"rain": list(config.RAIN_LEVELS), "fog": list(config.FOG_VISIBILITY),
          "wet_road": list(config.WET_ROAD_LEVELS), "snow": list(config.SNOW_LEVELS),
          "rain+wet_road": [lv for lv in config.WET_ROAD_LEVELS if lv in config.RAIN_LEVELS]}
_CHAINABLE = {"rain": simulate_rain, "wet_road": simulate_puddles, "snow": simulate_snow}


def apply_weather(points: np.ndarray, weather: str, level: str,
                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Áp weather (một key của LEVELS) với level -> (points, labels)."""
    if weather not in LEVELS:
        raise ValueError(f"weather phải thuộc {list(LEVELS)}, nhận {weather!r}")
    if level not in LEVELS[weather]:
        raise ValueError(f"level của {weather} phải thuộc {LEVELS[weather]}, nhận {level!r}")
    if weather == "fog":
        return simulate_fog(points, config.FOG_VISIBILITY[level], rng, return_labels=True)
    labels = None
    for step in COMBOS.get(weather, (weather,)):  # nhãn của bước trước được truyền sang bước sau
        points, labels = _CHAINABLE[step](points, level, rng, return_labels=True, labels=labels)
    return points, labels
    # TODO: kết hợp với fog (simulate_fog chưa nhận labels đầu vào).


def label_counts(labels: np.ndarray) -> dict:
    """Số điểm theo tên nhãn (config.LABEL_NAMES)."""
    return {name: int((labels == v).sum()) for v, name in config.LABEL_NAMES.items()}


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
    ap.add_argument("--weather", choices=list(LEVELS), default="rain")
    ap.add_argument("--level", default="moderate",
                    help="light/moderate/heavy; riêng fog: light/moderate/dense")
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
        n = label_counts(res["labels"])
        print(f"{weather}_{level:9s}: {len(pts)} -> {len(res['points'])} pts "
              f"(noise {n['noise']}, puddle {n['puddle']}) -> {res['dir']}")
        grid.append({"title": f"{weather} {level}", "points": res["points"], "labels": res["labels"]})

    import matplotlib.pyplot as plt

    from utils.visualization import label_colors, plot_bev, show_open3d
    ncols = min(len(grid), 4)
    nrows = -(-len(grid) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for ax in axes.ravel()[len(grid):]:
        ax.axis("off")
    for ax, g in zip(axes.ravel(), grid):
        plot_bev(g["points"], colors=label_colors(g["labels"]), title=g["title"], ax=ax)
    png = os.path.join(config.VIS_DIR, f"weather_{name}.png")
    os.makedirs(config.VIS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print("saved", png)
    if args.vis:
        show_open3d(grid[-1]["points"], colors=label_colors(grid[-1]["labels"]), window_name=grid[-1]["title"])
