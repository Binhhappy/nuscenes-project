"""Kiểm chứng mô hình mặt đường ướt (task_b/storm.py, wet_road_mask) bằng dữ liệu thật nuScenes mini.

nuScenes mini không có cảnh đang mưa; có scene-1094 "Night, after rain" (đường ướt). So với scene-1077, scene-1100
(đêm, đường khô), cùng khu singapore-hollandvillage, cùng cảm biến HDL-32E.

Chỉ số theo dải khoảng cách BEV (RANGE_BANDS), tính từng keyframe:
    retention  số điểm mặt đường trong dải / số điểm mặt đường ở dải tham chiếu 5-10 m (bớt ảnh hưởng hình dạng đường)
    intensity  cường độ trung bình của điểm mặt đường trong dải
Điểm mặt đường: |khoảng cách tới mặt phẳng RANSAC| < GROUND_BAND (task_a/ground_plane.estimate_ground_ransac).
Lý thuyết: đường ướt phản xạ gương -> tia chiếu xiên ở xa không quay về -> retention giảm nhanh hơn, intensity thấp hơn.
So thêm: áp wet_road_mask lên cảnh khô, xem đường cong có tiến gần cảnh ướt thật không.
CI 95% = bootstrap theo khung (1000 lần). Hạn chế: 1 cảnh ướt, đường khác nhau giữa các cảnh, không biết độ ướt.

Chạy: python tools/wet_road_check.py   -> output/report/wet_road_check.png + .json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

WET = ["scene-1094"]
DRY = ["scene-1077", "scene-1100"]
RANGE_BANDS = [(5, 10), (10, 20), (20, 30), (30, 50)]
GROUND_BAND = 0.2   # (m)
N_BOOT = 1000


def scene_samples(nusc, name):
    sc = next(s for s in nusc.scene if s["name"] == name)
    tok, out = sc["first_sample_token"], []
    while tok:
        out.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return out


def ground_profile(points: np.ndarray, rng) -> dict:
    """retention, intensity theo RANGE_BANDS cho 1 point cloud."""
    from task_a.ground_plane import estimate_ground_ransac
    a, b, c, d = estimate_ground_ransac(points, rng=rng)
    dist = np.abs(points[:, :3] @ np.array([a, b, c]) + d)
    g = points[dist < GROUND_BAND]
    r = np.hypot(g[:, 0], g[:, 1])
    counts = np.array([((r >= lo) & (r < hi)).sum() for lo, hi in RANGE_BANDS], float)
    inten = np.array([g[(r >= lo) & (r < hi), 3].mean() if ((r >= lo) & (r < hi)).any() else np.nan
                      for lo, hi in RANGE_BANDS])
    return {"retention": counts / max(counts[0], 1), "intensity": inten}


def boot_ci(x: np.ndarray, rng) -> tuple:
    """mean, CI 95% bootstrap theo hàng (khung), bỏ NaN."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(x, (N_BOOT, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from task_b.storm import wet_road_mask
    from utils.nuscenes_loader import get_pointcloud, load_nuscenes

    nusc = load_nuscenes()
    rng = np.random.default_rng(config.SEED)
    groups = {"wet (real, scene-1094)": [], "dry (real, 1077+1100)": [], "dry + wet_road_mask (sim)": []}
    for names, key in ((WET, "wet (real, scene-1094)"), (DRY, "dry (real, 1077+1100)")):
        for name in names:
            for tok in scene_samples(nusc, name):
                pts = get_pointcloud(nusc, tok)
                groups[key].append(ground_profile(pts, np.random.default_rng(0)))
                if key.startswith("dry"):
                    sim = pts[wet_road_mask(pts, rng)]
                    groups["dry + wet_road_mask (sim)"].append(ground_profile(sim, np.random.default_rng(0)))

    labels = [f"{lo}-{hi} m" for lo, hi in RANGE_BANDS]
    report = {"bands": labels, "n_frames": {k: len(v) for k, v in groups.items()}, "groups": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    for (k, prof), col in zip(groups.items(), ("#2563eb", "#9ca3af", "#f59e0b")):
        report["groups"][k] = {}
        for ax, metric in zip(axes, ("retention", "intensity")):
            arr = np.array([p[metric] for p in prof])
            stats = [boot_ci(arr[:, j], rng) for j in range(len(labels))]
            report["groups"][k][metric] = [{"mean": m, "ci95": [lo, hi]} for m, lo, hi in stats]
            m = np.array([s[0] for s in stats])
            err = np.array([[s[0] - s[1], s[2] - s[0]] for s in stats]).T
            ax.errorbar(x, m, yerr=err, marker="o", capsize=4, color=col, label=k)
    axes[0].set(title="Điểm mặt đường còn lại (chuẩn hoá theo 5-10 m)", xticks=x, xticklabels=labels, ylabel="tỉ lệ")
    axes[1].set(title="Cường độ trung bình điểm mặt đường", xticks=x, xticklabels=labels, ylabel="intensity")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Kiểm chứng mặt đường ướt: nuScenes mini, đêm, singapore-hollandvillage (CI 95% bootstrap theo khung)")
    fig.tight_layout()
    out_dir = os.path.join(config.OUTPUT_DIR, "report")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "wet_road_check.png"), dpi=120)
    with open(os.path.join(out_dir, "wet_road_check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    for k, v in report["groups"].items():
        print(k, f"(n={report['n_frames'][k]})")
        for metric in ("retention", "intensity"):
            print("  ", metric, [f"{s['mean']:.2f} [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]" for s in v[metric]])
    print("saved", os.path.join(out_dir, "wet_road_check.png"))


if __name__ == "__main__":
    main()
