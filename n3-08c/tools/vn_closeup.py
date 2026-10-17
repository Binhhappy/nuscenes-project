"""Ảnh cận cảnh điểm LiDAR của vật thể VN đã chèn, cạnh object thật nuScenes để so (nhìn từ trên + nhìn ngang).

Lấy từ dữ liệu đã sinh (output/vn/seed0/<kịch bản>/), mỗi loại 1 object có nhiều điểm nhất trong khoảng 8-15 m.
Điểm trong box object (màu cam) = điểm LiDAR của vật thể; điểm xám = cảnh xung quanh. Hệ toạ độ object
(x hướng đầu, z lên). Chạy (Windows / WSL, không cần GPU):
    python tools/vn_closeup.py
Output: output/report/vn_closeup.png
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.merge import points_in_new_box  # noqa: E402
from utils.nuscenes_loader import get_annotations, get_pointcloud, load_bin, load_nuscenes  # noqa: E402
from utils.transforms import to_local_frame  # noqa: E402

ROOT = os.path.join(config.OUTPUT_DIR, "vn", "seed0")
OBJ, CTX, INK, INK2, SURFACE = "#eb6834", "#c9c8c3", "#0b0b0b", "#52514e", "#fcfcfb"
ITEMS = [  # (nhãn, kịch bản, category)
    ("Xe máy chở 2-3 người", "multi_rider", "vn.moto_multi"),
    ("Xe máy chở hàng cồng kềnh", "cargo_moto", "vn.moto_cargo"),
    ("Xe ba gác", "ba_gac_slow", "vn.ba_gac"),
    ("Xe máy kéo thùng hàng", "tow_moto", "vn.moto_tow"),
    ("Xe hàng rong", "vendor_curb", "vn.vendor_cart"),
    ("Chó", "dog_dart", "vn.dog"),
    ("Trâu / bò", "cow_road", "vn.cow"),
    ("Ô tô (ray-cast)", "rc_control_car", "vn.car"),
]


def pick(scenario, category, rmin=8.0, rmax=15.0):
    """(points, annotation) object có nhiều điểm nhất ở rmin-rmax m."""
    best = None
    for f in glob.glob(os.path.join(ROOT, scenario, "*.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        for a in r["annotations_sensor"]:
            d = np.hypot(*a["translation"][:2])
            if a["category"] == category and rmin <= d <= rmax and (best is None or a["num_lidar_pts"] > best[1]["num_lidar_pts"]):
                best = (f[:-5] + ".bin", a)
    return (load_bin(best[0]), best[1]) if best else (None, None)


def pick_real(nusc, category, rmin=8.0, rmax=15.0, rider=False):
    """Object thật nuScenes (cả mini: mini_val không có xe máy có người lái) nhiều điểm nhất ở rmin-rmax m, để so."""
    best = None
    for tok in [s["token"] for s in nusc.sample][::3]:
        for a in get_annotations(nusc, tok):
            ann = nusc.get("sample_annotation", a["token"])
            attrs = [nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]]
            if a["category"] != category or (rider and "cycle.with_rider" not in attrs):
                continue
            d = np.hypot(*a["translation"][:2])
            if rmin <= d <= rmax and (best is None or ann["num_lidar_pts"] > best[2]):
                best = (tok, a, ann["num_lidar_pts"])
    return (get_pointcloud(nusc, best[0]), best[1]) if best else (None, None)


def draw(ax_top, ax_side, points, ann, title):
    w, l, h = ann["size"]
    margin = 1.0
    loc = to_local_frame(points[:, :3], ann["translation"], ann["rotation"])
    near = (np.abs(loc[:, 0]) < l / 2 + margin) & (np.abs(loc[:, 1]) < w / 2 + margin) & (np.abs(loc[:, 2]) < h / 2 + 0.6)
    inside = points_in_new_box(points, ann) & near
    ctx = near & ~inside
    for ax, (i, j), (a, b) in ((ax_top, (0, 1), (l, w)), (ax_side, (0, 2), (l, h))):
        ax.set_facecolor(SURFACE)
        ax.scatter(loc[ctx, i], loc[ctx, j], s=3, c=CTX, linewidths=0)
        ax.scatter(loc[inside, i], loc[inside, j], s=7, c=OBJ, linewidths=0)
        ax.plot([-a / 2, a / 2, a / 2, -a / 2, -a / 2], [-b / 2, -b / 2, b / 2, b / 2, -b / 2], color=INK2, lw=0.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e6e5e1")
    d = np.hypot(*ann["translation"][:2])
    ax_top.set_title(f"{title}\n{int(inside.sum())} điểm, cách {d:.0f} m", fontsize=8.5, color=INK, loc="left")
    ax_top.set_ylabel("từ trên", fontsize=7, color=INK2)
    ax_side.set_ylabel("nhìn ngang", fontsize=7, color=INK2)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nusc = load_nuscenes()
    panels = [("Xe máy THẬT nuScenes", *pick_real(nusc, "vehicle.motorcycle", rider=True)),
              ("Ô tô THẬT nuScenes", *pick_real(nusc, "vehicle.car"))]
    panels += [(lab, *pick(sc, cat)) for lab, sc, cat in ITEMS]
    panels = [p for p in panels if p[1] is not None]
    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(2 * rows, cols, figsize=(3.4 * cols, 4.4 * rows), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.3, 1] * rows})
    for ax in axes.ravel():
        ax.axis("off")
    for k, (title, pts, ann) in enumerate(panels):
        r, c = divmod(k, cols)
        for ax in (axes[2 * r, c], axes[2 * r + 1, c]):
            ax.axis("on")
        draw(axes[2 * r, c], axes[2 * r + 1, c], pts, ann, title)
    fig.suptitle("Điểm LiDAR của vật thể (cam) trong cảnh (xám) — object thật nuScenes vs vật thể VN ray-cast, "
                 "hệ toạ độ object (đầu xe bên phải)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "report", "vn_closeup.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print("->", out)


if __name__ == "__main__":
    main()
