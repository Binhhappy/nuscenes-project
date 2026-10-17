"""Kiểm tra bằng mắt từng vật thể chèn: ảnh cận cảnh, đặt cạnh xe thật cùng lớp ở khoảng cách gần nhất.

Mỗi vật thể chèn là 1 hàng, 3 ô:
    1. Nhìn từ trên (sensor frame), vùng quanh box: điểm chèn đỏ, điểm scene xám, mũi tên = hướng về LiDAR.
       Đúng vật lý: điểm đỏ nằm ở mặt quay về mũi tên; vùng sau xe (bóng che) không còn điểm scene.
    2. Nhìn ngang dọc tia LiDAR (trục ngang = khoảng cách tới sensor, trục dọc = z), chỉ lấy điểm trong
       góc phương vị của box: đáy box phải chạm dải điểm mặt đất; phía sau box là bóng che.
    3. Xe thật cùng lớp trong scene đích ở khoảng cách gần nhất, cùng cách vẽ ô 1: so số điểm và mặt thấy.

Chạy:
    python tools/inspect_insertion.py                                     # file đầu tiên trong output/augmented
    python tools/inspect_insertion.py --token <sample_token>
    python tools/inspect_insertion.py --aug-dir <dir_vật_lý> --compare <dir_naive> --token <tok>
        # 2 khối hàng: "naive" ở trên, "vật lý" ở dưới (2 dir chạy cùng seed bằng run_insertion --naive)
Output: <OUTPUT_DIR>/vis/inspect_<token>.png (hoặc --out).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import GroundModel  # noqa: E402
from utils.nuscenes_loader import get_annotations, get_pointcloud, load_bin, load_nuscenes  # noqa: E402
from utils.transforms import points_in_cuboid, quaternion_to_yaw, to_local_frame, wrap_angle  # noqa: E402
from utils.visualization import to_box  # noqa: E402

MARGIN = 3.0  # m quanh box ở ô nhìn từ trên


def draw_box_xy(ax, ann, color):
    b = to_box(ann)
    xy = b.bottom_corners()[:2]
    ax.plot(np.append(xy[0], xy[0, 0]), np.append(xy[1], xy[1, 0]), color=color, lw=1.5)
    head = b.center[:2] + b.rotation_matrix[:2, 0] * b.wlh[1] / 2
    ax.plot([b.center[0], head[0]], [b.center[1], head[1]], color=color, lw=1.5)


def facing(points, ann):
    """cos(tâm điểm, hướng về sensor) trong object frame; > 0 là mặt quay về LiDAR."""
    if len(points) == 0:
        return float("nan")
    loc = to_local_frame(points[:, :3], ann["translation"], ann["rotation"])[:, :2].mean(0)
    sensor = to_local_frame(np.zeros((1, 3)), ann["translation"], ann["rotation"])[0, :2]
    return float(loc @ sensor / (np.linalg.norm(loc) * np.linalg.norm(sensor) + 1e-9))


def plot_top(ax, points, ann, color, title):
    x, y = ann["translation"][:2]
    half = max(ann["size"][:2]) / 2 + MARGIN
    near = (np.abs(points[:, 0] - x) < half) & (np.abs(points[:, 1] - y) < half)
    inside = points_in_cuboid(points, ann["translation"], ann["size"], ann["rotation"])
    ax.scatter(points[near & ~inside, 0], points[near & ~inside, 1], s=2, c="0.6", linewidths=0)
    ax.scatter(points[inside, 0], points[inside, 1], s=6, c=color, linewidths=0)
    draw_box_xy(ax, ann, color)
    r = np.hypot(x, y)
    ax.annotate("", xy=(x - x / r * half * 0.9, y - y / r * half * 0.9), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set(xlim=(x - half, x + half), ylim=(y - half, y + half), aspect="equal")
    ax.set_title(title, fontsize=8)
    return int(inside.sum()), facing(points[inside], ann)


def plot_side(ax, points, ann, ground_z, color):
    x, y, z = ann["translation"]
    w, l, h = ann["size"]
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    half_az = np.arctan2(max(w, l) / 2, r) + np.radians(1.0)
    az = wrap_angle(np.arctan2(points[:, 1], points[:, 0]) - phi)
    rng = np.hypot(points[:, 0], points[:, 1])
    sector = (np.abs(az) < half_az) & (np.abs(rng - r) < l / 2 + 15)
    inside = points_in_cuboid(points, ann["translation"], ann["size"], ann["rotation"])
    ax.scatter(rng[sector & ~inside], points[sector & ~inside, 2], s=2, c="0.6", linewidths=0)
    ax.scatter(rng[inside], points[inside, 2], s=6, c=color, linewidths=0)
    # bề dày box theo tia: chiếu 4 góc đáy lên hướng tia
    corners = to_box(ann).bottom_corners()[:2].T
    proj = corners @ np.array([np.cos(phi), np.sin(phi)])
    ax.add_patch(__import__("matplotlib").patches.Rectangle((proj.min(), z - h / 2), np.ptp(proj), h,
                                                              fill=False, ec=color, lw=1.5))
    ax.axhline(ground_z, color="green", ls="--", lw=1)
    ax.set(xlabel="khoảng cách tới LiDAR (m)", ylabel="z (m)")
    ax.set_title(f"nhìn ngang dọc tia; đáy box {z - h / 2 - ground_z:+.2f} m so với đất", fontsize=8)


def nearest_real(real_anns, ann):
    r = np.hypot(*ann["translation"][:2])
    same = [a for a in real_anns if a["category"] == ann["category"]]
    if not same:
        return None
    return min(same, key=lambda a: abs(np.hypot(*a["translation"][:2]) - r))


def inspect(nusc, token, blocks, out_path):
    """blocks: list (nhãn, aug_dir). Mỗi block vẽ mọi vật thể chèn trong <aug_dir>/<token>.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig = get_pointcloud(nusc, token)
    real = [a for a in get_annotations(nusc, token)
            if points_in_cuboid(orig, a["translation"], a["size"], a["rotation"]).sum() > 0]
    ground = GroundModel.from_points(orig, rng=np.random.default_rng(42))

    rows = []
    for label, d in blocks:
        meta = json.load(open(os.path.join(d, f"{token}.json"), encoding="utf-8"))
        aug = load_bin(os.path.join(d, f"{token}.bin"))
        rows += [(label, aug, a) for a in meta["annotations_sensor"]]
    if not rows:
        raise SystemExit(f"{token}: không có vật thể chèn nào")

    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.2 * len(rows)), squeeze=False)
    report = []
    for (label, aug, ann), ax in zip(rows, axes):
        r = np.hypot(*ann["translation"][:2])
        view = np.degrees(wrap_angle(np.arctan2(ann["translation"][1], ann["translation"][0])
                                     - quaternion_to_yaw(ann["rotation"])))
        gz = float(ground.height_at(*ann["translation"][:2])[0][0])
        n, cos = plot_top(ax[0], aug, ann, "red", "")
        ax[0].set_title(f"[{label}] {ann['category']}  r={r:.1f} m\n{n} điểm, facing cos={cos:+.2f}, "
                        f"view={view:.0f}°", fontsize=8)
        plot_side(ax[1], aug, ann, gz, "red")
        ref = nearest_real(real, ann)
        row = {"block": label, "category": ann["category"], "r": round(r, 1), "n_pts": n,
               "facing_cos": round(cos, 2), "bottom_minus_ground": round(ann["translation"][2]
                                                                         - ann["size"][2] / 2 - gz, 2)}
        if ref is None:
            ax[2].set_axis_off()
            ax[2].set_title("không có xe thật cùng lớp trong scene", fontsize=8)
        else:
            rr = np.hypot(*ref["translation"][:2])
            rn, rcos = plot_top(ax[2], orig, ref, "limegreen", "")
            ax[2].set_title(f"xe thật cùng lớp gần khoảng cách nhất: r={rr:.1f} m\n{rn} điểm, "
                            f"facing cos={rcos:+.2f}", fontsize=8)
            row.update(ref_r=round(rr, 1), ref_n_pts=rn)
        report.append(row)
    fig.suptitle(f"{token}  (mũi tên = hướng về LiDAR, đường xanh đứt = mặt đất ước lượng)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", default=config.AUGMENTED_DIR, help="output của run_insertion (bản vật lý)")
    ap.add_argument("--compare", default=None, help="dir thứ 2 để so (vd output của run_insertion --naive)")
    ap.add_argument("--token", default=None, help="sample token; mặc định file .json đầu tiên trong --aug-dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    token = args.token or Path(sorted(glob.glob(os.path.join(args.aug_dir, "*.json")))[0]).stem
    blocks = ([("naive", args.compare)] if args.compare else []) + [("vật lý", args.aug_dir)]
    out = args.out or os.path.join(config.OUTPUT_DIR, "vis", f"inspect_{token}.png")
    for row in inspect(load_nuscenes(), token, blocks, out):
        print(row)
    print("saved", out)
