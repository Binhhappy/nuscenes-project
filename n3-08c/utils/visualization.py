"""Visualize point cloud: Open3D (3D) + matplotlib (BEV).

Box = annotation dict (translation, size [w,l,h], rotation [w,x,y,z]) ở cùng frame với points.
Chạy demo: python utils/visualization.py  (lưu ảnh BEV vào config.VIS_DIR)
"""
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuscenes.utils.data_classes import Box  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

# Cạnh nối 8 góc theo thứ tự Box.corners() của nuScenes
_BOX_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def to_box(ann: dict) -> Box:
    return Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))


def intensity_colors(points: np.ndarray) -> np.ndarray:
    i = points[:, 3]
    i = (i - i.min()) / (i.max() - i.min() + 1e-8)
    return np.stack([i, i * 0.5, 1 - i], axis=1)


def label_colors(labels: np.ndarray) -> np.ndarray:
    """labels 0 = điểm gốc (xám), !=0 = điểm thêm vào (đỏ)."""
    colors = np.tile([0.55, 0.55, 0.55], (len(labels), 1))
    colors[np.asarray(labels) != 0] = [0.9, 0.1, 0.1]
    return colors


def show_open3d(points: np.ndarray, boxes: Sequence[dict] = (), new_boxes: Sequence[dict] = (),
                colors: Optional[np.ndarray] = None, window_name: str = "nuScenes LiDAR") -> None:
    """Cửa sổ Open3D: points + box gốc (xanh lá) + box mới (đỏ)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(intensity_colors(points) if colors is None else colors)
    geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)]
    for anns, color in ((boxes, (0.1, 0.8, 0.1)), (new_boxes, (1.0, 0.1, 0.1))):
        for ann in anns:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(to_box(ann).corners().T.astype(np.float64))
            ls.lines = o3d.utility.Vector2iVector(_BOX_EDGES)
            ls.colors = o3d.utility.Vector3dVector([color] * len(_BOX_EDGES))
            geoms.append(ls)
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def plot_bev(points: np.ndarray, boxes: Sequence[dict] = (), new_boxes: Sequence[dict] = (),
             region=None, colors: Optional[np.ndarray] = None, xlim: float = 50.0, title: str = "",
             ax=None, save_path: Optional[str] = None, show: bool = False):
    """BEV: points + box gốc (xanh lá) + box mới (đỏ) + region shapely (vd drivable area)."""
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 9))
    if region is not None and not region.is_empty:
        for poly in getattr(region, "geoms", [region]):
            ax.fill(*poly.exterior.xy, color=(0.85, 0.9, 1.0), zorder=0)
            for hole in poly.interiors:
                ax.fill(*hole.xy, color="white", zorder=0)

    m = (np.abs(points[:, 0]) < xlim) & (np.abs(points[:, 1]) < xlim)
    c = points[m, 2] if colors is None else np.asarray(colors)[m]
    ax.scatter(points[m, 0], points[m, 1], s=0.3, c=c, cmap="viridis" if colors is None else None,
               linewidths=0)
    for anns, color, lw in ((boxes, "limegreen", 1.2), (new_boxes, "red", 1.8)):
        for ann in anns:
            b = to_box(ann)
            xy = b.bottom_corners()[:2]
            ax.plot(np.append(xy[0], xy[0, 0]), np.append(xy[1], xy[1, 0]), color=color, lw=lw)
            head = b.center[:2] + b.rotation_matrix[:2, 0] * b.wlh[1] / 2
            ax.plot([b.center[0], head[0]], [b.center[1], head[1]], color=color, lw=lw)
    ax.plot(0, 0, marker="^", color="black", ms=8)
    ax.set(xlim=(-xlim, xlim), ylim=(-xlim, xlim), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax.set_title(title, fontsize=9)

    if own_fig:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
    return ax


if __name__ == "__main__":
    import argparse

    import config
    from utils.nuscenes_loader import get_annotations, get_pointcloud, load_nuscenes

    ap = argparse.ArgumentParser()
    ap.add_argument("--open3d", action="store_true", help="mở thêm cửa sổ Open3D")
    args = ap.parse_args()

    nusc = load_nuscenes()
    tok = nusc.sample[0]["token"]
    pts, anns = get_pointcloud(nusc, tok), get_annotations(nusc, tok)
    out = os.path.join(config.VIS_DIR, f"bev_{tok}.png")
    plot_bev(pts, anns, title=tok, save_path=out)
    print("Saved", out)
    if args.open3d:
        show_open3d(pts, anns)
