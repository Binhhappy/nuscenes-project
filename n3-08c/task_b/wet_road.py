"""Mô phỏng mặt đường ướt và vũng nước (wet road / puddles) trên LiDAR point cloud.

Tia LiDAR chiếu xuống vũng nước bị phản xạ gương (specular), tín hiệu không quay về sensor
-> mảng mất điểm trên mặt đường; điểm còn lại trong vũng có intensity thấp.

3 bước: (1) tìm điểm mặt đường (GroundModel) -> (2) sinh vũng ellipse trên mặt đường
-> (3) dropout + giảm intensity điểm mặt đường trong vũng. Không sinh thêm điểm.
Chạy demo: python task_b/wet_road.py
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import GroundModel, estimate_ground_ransac, select_ground_candidates  # noqa: E402

# estimate_ground_ransac refine bằng SVD đầy đủ trên mọi inlier (ma trận U cỡ K x K): 20 000 inlier mất ~8 s.
# Fit plane trên mẫu con là đủ chính xác; ground candidates vẫn lấy từ toàn bộ points.
MAX_PLANE_FIT_POINTS = 3000


def ground_mask(points: np.ndarray, tol: float = config.WET_ROAD_GROUND_TOL) -> np.ndarray:
    """Mask (N,) điểm mặt đường: z <= z mặt đất cục bộ (GroundModel) + tol.

    Độ cao mặt đất lấy theo cell BEV (config.GROUND["cell"]) nên bám được đường dốc.
    Fallback khi không fit được plane (quá ít điểm thấp): z < config.GROUND["z_threshold"].
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    fit_rng = np.random.default_rng(config.SEED)  # không dùng rng của caller: mặt đường không đổi theo seed
    sub = points
    if len(points) > MAX_PLANE_FIT_POINTS:
        sub = points[fit_rng.choice(len(points), MAX_PLANE_FIT_POINTS, replace=False)]
    try:
        plane = estimate_ground_ransac(sub, rng=fit_rng)
        ground = GroundModel(plane, select_ground_candidates(points, plane))
    except (ValueError, RuntimeError):
        return points[:, 2] < config.GROUND["z_threshold"]

    n, d = np.asarray(ground.plane[:3]), ground.plane[3]
    near = np.abs(points[:, :3] @ n + d) < config.GROUND["band"]
    mask = np.zeros(len(points), dtype=bool)
    if not near.any():
        return mask
    cell = config.GROUND["cell"]
    ij = np.floor(points[near, :2] / cell).astype(np.int64)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    z_cell, _ = ground.height_at((uniq[:, 0] + 0.5) * cell, (uniq[:, 1] + 0.5) * cell)
    mask[near] = points[near, 2] <= z_cell[inv.ravel()] + tol
    return mask


def sample_puddles(ground_xy: np.ndarray, level: str = "moderate",
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sinh vũng nước (K, 4) [cx, cy, rx, ry], bán trục đều trong [min_radius, max_radius].

    Tâm lấy đều theo DIỆN TÍCH mặt đường LiDAR thấy được (cell BEV có điểm mặt đường), không theo
    số điểm: điểm LiDAR dày đặc quanh xe, lấy theo điểm thì mọi vũng dồn về sát sensor.
    """
    rng = rng or np.random.default_rng()
    p = config.WET_ROAD_LEVELS[level]
    k = p["num_puddles"] if len(ground_xy) else 0
    if k:
        cell = config.GROUND["cell"]
        cells = np.unique(np.floor(np.asarray(ground_xy, float) / cell), axis=0)
        centers = (cells[rng.integers(len(cells), size=k)] + rng.random((k, 2))) * cell
    else:
        centers = np.zeros((0, 2))
    radii = rng.uniform(p["min_radius"], p["max_radius"], (k, 2))
    return np.c_[centers, radii].astype(np.float64)


def in_puddles(xy: np.ndarray, puddles: np.ndarray) -> np.ndarray:
    """Mask (N,) điểm (x, y) nằm trong ít nhất 1 vũng ellipse."""
    mask = np.zeros(len(xy), dtype=bool)
    for cx, cy, rx, ry in puddles:
        mask |= ((xy[:, 0] - cx) / rx) ** 2 + ((xy[:, 1] - cy) / ry) ** 2 <= 1.0
    return mask


def simulate_puddles(points: np.ndarray, level: str = "moderate",
                     rng: Optional[np.random.Generator] = None, return_labels: bool = False,
                     labels: Optional[np.ndarray] = None, puddles: Optional[np.ndarray] = None):
    """Đường ướt theo level trong config.WET_ROAD_LEVELS: light / moderate / heavy.

    labels: nhãn đầu vào (N,) của points (khi nối sau weather khác); mặc định toàn LABEL_ORIGINAL.
    puddles: (K, 4) [cx, cy, rx, ry] chỉ định sẵn thay cho sinh ngẫu nhiên.
    Returns points (K, 5); nếu return_labels thì thêm labels (K,): điểm gốc còn sống trong vũng
    -> LABEL_PUDDLE, còn lại giữ nhãn đầu vào. Điểm ngoài vũng và điểm không thuộc mặt đường giữ nguyên.
    """
    if level not in config.WET_ROAD_LEVELS:
        raise ValueError(f"level phải thuộc {list(config.WET_ROAD_LEVELS)}")
    rng = rng or np.random.default_rng()
    p = config.WET_ROAD_LEVELS[level]
    labels = (np.full(len(points), config.LABEL_ORIGINAL, np.int8) if labels is None
              else np.asarray(labels, np.int8))

    # (1) Điểm mặt đường
    on_ground = ground_mask(points)
    # (2) Vũng nước
    if puddles is None:
        puddles = sample_puddles(points[on_ground, :2], level, rng)
    wet = on_ground & in_puddles(points[:, :2], np.asarray(puddles, float).reshape(-1, 4))
    # (3) Phản xạ gương: phần lớn điểm trong vũng không quay về sensor
    dropped = wet & (rng.random(len(points)) < p["dropout"])
    keep = ~dropped
    survived = (wet & keep)[keep]

    out = points[keep].copy()
    out[survived, 3] *= p["intensity_scale"]  # điểm còn lại: tín hiệu yếu
    if return_labels:
        out_labels = labels[keep].copy()
        out_labels[survived & (out_labels == config.LABEL_ORIGINAL)] = config.LABEL_PUDDLE
        return out, out_labels
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    n_ground, n_obj = 40000, 10000
    xy = rng.uniform(-40, 40, (n_ground + n_obj, 2))
    z = np.r_[-1.8 + rng.normal(0, 0.02, n_ground), -1.8 + rng.uniform(0.3, 1.8, n_obj)]  # đường + vật thể
    pts = np.c_[xy, z, rng.uniform(50, 255, len(xy)), rng.integers(0, 32, len(xy))].astype(np.float32)

    print(f"điểm mặt đường: {ground_mask(pts).sum()} / {len(pts)}")
    for level in config.WET_ROAD_LEVELS:
        out, lab = simulate_puddles(pts, level, rng, return_labels=True)
        print(f"{level:9s}: {len(pts)} -> {len(out)} pts (còn trong vũng {(lab == config.LABEL_PUDDLE).sum()}), "
              f"mean intensity {pts[:, 3].mean():.1f} -> {out[:, 3].mean():.1f}")
