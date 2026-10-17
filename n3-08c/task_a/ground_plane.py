"""Ground estimation (RANSAC + độ cao cục bộ) + vùng đặt object từ nuScenes map expansion.

Mọi thứ ở LIDAR_TOP sensor frame. Plane: a*x + b*y + c*z + d = 0, pháp tuyến hướng lên (c > 0).

Vì sao cần GroundModel ngoài plane: đường thực tế không phẳng. Đo trên đáy box xe thật nhìn thấy
được (v1.0-mini, 555 xe, r < 50 m), sai số |z_ground - z_bottom|:
    plane RANSAC toàn cục    median 0.10 m, p90 0.42 m (15-30 m: p90 0.53 m)
    GroundModel (mặc định)   median 0.10 m, p90 0.31 m (15-30 m: p90 0.38 m)
Ở > 30 m cả hai đều kém (p90 > 1 m) do điểm quá thưa.

Chạy demo: python task_a/ground_plane.py  (lưu ảnh BEV vùng đặt object vào config.VIS_DIR)
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.nuscenes_loader import get_calibration, get_ego_pose, get_lidar_token, get_map_name  # noqa: E402
from utils.transforms import sensor_to_global_matrix  # noqa: E402

Plane = Tuple[float, float, float, float]
_G = config.GROUND


def estimate_ground_ransac(points: np.ndarray, z_threshold: float = _G["z_threshold"],
                           dist_thresh: float = 0.15, n_iters: int = 200, max_tilt_deg: float = 10.0,
                           rng: Optional[np.random.Generator] = None) -> Plane:
    """RANSAC plane trên các điểm có z < z_threshold (LIDAR_TOP cao ~1.84 m -> ground ~ -1.84).

    Returns (a, b, c, d) đã chuẩn hoá, c > 0.
    """
    rng = rng or np.random.default_rng(config.SEED)
    cand = points[points[:, 2] < z_threshold, :3]
    if len(cand) < 3:
        raise ValueError("Không đủ điểm dưới z_threshold để fit ground")
    min_cos = np.cos(np.radians(max_tilt_deg))

    best_inliers, best_count = None, 0
    for _ in range(n_iters):
        p0, p1, p2 = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-6 or abs(n[2]) / norm < min_cos:  # suy biến hoặc quá nghiêng
            continue
        n /= norm
        inliers = np.abs(cand @ n - n @ p0) < dist_thresh
        if inliers.sum() > best_count:
            best_inliers, best_count = inliers, inliers.sum()
    if best_inliers is None:
        raise RuntimeError("RANSAC không tìm được plane")

    # Refine: least squares (SVD) trên inliers
    inl = cand[best_inliers]
    centroid = inl.mean(0)
    n = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]  # full_matrices=True dựng U (N x N): rất chậm
    if n[2] < 0:
        n = -n
    return float(n[0]), float(n[1]), float(n[2]), float(-n @ centroid)


def plane_z(x, y, plane_coeffs: Plane):
    """z của plane tại (x, y), nhận scalar hoặc array."""
    a, b, c, d = plane_coeffs
    return -(a * np.asarray(x) + b * np.asarray(y) + d) / c


def select_ground_candidates(points: np.ndarray, plane_coeffs: Plane, band: float = _G["band"],
                             cell: float = _G["cell"], cell_tol: float = _G["cell_tol"]) -> np.ndarray:
    """Điểm mặt đất (K, 3): gần plane (|dist| < band, đủ rộng cho dốc nhẹ) VÀ thấp nhất trong
    cell BEV của nó (z <= z_min_cell + cell_tol), loại thân xe / bụi cây nằm trong band."""
    n, d = np.asarray(plane_coeffs[:3]), plane_coeffs[3]
    p = points[np.abs(points[:, :3] @ n + d) < band, :3]
    if len(p) == 0:
        return p
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.ravel()
    z_min = np.full(inv.max() + 1, np.inf)
    np.minimum.at(z_min, inv, p[:, 2])
    return p[p[:, 2] <= z_min[inv] + cell_tol]


@dataclass
class GroundModel:
    """Plane toàn cục + median z cục bộ của ground candidates (KD-tree)."""
    plane: Plane
    ground_points: np.ndarray  # (K, 3)
    radius: float = _G["radius"]
    min_support: int = _G["min_support"]

    def __post_init__(self):
        self._tree = cKDTree(self.ground_points[:, :2]) if len(self.ground_points) else None

    @classmethod
    def from_points(cls, points: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs) -> "GroundModel":
        plane = estimate_ground_ransac(points, rng=rng)
        sel = {k: kwargs.pop(k) for k in ("band", "cell", "cell_tol") if k in kwargs}
        return cls(plane, select_ground_candidates(points, plane, **sel), **kwargs)

    def height_at(self, x, y, radius: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(z, support): z mặt đất tại (x, y) và số ground point trong bán kính.

        support < min_support -> z lấy từ plane (vùng bị che / ngoài tầm, ước lượng kém tin cậy).
        """
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        z = plane_z(x, y, self.plane)
        support = np.zeros(len(x), dtype=int)
        if self._tree is not None:
            for i, idx in enumerate(self._tree.query_ball_point(np.c_[x, y], radius or self.radius)):
                support[i] = len(idx)
                if len(idx) >= self.min_support:
                    z[i] = np.median(self.ground_points[idx, 2])
        return z, support

    def is_supported(self, x: float, y: float, radius: Optional[float] = None) -> bool:
        """True nếu LiDAR thấy mặt đất quanh (x, y): object đặt ở đây sẽ không nằm trong vùng bị che."""
        return bool(self.height_at(x, y, radius)[1][0] >= self.min_support)


class ConsensusGround(GroundModel):
    """Đồng thuận nhiều GroundModel dựng với các seed RANSAC khác nhau.

    Ở vùng xa / bị che, ground candidate nằm sát mép band quanh plane: plane lệch nhẹ theo seed là vùng đó lúc có
    lúc không có mặt đất, hoặc độ cao lệch 0.6-0.9 m (đo bằng evaluate.validate_scene). Ở đây một vị trí chỉ được
    coi là thấy mặt đất khi MỌI model đều thấy và độ cao các model lệch nhau <= max_spread.
    Seed cố định, không dùng rng của caller: chuỗi ngẫu nhiên của placement không đổi khi đổi số model.
    """

    def __init__(self, models: Sequence[GroundModel], max_spread: float = 0.15):
        self.models = list(models)
        self.max_spread = max_spread
        first = self.models[0]
        self.plane, self.ground_points = first.plane, first.ground_points
        self.radius, self.min_support = first.radius, first.min_support
        self._tree = first._tree

    @classmethod
    def from_points(cls, points: np.ndarray, seeds: Sequence[int] = (1, 2, 3), max_spread: float = 0.15,
                    **kwargs) -> "ConsensusGround":
        return cls([GroundModel.from_points(points, rng=np.random.default_rng(s), **kwargs) for s in seeds],
                   max_spread)

    def height_at(self, x, y, radius: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(z, support): z = median các model, support = min các model; lệch > max_spread thì support = 0."""
        zs, sups = zip(*(m.height_at(x, y, radius) for m in self.models))
        zs, sups = np.stack(zs), np.stack(sups)
        support = sups.min(0)
        support[zs.max(0) - zs.min(0) > self.max_spread] = 0
        return np.median(zs, 0), support


def project_z_to_ground(x: float, y: float, plane_coeffs: Union[Plane, GroundModel]) -> float:
    """z của mặt đất tại (x, y). plane_coeffs: (a, b, c, d) hoặc GroundModel (ưu tiên, chính xác hơn)."""
    if isinstance(plane_coeffs, GroundModel):
        return float(plane_coeffs.height_at(x, y)[0][0])
    return float(plane_z(x, y, plane_coeffs))


# ===================== Map: vùng hợp lệ để đặt object =====================

_MAP_CACHE: Dict[tuple, object] = {}


def load_map_for_sample(nusc, sample_token: str):
    """NuScenesMap của sample (cache theo map name; load map mất ~1-2 s)."""
    from nuscenes.map_expansion.map_api import NuScenesMap

    key = (nusc.dataroot, get_map_name(nusc, sample_token))
    if key not in _MAP_CACHE:
        _MAP_CACHE[key] = NuScenesMap(dataroot=nusc.dataroot, map_name=key[1])
    return _MAP_CACHE[key]


def layers_for_category(category: str) -> Tuple[str, ...]:
    """Map layers hợp lệ cho category (prefix dài nhất khớp trong config.PLACEMENT_LAYERS)."""
    hits = [p for p in config.PLACEMENT_LAYERS if category.startswith(p)]
    return config.PLACEMENT_LAYERS[max(hits, key=len)] if hits else config.DEFAULT_LAYERS


def _to_multipolygon(geom) -> MultiPolygon:
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom]) if not geom.is_empty else MultiPolygon()
    return MultiPolygon([g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty])


def get_map_area(nusc, nusc_map, sample_token: str, layers: Sequence[str] = ("drivable_area",),
                 radius: float = config.MAX_DISTANCE) -> MultiPolygon:
    """Hợp các polygon của layers quanh ego (bán kính radius), chuyển về LIDAR_TOP sensor frame."""
    sd_token = get_lidar_token(nusc, sample_token)
    T = sensor_to_global_matrix(get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token))
    ex, ey = T[0, 3], T[1, 3]

    records = nusc_map.get_records_in_patch((ex - radius, ey - radius, ex + radius, ey + radius),
                                            list(layers), mode="intersect")
    polys = []
    for layer in layers:
        for tok in records[layer]:
            rec = nusc_map.get(layer, tok)
            # drivable_area có nhiều polygon ('polygon_tokens'); walkway, ped_crossing... có 1 ('polygon_token')
            for pt in rec.get("polygon_tokens") or [rec["polygon_token"]]:
                p = nusc_map.extract_polygon(pt)
                if not p.is_empty:
                    polys.append(p.buffer(0))
    if not polys:
        return MultiPolygon()
    area = unary_union(polys).intersection(Point(ex, ey).buffer(radius))

    # Global (map 2D, z = 0) -> sensor: x' = G00 x + G01 y + G03, y' = G10 x + G11 y + G13
    G = np.linalg.inv(T)
    return _to_multipolygon(affine_transform(area, [G[0, 0], G[0, 1], G[1, 0], G[1, 1], G[0, 3], G[1, 3]]))


def get_drivable_area(nusc, nusc_map, sample_token: str, radius: float = config.MAX_DISTANCE) -> MultiPolygon:
    """Drivable area quanh ego (bán kính radius) ở LIDAR_TOP sensor frame."""
    return get_map_area(nusc, nusc_map, sample_token, ("drivable_area",), radius)


if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.nuscenes_loader import get_annotations, get_pointcloud, load_nuscenes
    from utils.visualization import plot_bev

    nusc = load_nuscenes()
    tok = nusc.sample[0]["token"]
    pts = get_pointcloud(nusc, tok)

    ground = GroundModel.from_points(pts)
    a, b, c, d = ground.plane
    print(f"plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0; z(0,0)={project_z_to_ground(0, 0, ground.plane):.2f} m")
    print(f"ground candidates: {len(ground.ground_points)} / {len(pts)}")
    for x, y in ((10, 0), (25, 5), (-30, 10)):
        z, n = ground.height_at(x, y)
        print(f"  ground z({x:4d},{y:4d}) = {z[0]:+.2f} m (plane {project_z_to_ground(x, y, ground.plane):+.2f}), support {n[0]}")

    nusc_map = load_map_for_sample(nusc, tok)
    anns = get_annotations(nusc, tok)
    drivable = get_drivable_area(nusc, nusc_map, tok)
    walk = get_map_area(nusc, nusc_map, tok, layers_for_category("human.pedestrian.adult"))
    print(f"drivable area: {len(drivable.geoms)} polygons, {drivable.area:.0f} m^2; "
          f"walkway+ped_crossing: {walk.area:.0f} m^2")

    # Ảnh: drivable | walkway | độ lệch ground cục bộ so với plane
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    plot_bev(pts, anns, region=drivable, title="drivable_area", ax=axes[0])
    plot_bev(pts, anns, region=walk, title="walkway + ped_crossing", ax=axes[1])
    gx, gy = np.meshgrid(np.arange(-50, 51, 2.0), np.arange(-50, 51, 2.0))
    z, n = ground.height_at(gx.ravel(), gy.ravel())
    dz = np.where(n >= ground.min_support, z - plane_z(gx.ravel(), gy.ravel(), ground.plane), np.nan)
    im = axes[2].scatter(gx.ravel(), gy.ravel(), c=dz, cmap="coolwarm", vmin=-0.5, vmax=0.5, s=18, marker="s")
    axes[2].set(aspect="equal", title="local ground - plane (m); trống = không có support")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    out = os.path.join(config.VIS_DIR, f"ground_{tok}.png")
    os.makedirs(config.VIS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)
