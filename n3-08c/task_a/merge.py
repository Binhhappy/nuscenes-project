"""Merge: transform template tới vị trí mới, concat vào target point cloud, tạo annotation mới.

insert_object áp các luật vật lý của LiDAR (mỗi tia chỉ có 1 điểm trả về, điểm gần nhất):
    1. match_density      object dời ra xa hơn khoảng cách gốc -> subsample theo (r_cũ / r_mới)^2
    2. assign_rings       gán lại ring theo góc ngước mới (bảng góc ngước lấy từ chính target)
    3. points_in_new_box  xoá điểm target nằm trong box mới (trên mặt đất)
    4. resolve_occlusion  range image (ring x azimuth): object che điểm phía sau nó,
                          vật gần sensor hơn che phần object phía sau. Object gần (< OCCLUSION_NEAR_RANGE)
                          che cả ring kề trên/dưới, vì ring gán theo góc ngước có thể lệch ±1 ở cự ly gần

Chạy demo: python task_a/merge.py
"""
import sys
import uuid
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import GroundModel, Plane, project_z_to_ground  # noqa: E402
from utils.transforms import from_local_frame, points_in_cuboid, yaw_to_quaternion  # noqa: E402


def elevation(points: np.ndarray) -> np.ndarray:
    """Góc ngước (rad) của từng điểm nhìn từ sensor."""
    return np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))


def ring_elevations(points: np.ndarray, n_rings: int = config.LIDAR_RINGS) -> np.ndarray:
    """Góc ngước median của từng ring trong point cloud (n_rings,). Ring thiếu điểm thì nội suy."""
    ring, el = points[:, 4].astype(int), elevation(points)
    table = np.full(n_rings, np.nan)
    for r in range(n_rings):
        m = ring == r
        if m.any():
            table[r] = np.median(el[m])
    ok = ~np.isnan(table)
    if not ok.any():
        raise ValueError("Point cloud không có cột ring hợp lệ")
    return np.interp(np.arange(n_rings), np.flatnonzero(ok), table[ok])


def assign_rings(points: np.ndarray, ring_table: np.ndarray) -> np.ndarray:
    """Ring gần nhất theo góc ngước (ring_table tăng dần theo index)."""
    el = elevation(points)
    idx = np.clip(np.searchsorted(ring_table, el), 1, len(ring_table) - 1)
    lower_closer = np.abs(el - ring_table[idx - 1]) <= np.abs(el - ring_table[idx])
    return np.where(lower_closer, idx - 1, idx)


def match_density(points: np.ndarray, r_source: float, r_target: float,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Mật độ điểm trên vật ~ 1/r^2. Dời ra xa: giữ ngẫu nhiên tỉ lệ (r_source / r_target)^2.

    Dời lại gần: giữ nguyên (không thể tự sinh điểm thật), nên placement chỉ cho lệch r nhỏ.
    """
    if r_target <= r_source or len(points) == 0:
        return points
    rng = rng or np.random.default_rng()
    return points[rng.random(len(points)) < (r_source / r_target) ** 2]


def points_in_new_box(points: np.ndarray, box: dict,
                      ground_margin: float = config.TEMPLATE_GROUND_MARGIN) -> np.ndarray:
    """Mask điểm nằm trong box, trừ dải ground_margin sát đáy (giữ mặt đường dưới gầm xe)."""
    w, l, h = box["size"]
    lifted = dict(box, translation=[*box["translation"][:2], box["translation"][2] + ground_margin / 2])
    return points_in_cuboid(points, lifted["translation"], [w, l, h - ground_margin], box["rotation"])


def range_image_cells(points: np.ndarray, n_az: int = config.LIDAR_AZIMUTH_BINS,
                      n_rings: int = config.LIDAR_RINGS) -> np.ndarray:
    """Chỉ số ô (ring, azimuth) của từng điểm: mỗi ô ~ 1 tia laser."""
    az = np.arctan2(points[:, 1], points[:, 0])
    col = ((az + np.pi) / (2 * np.pi) * n_az).astype(np.int64) % n_az
    return np.clip(points[:, 4].astype(np.int64), 0, n_rings - 1) * n_az + col


def resolve_occlusion(points: np.ndarray, is_object: np.ndarray, eps: float = config.OCCLUSION_EPS,
                      n_az: int = config.LIDAR_AZIMUTH_BINS,
                      near_range: float = config.OCCLUSION_NEAR_RANGE) -> np.ndarray:
    """Mask điểm còn thấy được sau khi chèn object.

    Trong mỗi ô range image: điểm target bị xoá nếu có điểm object gần sensor hơn quá eps (bị object che),
    điểm object bị xoá nếu có điểm target gần hơn quá eps (object bị vật thật che).
    Ô không có điểm object giữ nguyên.

    Điểm object cách sensor < near_range (BEV) che thêm ô ring ± 1 cùng azimuth: ring của object là ước
    lượng (assign_rings), dưới 10 m có thể lệch 1 ring do gốc phát tia lệch khỏi gốc sensor frame. Không nới
    thì điểm nền phía sau lọt qua xe ở ring kề. Chiều ngược lại (vật thật che object) giữ đúng 1 ô.
    """
    cell = range_image_cells(points, n_az)
    r = np.linalg.norm(points[:, :3], axis=1)
    n_cells = config.LIDAR_RINGS * n_az
    obj_min = np.full(n_cells, np.inf)
    tgt_min = np.full(n_cells, np.inf)
    np.minimum.at(obj_min, cell[is_object], r[is_object])
    np.minimum.at(tgt_min, cell[~is_object], r[~is_object])
    near = is_object & (np.hypot(points[:, 0], points[:, 1]) < near_range)
    ring, col = cell[near] // n_az, cell[near] % n_az
    for d in (-1, 1):
        ok = (ring + d >= 0) & (ring + d < config.LIDAR_RINGS)
        np.minimum.at(obj_min, (ring[ok] + d) * n_az + col[ok], r[near][ok])
    other_min = np.where(is_object, tgt_min[cell], obj_min[cell])
    return r <= other_min + eps


def insert_object(target_points: np.ndarray, object_template: dict, position: Tuple[float, float, float],
                  ground_plane: Union[Plane, GroundModel], rng: Optional[np.random.Generator] = None,
                  physical: bool = True) -> Tuple[np.ndarray, dict]:
    """Đặt template tại position = (x, y, yaw), đáy box chạm mặt đất.

    ground_plane: plane (a, b, c, d) hoặc GroundModel (độ cao cục bộ, chính xác hơn).
    physical=False: chỉ concat như bản cũ (để so sánh).
    Returns (augmented points, info). info: n_visible (điểm object còn lại), n_template,
    n_density_dropped, n_box_cleared (điểm target trong box bị xoá), n_target_occluded, n_object_occluded.
    """
    x, y, yaw = position
    h = object_template["size"][2]
    center = [x, y, project_z_to_ground(x, y, ground_plane) + h / 2]
    rotation = yaw_to_quaternion(yaw)
    placed = from_local_frame(object_template["points"], center, rotation).astype(target_points.dtype)
    info = {"n_template": int(len(placed)), "n_density_dropped": 0, "n_box_cleared": 0,
            "n_target_occluded": 0, "n_object_occluded": 0}
    if not physical:
        info["n_visible"] = int(len(placed))
        return np.concatenate([target_points, placed], axis=0), info

    r_source = object_template.get("distance")
    if r_source is not None:
        kept = match_density(placed, r_source, float(np.hypot(x, y)), rng)
        info["n_density_dropped"] = int(len(placed) - len(kept))
        placed = kept
    placed[:, 4] = assign_rings(placed, ring_elevations(target_points))

    in_box = points_in_new_box(target_points, {"translation": center, "size": object_template["size"],
                                               "rotation": rotation})
    target = target_points[~in_box]
    info["n_box_cleared"] = int(in_box.sum())

    merged = np.concatenate([target, placed], axis=0)
    is_obj = np.r_[np.zeros(len(target), bool), np.ones(len(placed), bool)]
    keep = resolve_occlusion(merged, is_obj)
    info["n_target_occluded"] = int((~keep & ~is_obj).sum())
    info["n_object_occluded"] = int((~keep & is_obj).sum())
    info["n_visible"] = int((keep & is_obj).sum())
    # TODO: (tuỳ chọn) chỉnh intensity theo khoảng cách mới; hiện r lệch ít nên giữ nguyên.
    return merged[keep], info


def create_annotation(object_template: dict, position: Tuple[float, float, float], ground_z: float,
                      num_lidar_pts: Optional[int] = None) -> dict:
    """Annotation dict (sensor frame) cho object vừa chèn. num_lidar_pts: số điểm còn thấy sau che khuất."""
    x, y, yaw = position
    size: Sequence[float] = object_template["size"]
    return {
        "token": uuid.uuid4().hex,
        "translation": [float(x), float(y), float(ground_z + size[2] / 2)],
        "size": [float(v) for v in size],
        "rotation": yaw_to_quaternion(yaw),
        "category": object_template["category"],
        "num_lidar_pts": int(len(object_template["points"]) if num_lidar_pts is None else num_lidar_pts),
        "synthetic": True,
        "source_ann_token": object_template.get("ann_token"),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Target: mặt đất z = -1.8 + một bức tường ở x = 20 m; ring giả theo góc ngước
    ground = np.c_[rng.uniform(2, 30, (6000, 1)), rng.uniform(-10, 10, (6000, 1)), np.full((6000, 1), -1.8)]
    wall = np.c_[np.full((2000, 1), 20.0), rng.uniform(-10, 10, (2000, 1)), rng.uniform(-1.8, 2.0, (2000, 1))]
    xyz = np.r_[ground, wall]
    table = np.radians(np.linspace(-30.67, 10.67, config.LIDAR_RINGS))
    target = np.c_[xyz, rng.uniform(0, 255, len(xyz)), assign_rings(np.c_[xyz, np.zeros((len(xyz), 2))], table)]
    # Template = mặt đuôi xe (local x = -l/2) dày điểm, là mặt LiDAR thấy khi xe đứng trước sensor
    face = np.c_[np.full(3000, -2.25), rng.uniform(-0.9, 0.9, 3000), rng.uniform(-0.7, 0.8, 3000)]
    template = {
        "points": np.c_[face, rng.uniform(0, 255, (3000, 2))],
        "size": [1.9, 4.5, 1.6], "category": "vehicle.car", "ann_token": "demo", "distance": 10.0,
    }
    plane = (0.0, 0.0, 1.0, 1.8)  # z = -1.8
    pos = (10.0, 0.0, 0.0)

    def wall_behind(p):  # tường nằm trong góc nhìn của đuôi xe: |y| < 0.5 m, z từ -1.6 đến -1.0 m
        return int(((p[:, 0] > 19.9) & (np.abs(p[:, 1]) < 0.5) & (p[:, 2] > -1.6) & (p[:, 2] < -1.0)).sum())

    aug, info = insert_object(target, template, pos, plane, rng)
    ann = create_annotation(template, pos, project_z_to_ground(pos[0], pos[1], plane), info["n_visible"])
    print(f"target {target.shape} -> augmented {aug.shape}; info: {info}")
    print(f"điểm tường sau xe: trước {wall_behind(target)} -> sau {wall_behind(aug)} (phải ~0, bị xe che)")
    print("annotation:", {k: ann[k] for k in ("translation", "size", "category", "num_lidar_pts")})
