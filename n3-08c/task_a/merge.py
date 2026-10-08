"""Merge: transform template tới vị trí mới, concat vào target point cloud, tạo annotation mới.

Chạy demo: python task_a/merge.py
"""
import sys
import uuid
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_a.ground_plane import GroundModel, Plane, project_z_to_ground  # noqa: E402
from utils.transforms import from_local_frame, yaw_to_quaternion  # noqa: E402


def insert_object(target_points: np.ndarray, object_template: dict, position: Tuple[float, float, float],
                  ground_plane: Union[Plane, GroundModel]) -> np.ndarray:
    """Đặt template tại position = (x, y, yaw), đáy box chạm mặt đất -> augmented points (N+M, 5).

    ground_plane: plane (a, b, c, d) hoặc GroundModel (độ cao cục bộ, chính xác hơn).
    """
    x, y, yaw = position
    h = object_template["size"][2]
    center = [x, y, project_z_to_ground(x, y, ground_plane) + h / 2]
    placed = from_local_frame(object_template["points"], center, yaw_to_quaternion(yaw))
    # TODO: xoá điểm target nằm trong box mới (ground clutter).
    # TODO: occlusion — xoá điểm target bị object mới che (range image theo azimuth/elevation),
    #       và điểm object bị vật thể gần hơn che.
    # TODO: gán lại ring theo elevation mới, resample theo beam pattern (1 return / beam).
    # TODO: (tuỳ chọn) chỉnh intensity theo khoảng cách mới.
    return np.concatenate([target_points, placed.astype(target_points.dtype)], axis=0)


def create_annotation(object_template: dict, position: Tuple[float, float, float], ground_z: float) -> dict:
    """Annotation dict (sensor frame) cho object vừa chèn."""
    x, y, yaw = position
    size: Sequence[float] = object_template["size"]
    return {
        "token": uuid.uuid4().hex,
        "translation": [float(x), float(y), float(ground_z + size[2] / 2)],
        "size": [float(v) for v in size],
        "rotation": yaw_to_quaternion(yaw),
        "category": object_template["category"],
        "num_lidar_pts": int(len(object_template["points"])),  # TODO: cập nhật sau occlusion
        "synthetic": True,
        "source_ann_token": object_template.get("ann_token"),
    }


if __name__ == "__main__":
    from utils.transforms import points_in_cuboid

    rng = np.random.default_rng(0)
    target = np.c_[rng.uniform(-30, 30, (5000, 2)), np.full(5000, -1.8), rng.uniform(0, 255, (5000, 2))]
    template = {
        "points": np.c_[rng.uniform([-2.2, -0.9, -0.8], [2.2, 0.9, 0.8], (300, 3)), rng.uniform(0, 255, (300, 2))],
        "size": [1.9, 4.5, 1.6], "category": "vehicle.car", "ann_token": "demo",
    }
    plane = (0.0, 0.0, 1.0, 1.8)  # z = -1.8
    pos = (12.0, -3.0, np.pi / 4)

    aug = insert_object(target, template, pos, plane)
    ann = create_annotation(template, pos, project_z_to_ground(pos[0], pos[1], plane))
    inside = points_in_cuboid(aug, ann["translation"], ann["size"], ann["rotation"]).sum()
    print(f"target {target.shape} -> augmented {aug.shape}; points trong box mới: {inside}")
    print("annotation:", {k: ann[k] for k in ("translation", "size", "category")})
