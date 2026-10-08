"""Sample vị trí hợp lệ (x, y, yaw) trên drivable area + collision check BEV.

Box = annotation dict (translation, size [w, l, h], rotation [w,x,y,z]) ở sensor frame.
Chạy demo: python task_a/placement.py
"""
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.transforms import quaternion_to_yaw, yaw_to_quaternion  # noqa: E402

Position = Tuple[float, float, float]  # (x, y, yaw)


def bev_footprint(center_xy: Sequence[float], size: Sequence[float], yaw: float) -> Polygon:
    """Hình chữ nhật BEV của box (size = [w, l, h])."""
    w, l = size[0], size[1]
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon(corners @ np.array([[c, s], [-s, c]]) + np.asarray(center_xy[:2]))


def box_footprint(box: dict) -> Polygon:
    return bev_footprint(box["translation"], box["size"], quaternion_to_yaw(box["rotation"]))


def bev_iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter + 1e-9)


def check_collision_bev(new_box: dict, existing_boxes: Sequence[dict]) -> bool:
    """True nếu new_box có BEV IoU > COLLISION_IOU_THRESH với bất kỳ box nào."""
    fp = box_footprint(new_box)
    # TODO: buffer footprint thêm margin an toàn (vd 0.3 m) để object không dính sát nhau.
    return any(bev_iou(fp, box_footprint(b)) > config.COLLISION_IOU_THRESH for b in existing_boxes)


def is_on_drivable(point_xy: Sequence[float], drivable_polygon: BaseGeometry) -> bool:
    """True nếu điểm (x, y) nằm trong drivable area."""
    return drivable_polygon.contains(Point(point_xy[0], point_xy[1]))


def sample_valid_position(drivable_polygon: BaseGeometry, existing_boxes: Sequence[dict],
                          object_size: Sequence[float], n_attempts: int = 100,
                          rng: Optional[np.random.Generator] = None) -> Optional[Position]:
    """Rejection sampling: (x, y) đều trong vành khuyên [MIN_DISTANCE, MAX_DISTANCE], yaw ngẫu nhiên.

    Hợp lệ khi: tâm + toàn bộ footprint nằm trong drivable area và không va chạm BEV.
    Returns (x, y, yaw) hoặc None nếu hết n_attempts.
    """
    rng = rng or np.random.default_rng()
    r_min, r_max = config.MIN_DISTANCE, config.MAX_DISTANCE
    for _ in range(n_attempts):
        r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2))  # đều theo diện tích
        phi = rng.uniform(-np.pi, np.pi)
        x, y = r * np.cos(phi), r * np.sin(phi)
        if not is_on_drivable((x, y), drivable_polygon):
            continue
        # TODO: yaw theo hướng lane gần nhất (nusc_map.get_closest_lane + arcline_path_utils)
        #       và giữ view_angle gần với template để mặt nhìn thấy khớp.
        yaw = rng.uniform(-np.pi, np.pi)
        if not drivable_polygon.contains(bev_footprint((x, y), object_size, yaw)):
            continue
        new_box = {"translation": [x, y, 0.0], "size": list(object_size), "rotation": yaw_to_quaternion(yaw)}
        if check_collision_bev(new_box, existing_boxes):
            continue
        # TODO: reject nếu box chứa nhiều điểm non-ground (vật cản không có annotation: cột, tường).
        return float(x), float(y), float(yaw)
    return None


def sample_many(drivable_polygon: BaseGeometry, existing_boxes: Sequence[dict],
                sizes: Sequence[Sequence[float]], n_attempts: int = 100,
                rng: Optional[np.random.Generator] = None) -> List[Optional[Position]]:
    """Đặt lần lượt nhiều object; object đã đặt được tính vào collision check của object sau."""
    boxes, out = list(existing_boxes), []
    for size in sizes:
        pos = sample_valid_position(drivable_polygon, boxes, size, n_attempts, rng)
        out.append(pos)
        if pos is not None:
            boxes.append({"translation": [pos[0], pos[1], 0.0], "size": list(size),
                          "rotation": yaw_to_quaternion(pos[2])})
    return out


if __name__ == "__main__":
    from shapely.geometry import box as shp_box

    rng = np.random.default_rng(config.SEED)
    road = shp_box(-60, -4, 60, 4).union(shp_box(-4, -60, 4, 60))  # ngã tư giả
    existing = [{"translation": [10.0, 1.5, 0.0], "size": [1.9, 4.5, 1.6], "rotation": yaw_to_quaternion(0.0)}]
    car = [1.9, 4.5, 1.6]

    print("is_on_drivable((10, 0)):", is_on_drivable((10, 0), road))
    print("collision với chính nó:", check_collision_bev(existing[0], existing))
    positions = sample_many(road, existing, [car] * 5, rng=rng)
    for p in positions:
        print("  ->", None if p is None else tuple(round(v, 2) for v in p))
