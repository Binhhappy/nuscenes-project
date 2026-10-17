"""Coordinate transforms: sensor <-> ego <-> global <-> object-local.

Quy ước: points (N, >=3), chỉ 3 cột đầu (xyz) bị biến đổi, các cột khác giữ nguyên.
Quaternion theo nuScenes: [w, x, y, z]. size theo nuScenes: [w, l, h].

Chạy demo: python utils/transforms.py
"""
from typing import Sequence

import numpy as np
from pyquaternion import Quaternion


def transform_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    """Ma trận 4x4 từ translation (3,) và quaternion [w, x, y, z]."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Áp T (4x4) lên xyz của points, trả về bản copy."""
    out = np.array(points, copy=True)
    out[:, :3] = points[:, :3] @ T[:3, :3].T + T[:3, 3]
    return out


def sensor_to_global_matrix(calibration: dict, ego_pose: dict) -> np.ndarray:
    """T_global<-sensor = T_global<-ego @ T_ego<-sensor."""
    return (transform_matrix(ego_pose["translation"], ego_pose["rotation"])
            @ transform_matrix(calibration["translation"], calibration["rotation"]))


def sensor_to_global(points: np.ndarray, calibration: dict, ego_pose: dict) -> np.ndarray:
    """Points sensor frame -> global frame."""
    return apply_transform(points, sensor_to_global_matrix(calibration, ego_pose))


def global_to_sensor(points: np.ndarray, calibration: dict, ego_pose: dict) -> np.ndarray:
    """Points global frame -> sensor frame."""
    return apply_transform(points, np.linalg.inv(sensor_to_global_matrix(calibration, ego_pose)))


def to_local_frame(points: np.ndarray, center: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    """Points -> object-local frame: gốc = center, trục x = hướng đầu object."""
    out = np.array(points, copy=True)
    R = Quaternion(rotation).rotation_matrix
    out[:, :3] = (points[:, :3] - np.asarray(center)) @ R  # R^T @ p cho từng điểm
    return out


def from_local_frame(points: np.ndarray, target_center: Sequence[float],
                     target_rotation: Sequence[float]) -> np.ndarray:
    """Object-local frame -> vị trí mới (target_center, target_rotation)."""
    out = np.array(points, copy=True)
    R = Quaternion(target_rotation).rotation_matrix
    out[:, :3] = points[:, :3] @ R.T + np.asarray(target_center)
    return out


def points_in_cuboid(points: np.ndarray, center: Sequence[float], size: Sequence[float],
                     rotation: Sequence[float]) -> np.ndarray:
    """Mask bool (N,) các điểm nằm trong cuboid (size = [w, l, h])."""
    local = to_local_frame(points[:, :3], center, rotation)
    w, l, h = size
    return ((np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 1]) <= w / 2)
            & (np.abs(local[:, 2]) <= h / 2))


def yaw_to_quaternion(yaw: float) -> list:
    """Yaw (rad, quanh trục z) -> quaternion [w, x, y, z]."""
    return list(Quaternion(axis=[0, 0, 1], angle=yaw).elements)


def quaternion_to_yaw(rotation: Sequence[float]) -> float:
    return float(Quaternion(rotation).yaw_pitch_roll[0])


def wrap_angle(angle):
    """Đưa góc (rad, scalar hoặc array) về [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def box_sensor_to_global(box: dict, calibration: dict, ego_pose: dict) -> dict:
    """Annotation dict (translation, rotation) sensor frame -> global frame."""
    T = sensor_to_global_matrix(calibration, ego_pose)
    q = Quaternion(matrix=T[:3, :3]) * Quaternion(box["rotation"])
    center = apply_transform(np.asarray([box["translation"]], dtype=float), T)[0]
    return {**box, "translation": center.tolist(), "rotation": list(q.elements)}



def box_global_to_sensor(box: dict, calibration: dict, ego_pose: dict) -> dict:
    """Annotation dict global frame -> sensor frame (nghịch đảo box_sensor_to_global)."""
    T = np.linalg.inv(sensor_to_global_matrix(calibration, ego_pose))
    q = Quaternion(matrix=T[:3, :3]) * Quaternion(box["rotation"])
    center = apply_transform(np.asarray([box["translation"]], dtype=float), T)[0]
    return {**box, "translation": center.tolist(), "rotation": list(q.elements)}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-2, 2, (1000, 3)), rng.uniform(0, 255, (1000, 2))]
    calib = {"translation": [0.9, 0.0, 1.8], "rotation": yaw_to_quaternion(-np.pi / 2)}
    pose = {"translation": [400.0, 1100.0, 0.0], "rotation": yaw_to_quaternion(0.3)}

    back = global_to_sensor(sensor_to_global(pts, calib, pose), calib, pose)
    print("sensor->global->sensor max err:", np.abs(back - pts).max())

    center, size, rot = [0.5, 0.0, 0.0], [1.0, 2.0, 1.0], yaw_to_quaternion(0.4)
    mask = points_in_cuboid(pts, center, size, rot)
    local = to_local_frame(pts[mask], center, rot)
    moved = from_local_frame(local, [10.0, 5.0, 0.0], yaw_to_quaternion(1.2))
    print(f"in cuboid: {mask.sum()} pts; local max |x|={np.abs(local[:, 0]).max():.2f} (<= l/2=1.0)")
    print("moved centroid:", moved[:, :3].mean(0).round(2))
