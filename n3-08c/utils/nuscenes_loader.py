"""Load nuScenes: instance, point cloud, annotations, ego pose, calibration.

Chạy demo: python utils/nuscenes_loader.py
"""
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.geometry_utils import BoxVisibility  # noqa: E402

_NUSC_CACHE: Dict[tuple, NuScenes] = {}


def load_nuscenes(version: str = config.VERSION, dataroot: str = config.DATAROOT,
                  verbose: bool = False) -> NuScenes:
    """NuScenes instance (cache theo version + dataroot)."""
    if not os.path.isfile(os.path.join(dataroot, version, "sample.json")):
        raise FileNotFoundError(f"Không thấy {version}/sample.json trong {dataroot} "
                                f"(sửa config.DATAROOT hoặc đặt NUSCENES_DATAROOT)")
    key = (version, dataroot)
    if key not in _NUSC_CACHE:
        _NUSC_CACHE[key] = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    return _NUSC_CACHE[key]


def get_lidar_token(nusc: NuScenes, sample_token: str) -> str:
    """sample_data token của LIDAR_TOP trong sample."""
    return nusc.get("sample", sample_token)["data"][config.LIDAR_CHANNEL]


def get_pointcloud(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """Point cloud (N, 5) float32 [x, y, z, intensity, ring] trong LIDAR_TOP sensor frame."""
    path = nusc.get_sample_data_path(get_lidar_token(nusc, sample_token))
    return load_bin(path)


def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def save_bin(path: str, points: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.ascontiguousarray(points, dtype=np.float32).tofile(path)


def get_annotations(nusc: NuScenes, sample_token: str, frame: str = "sensor") -> List[dict]:
    """Annotations của sample dạng dict: token, translation, size (wlh), rotation [w,x,y,z], category.

    frame="sensor": box ở LIDAR_TOP sensor frame (cùng frame với get_pointcloud).
    frame="global": box ở global frame (nguyên gốc nuScenes).
    """
    if frame == "global":
        anns = [nusc.get("sample_annotation", t) for t in nusc.get("sample", sample_token)["anns"]]
        return [{"token": a["token"], "translation": list(a["translation"]), "size": list(a["size"]),
                 "rotation": list(a["rotation"]), "category": a["category_name"]} for a in anns]
    if frame != "sensor":
        raise ValueError(f"frame phải là 'sensor' hoặc 'global', nhận {frame!r}")
    _, boxes, _ = nusc.get_sample_data(get_lidar_token(nusc, sample_token),
                                       box_vis_level=BoxVisibility.NONE)
    return [{"token": b.token, "translation": b.center.tolist(), "size": b.wlh.tolist(),
             "rotation": list(b.orientation.elements), "category": b.name} for b in boxes]


def get_ego_pose(nusc: NuScenes, sample_data_token: str) -> dict:
    """ego_pose record (translation, rotation) lúc chụp sample_data."""
    sd = nusc.get("sample_data", sample_data_token)
    return nusc.get("ego_pose", sd["ego_pose_token"])


def get_calibration(nusc: NuScenes, sample_data_token: str) -> dict:
    """calibrated_sensor record (sensor -> ego) của sample_data."""
    sd = nusc.get("sample_data", sample_data_token)
    return nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])


def get_map_name(nusc: NuScenes, sample_token: str) -> str:
    """Tên map (location) của sample, vd 'singapore-onenorth'."""
    scene = nusc.get("scene", nusc.get("sample", sample_token)["scene_token"])
    return nusc.get("log", scene["log_token"])["location"]


if __name__ == "__main__":
    nusc = load_nuscenes()
    sample_token = nusc.sample[0]["token"]
    sd_token = get_lidar_token(nusc, sample_token)
    pts = get_pointcloud(nusc, sample_token)
    anns = get_annotations(nusc, sample_token)
    print(f"sample {sample_token}  map={get_map_name(nusc, sample_token)}")
    print(f"points: {pts.shape}, annotations: {len(anns)}")
    print("ego_pose:", get_ego_pose(nusc, sd_token)["translation"])
    print("calibration:", get_calibration(nusc, sd_token)["translation"])
    print("ann[0]:", anns[0])
