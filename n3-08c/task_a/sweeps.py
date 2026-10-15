"""T4: chèn object vào cả các sweep (LiDAR 20 Hz) trước keyframe, và gộp multi-sweep như model nuScenes.

Detector nuScenes (PointPillars, CenterPoint) đọc keyframe + 9 sweep trước đó (~0.45 s) gộp lại, thêm cột
time_lag. Chỉ chèn vào keyframe thì object chỉ có ở 1/10 khung: model thấy object "chập chờn", lệch phân bố
so với object thật (có điểm ở cả 10 sweep).

Cách làm: box object ở global frame tại keyframe; ở sweep cách keyframe dt giây, object ở
    translation_global - velocity * dt     (velocity = [vx, vy] global, m/s; 0 = đứng yên)
Chuyển box về sensor frame của sweep, chèn bằng merge.insert_object (mật độ, ring, xoá điểm trong box,
che khuất tính theo chính sweep đó). Mặt nhìn thấy lấy từ template như keyframe: ego dời vài mét trong
0.45 s nên góc nhìn lệch nhỏ (chấp nhận được).

Output: <out_dir>/sweeps/<sample_data_token>.bin cho từng sweep đã chèn.
Gộp: load_multisweep(nusc, sample_token, aug_dir) -> (N, 5) [x, y, z, intensity, time_lag] ở keyframe sensor
frame, lấy file đã chèn nếu có (cùng format LoadPointsFromMultiSweeps của mmdet3d).

Chạy demo: python task_a/sweeps.py   (chèn 3 xe vào sample[10] + 9 sweep, in số điểm từng sweep)
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import GroundModel  # noqa: E402
from task_a.merge import insert_object, points_in_new_box  # noqa: E402
from utils.nuscenes_loader import (get_calibration, get_ego_pose, get_lidar_token, load_bin,  # noqa: E402
                                   save_bin)
from utils.transforms import (apply_transform, box_global_to_sensor, quaternion_to_yaw,  # noqa: E402
                              sensor_to_global_matrix)


def sweep_tokens(nusc, sample_token: str, n_sweeps: int = config.N_SWEEPS) -> List[str]:
    """sample_data token LIDAR_TOP: [keyframe, sweep trước 1, ..., sweep trước n_sweeps - 1]."""
    tokens = [get_lidar_token(nusc, sample_token)]
    while len(tokens) < n_sweeps:
        prev = nusc.get("sample_data", tokens[-1])["prev"]
        if not prev:
            break
        tokens.append(prev)
    return tokens


def sweep_path(out_dir: str, sd_token: str) -> str:
    return os.path.join(out_dir, "sweeps", f"{sd_token}.bin")


def insert_into_sweeps(nusc, sample_token: str, annotations_global: Sequence[dict],
                       templates: Dict[str, dict], out_dir: str, n_sweeps: int = config.N_SWEEPS,
                       rng: Optional[np.random.Generator] = None) -> List[dict]:
    """Chèn các object (đã chèn ở keyframe) vào n_sweeps - 1 sweep trước keyframe, lưu .bin.

    annotations_global: annotation global frame của keyframe (có "velocity" [vx, vy] nếu object chạy,
        "source_ann_token" để tìm template).
    templates: {ann_token của template: template}.
    Returns list mỗi sweep: sample_data_token, time_lag (s), file (tương đối out_dir), n_visible (theo object).
    """
    rng = rng or np.random.default_rng(config.SEED)
    tokens = sweep_tokens(nusc, sample_token, n_sweeps)
    t_key = nusc.get("sample_data", tokens[0])["timestamp"]
    out = []
    for sd_token in tokens[1:]:
        sd = nusc.get("sample_data", sd_token)
        dt = (t_key - sd["timestamp"]) / 1e6
        calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)
        points = load_bin(nusc.get_sample_data_path(sd_token))
        ground = GroundModel.from_points(points, rng=rng)
        placed = []
        for ann in annotations_global:
            vx, vy = ann.get("velocity", (0.0, 0.0))
            moved = dict(ann, translation=[ann["translation"][0] - vx * dt, ann["translation"][1] - vy * dt,
                                           ann["translation"][2]])
            box = box_global_to_sensor(moved, calib, pose)
            x, y = box["translation"][:2]
            tpl = templates[ann["source_ann_token"]]
            if "primitives" in tpl:  # vật thể ray-cast (task_a/raycast.py): bắn tia lại theo sweep này
                from task_a.raycast import insert_raycast
                insert = insert_raycast
            else:
                insert = insert_object
            points, _ = insert(points, tpl, (x, y, quaternion_to_yaw(box["rotation"])), ground, rng)
            placed.append(box)
        # object chèn sau có thể che object chèn trước: đếm điểm còn lại trong từng box ở cuối
        n_visible = [int(points_in_new_box(points, b).sum()) for b in placed]
        save_bin(sweep_path(out_dir, sd_token), points)
        out.append({"sample_data_token": sd_token, "time_lag": dt,
                    "file": os.path.relpath(sweep_path(out_dir, sd_token), out_dir), "n_visible": n_visible})
    return out


def load_multisweep(nusc, sample_token: str, aug_dir: Optional[str] = None,
                    n_sweeps: int = config.N_SWEEPS, per_sweep=None) -> np.ndarray:
    """Keyframe + sweep trước đó gộp ở keyframe sensor frame: (N, 5) [x, y, z, intensity, time_lag].

    aug_dir: thư mục output đã chèn; dùng <aug_dir>/<sample_token>.bin cho keyframe và
    <aug_dir>/sweeps/<sd_token>.bin cho sweep nếu có, không có thì dùng file gốc nuScenes.
    per_sweep: hàm (points (M, 5) sensor frame của sweep, chỉ số sweep) -> points, áp trước khi gộp
    (vd Task B thời tiết: mỗi sweep là 1 lần quét riêng, cột 4 còn là ring).
    """
    tokens = sweep_tokens(nusc, sample_token, n_sweeps)
    key = tokens[0]
    t_key = nusc.get("sample_data", key)["timestamp"]
    to_key = np.linalg.inv(sensor_to_global_matrix(get_calibration(nusc, key), get_ego_pose(nusc, key)))
    clouds = []
    for i, sd_token in enumerate(tokens):
        path = nusc.get_sample_data_path(sd_token)
        if aug_dir:
            cand = os.path.join(aug_dir, f"{sample_token}.bin") if i == 0 else sweep_path(aug_dir, sd_token)
            path = cand if os.path.isfile(cand) else path
        pts = load_bin(path)
        if per_sweep is not None:
            pts = per_sweep(pts, i)
        if i > 0:
            T = to_key @ sensor_to_global_matrix(get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token))
            pts = pts.copy()
            pts[:, :3] = apply_transform(pts[:, :3].astype(float), T)
        lag = (t_key - nusc.get("sample_data", sd_token)["timestamp"]) / 1e6
        clouds.append(np.c_[pts[:, :4], np.full(len(pts), lag, np.float32)])
    return np.concatenate(clouds, axis=0).astype(np.float32)


if __name__ == "__main__":
    import tempfile

    from task_a.run_insertion import collect_templates, run_insertion, save_result
    from utils.nuscenes_loader import load_nuscenes

    nusc = load_nuscenes()
    rng = np.random.default_rng(config.SEED)
    target = nusc.sample[10]["token"]
    tpls = collect_templates(nusc, nusc.sample[0]["token"], None, 3, rng=rng)
    result = run_insertion(nusc, target, tpls, rng=rng)
    out_dir = tempfile.mkdtemp(prefix="sweeps_demo_")
    save_result(result, out_dir)
    sweeps = insert_into_sweeps(nusc, target, result["annotations_global"],
                                {t["ann_token"]: t for t in tpls}, out_dir, rng=rng)
    print("keyframe:", [a["num_lidar_pts"] for a in result["annotations_sensor"]], "điểm / object")
    for s in sweeps:
        print(f"  sweep -{s['time_lag']:.2f}s: {s['n_visible']}")
    ms = load_multisweep(nusc, target, out_dir)
    print("multi-sweep:", ms.shape, "time_lag max", ms[:, 4].max().round(3), "->", out_dir)
