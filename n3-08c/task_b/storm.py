"""Mô phỏng giông / bão trên LiDAR point cloud (mưa nhiệt đới rất to, thường gặp ở VN mùa hè).

Với LiDAR, giông / bão = các hiệu ứng của mưa ở mức cực đoan, cộng 2 hiệu ứng mưa thường không có:
    (1) mưa rất to: chạy simulate_rain("heavy") rồi thêm dropout tăng theo khoảng cách
        p_mất = 1 - exp(-r / EXTINCTION_RANGE) (tia đi xa qua nhiều giọt nước hơn)
    (2) mặt đường ướt: mặt nước phản xạ gương, tia chiếu xiên xuống đường ở xa không quay về sensor ->
        điểm mặt đường (z thấp) ở r > WET_START mất dần, tới WET_MAX ở r >= WET_START + WET_SPAN
    (3) bụi nước bắn sau bánh xe (spray): đám điểm nhiễu sau đuôi mỗi xe đang chạy (cần box xe)
Gió mạnh và sét không đổi trực tiếp phép đo LiDAR nên không mô phỏng (gió chỉ làm mưa xiên, đã nằm trong (1)).
Tham số là giả định có thứ tự hợp lý (mạnh hơn rain heavy), CHƯA hiệu chỉnh theo dữ liệu đo thật.

Chạy demo: python task_b/storm.py
"""
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_b.rain import simulate_rain  # noqa: E402

EXTINCTION_RANGE = 120.0  # (m) mất thêm ~8% điểm ở 10 m, ~34% ở 50 m
GROUND_Z = -1.4           # điểm thấp hơn (sensor frame, LIDAR_TOP cao ~1.84 m) coi là mặt đường
WET_START, WET_SPAN, WET_MAX = 8.0, 30.0, 0.85
SPRAY_POINTS = 60         # số điểm spray sau 1 xe ở 10 m (giảm theo 1/r^2 như mật độ tia)
SPRAY_LEN = 3.0           # (m) đám bụi nước kéo dài sau đuôi xe


def wet_road_mask(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mask điểm GIỮ LẠI sau hiệu ứng mặt đường ướt."""
    r = np.hypot(points[:, 0], points[:, 1])
    p = np.clip((r - WET_START) / WET_SPAN, 0, 1) * WET_MAX
    ground = points[:, 2] < GROUND_Z
    return ~(ground & (rng.random(len(points)) < p))


def puddle_mask(points: np.ndarray, rng: np.random.Generator, n: int = 8, radius=(0.5, 3.0),
                max_range: float = 25.0, drop: float = 0.85) -> np.ndarray:
    """Mask điểm GIỮ LẠI sau vũng nước: n vùng elip trên mặt đường (trong max_range), mất drop điểm (phản xạ gương).

    Gốc: task_b/wet_road.simulate_puddles trong repo nhóm (tranducquan2k5-afk/buildPhasev1.2), viết gọn lại;
    bổ sung cho wet_road_mask (mất theo khoảng cách) bằng các mảng mất điểm cục bộ gần xe.
    """
    ground = (points[:, 2] < GROUND_Z) & (np.hypot(points[:, 0], points[:, 1]) < max_range)
    if not ground.any() or n <= 0:
        return np.ones(len(points), bool)
    g = points[ground]
    hit = np.zeros(len(g), bool)
    for _ in range(n):
        c = g[rng.integers(len(g)), :2]  # tâm đặt trên điểm mặt đường có thật
        rx, ry = rng.uniform(*radius, 2)
        hit |= ((g[:, 0] - c[0]) / rx) ** 2 + ((g[:, 1] - c[1]) / ry) ** 2 <= 1
    keep = np.ones(len(points), bool)
    keep[np.flatnonzero(ground)[hit & (rng.random(len(g)) < drop)]] = False
    return keep


def spray(boxes: Sequence[dict], rng: np.random.Generator, dtype=np.float32) -> np.ndarray:
    """Điểm bụi nước sau đuôi các xe (box sensor frame có size [w, l, h], yaw từ rotation)."""
    from utils.transforms import quaternion_to_yaw
    out = []
    for b in boxes:
        x, y, z = b["translation"]
        w, l, h = b["size"]
        r = max(np.hypot(x, y), 3.0)
        n = int(rng.poisson(SPRAY_POINTS * (10.0 / r) ** 2))
        if n == 0:
            continue
        yaw = quaternion_to_yaw(b["rotation"])
        u = -l / 2 - rng.uniform(0, SPRAY_LEN, n)            # sau đuôi xe (object frame)
        v = rng.uniform(-w / 2, w / 2, n)
        zz = z - h / 2 + rng.uniform(0, 1.2, n) * rng.random(n)  # dày ở sát mặt đường
        c, s = np.cos(yaw), np.sin(yaw)
        out.append(np.c_[x + c * u - s * v, y + s * u + c * v, zz, rng.uniform(0, 4, n),
                         rng.integers(0, config.LIDAR_RINGS, n)])
    return np.concatenate(out).astype(dtype) if out else np.zeros((0, 5), dtype)


def simulate_storm(points: np.ndarray, rng: Optional[np.random.Generator] = None,
                   vehicle_boxes: Sequence[dict] = (), return_labels: bool = False, puddles: int = 0):
    """Giông / bão. vehicle_boxes: box xe đang chạy (sensor frame) để sinh spray; bỏ trống = không spray.
    puddles: số vũng nước cục bộ (0 = tắt, mặc định: giữ đúng các số đã đo ở tools/weather_eval.py).

    Returns (K, 5); nếu return_labels thì thêm labels (K,) 0 = điểm gốc, 1 = nhiễu (mưa / spray).
    """
    rng = rng or np.random.default_rng()
    pts, lab = simulate_rain(points, "heavy", rng, return_labels=True)
    r = np.linalg.norm(pts[:, :3], axis=1)
    keep = (rng.random(len(pts)) >= 1 - np.exp(-r / EXTINCTION_RANGE)) | (lab == 1)
    keep &= wet_road_mask(pts, rng) | (lab == 1)
    if puddles:
        keep &= puddle_mask(pts, rng, puddles) | (lab == 1)
    pts, lab = pts[keep], lab[keep]
    sp = spray(vehicle_boxes, rng, pts.dtype)
    out = np.concatenate([pts, sp], axis=0)
    lab = np.r_[lab, np.ones(len(sp), np.int8)]
    return (out, lab) if return_labels else out


if __name__ == "__main__":
    from utils.nuscenes_loader import get_annotations, get_pointcloud, load_nuscenes
    nusc = load_nuscenes()
    tok = nusc.sample[10]["token"]
    pts = get_pointcloud(nusc, tok)
    cars = [a for a in get_annotations(nusc, tok) if a["category"].startswith("vehicle.")]
    rng = np.random.default_rng(config.SEED)
    for name, fn in (("rain heavy", lambda p: simulate_rain(p, "heavy", rng, True)),
                     ("storm", lambda p: simulate_storm(p, rng, cars, True))):
        out, lab = fn(pts)
        ground_before = (pts[:, 2] < GROUND_Z).sum()
        ground_after = ((out[:, 2] < GROUND_Z) & (lab == 0)).sum()
        print(f"{name:10s}: {len(pts)} -> {len(out)} điểm (nhiễu {int(lab.sum())}), "
              f"điểm mặt đường {ground_before} -> {ground_after}")
