"""Thời tiết trên ảnh CAMERA, khớp với thời tiết đã mô phỏng trên LiDAR (task_b/rain.py, fog.py, storm.py).

Gốc: webapp/backend/services/camera_projection.py (apply_rain_2d, apply_fog_2d) trong repo nhóm
tranducquan2k5-afk/buildPhasev1.2; gom lại thành module độc lập, thêm giông bão = mưa to + sương nhẹ.
    mưa  : vệt mưa xiên theo gió + giọt nước đọng trên ống kính (mưa to) + giảm tương phản + mờ nhẹ
    sương: mô hình khí quyển Koschmieder I = J * t + A * (1 - t), t = exp(-beta * d), beta = 3.912 / tầm nhìn;
           d theo phối cảnh mặt đường (đường chân trời ~48% chiều cao ảnh) — xấp xỉ, không dùng bản đồ độ sâu thật
Chỉ để minh hoạ / đồng bộ hình ảnh; các số đo của đề tài là trên LiDAR.

Chạy demo: python task_b/camera_weather.py   (lưu output/vis/camera_weather.png)
"""
from typing import Optional

import cv2
import numpy as np

STREAKS = {"light": 600, "moderate": 1800, "heavy": 4500}
RAIN_ALPHA = {"light": 0.3, "moderate": 0.42, "heavy": 0.55}
CONTRAST = {"light": 0.95, "moderate": 0.88, "heavy": 0.78}
BLUR_SIGMA = {"light": 0.0, "moderate": 0.6, "heavy": 1.2}
FOG_VISIBILITY = {"light": 200.0, "moderate": 50.0, "dense": 20.0}  # (m), như config.FOG_VISIBILITY


def apply_rain(img: np.ndarray, level: str = "moderate", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """img: ảnh BGR / RGB uint8 (H, W, 3); level: light / moderate / heavy."""
    rng = rng or np.random.default_rng(42)
    h, w = img.shape[:2]
    s = w / 1600  # vệt mưa theo kích thước ảnh gốc nuScenes 1600 x 900
    layer = np.zeros_like(img)
    ang = np.radians(rng.uniform(76.0, 84.0))  # mưa xiên do gió
    n = int(STREAKS[level] * (w * h) / (1600 * 900))
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    lens = rng.integers(18, 42, n) * s
    for x, y, ln in zip(xs, ys, lens):
        c = int(rng.integers(190, 245))
        th = 2 if level == "heavy" and rng.random() < 0.25 else 1
        cv2.line(layer, (int(x), int(y)), (int(x + ln * np.cos(ang)), int(y + ln * np.sin(ang))), (c, c, c), th,
                 cv2.LINE_AA)
    if level == "heavy":  # giọt nước đọng trên ống kính
        for _ in range(int(rng.integers(20, 45))):
            cx, cy, rad = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20)), max(2, int(rng.integers(4, 16) * s))
            cv2.circle(layer, (cx, cy), rad, (220, 220, 230), 1, cv2.LINE_AA)
            cv2.circle(layer, (cx, cy), max(1, rad // 2), (240, 240, 250), -1)
    out = cv2.addWeighted(img, 1.0, layer, RAIN_ALPHA[level], 0)
    cf = CONTRAST[level]
    out = np.clip(out.astype(np.float32) * cf + 15 * (1 - cf), 0, 255).astype(np.uint8)
    if BLUR_SIGMA[level] > 0:
        out = cv2.GaussianBlur(out, (3 if BLUR_SIGMA[level] < 1 else 5,) * 2, BLUR_SIGMA[level])
    return out


def apply_fog(img: np.ndarray, visibility: float = 50.0) -> np.ndarray:
    """Koschmieder với độ sâu xấp xỉ theo phối cảnh mặt đường."""
    h, w = img.shape[:2]
    beta = 3.912 / max(5.0, visibility)
    a = np.array([225.0, 225.0, 230.0], np.float32)
    y_hor = int(h * 0.48)
    dy = np.maximum(np.arange(h) - y_hor, 1)
    # trên chân trời: nhà / cây ven đường ~60 m (bản gốc 220 m làm trắng xoá cả khi tầm nhìn 150 m)
    far = 60.0  # mặt đường xa cũng chặn ở 60 m để liền với phần trên chân trời (không có vệt sáng ở chân trời)
    depth = np.where(np.arange(h) < y_hor, far, np.clip(1.7 * 1266.0 * (h / 900) / dy, 3.0, far))
    t = np.exp(-beta * depth)[:, None, None]
    out = np.clip(img.astype(np.float32) * t + a * (1 - t), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (3, 3) if visibility >= 50 else (5, 5), 0)


def apply_camera_weather(img: np.ndarray, weather: str, level: str,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """weather: rain / fog / storm (cùng tên với task_b/run_weather.apply_weather)."""
    if weather == "rain":
        return apply_rain(img, level, rng)
    if weather == "fog":
        return apply_fog(img, FOG_VISIBILITY[level])
    if weather == "storm":  # giông bão: mưa rất to + hơi nước làm giảm tầm nhìn
        return apply_fog(apply_rain(img, "heavy", rng), 150.0)
    raise ValueError(weather)


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from utils.nuscenes_loader import load_nuscenes

    nusc = load_nuscenes()
    sd = nusc.get("sample_data", nusc.sample[10]["data"]["CAM_FRONT"])
    img = cv2.imread(nusc.get_sample_data_path(sd["token"]))
    tiles = [img] + [apply_camera_weather(img, w, lv) for w, lv in (("rain", "heavy"), ("fog", "dense"), ("storm", "storm"))]
    small = [cv2.resize(t, (800, 450)) for t in tiles]
    os.makedirs(config.VIS_DIR, exist_ok=True)
    out = os.path.join(config.VIS_DIR, "camera_weather.png")
    cv2.imwrite(out, np.vstack([np.hstack(small[:2]), np.hstack(small[2:])]))
    print("->", out)
