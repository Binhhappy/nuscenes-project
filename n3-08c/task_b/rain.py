"""Mô phỏng mưa trên LiDAR point cloud.

4 bước: (1) dropout theo khoảng cách -> (2) noise gần sensor (range < 10 m, giọt mưa)
-> (3) suy giảm intensity theo khoảng cách -> (4) range jitter dọc tia.
Chạy demo: python task_b/rain.py
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# Cường độ mưa (mm/h) tương ứng với từng level (dùng cho range jitter)
RAIN_RATES = {"light": 2.5, "moderate": 10.0, "heavy": 50.0}
# Hệ số suy giảm intensity (1/m) theo level. Công thức alpha = 0.01 * R^0.6 đã thử và bỏ: trên dữ liệu thật
# level heavy (alpha ~ 0.105) đưa intensity của mọi điểm xa hơn 20 m về ~0.
ATTENUATION = {"light": 0.005, "moderate": 0.01, "heavy": 0.02}
# Khoảng cách (m) tại đó tỉ lệ mất điểm = dropout trong config. Điểm LiDAR tập trung gần xe (trung vị ~7 m);
# với mốc 30 m số điểm mất chỉ xấp xỉ số noise thêm vào.
DROPOUT_REF_RANGE = 10.0
NOISE_MAX_RANGE = 10.0


def simulate_rain(points: np.ndarray, intensity_level: str = "moderate",
                  rng: Optional[np.random.Generator] = None, return_labels: bool = False,
                  labels: Optional[np.ndarray] = None):
    """Mưa theo level trong config.RAIN_LEVELS: light / moderate / heavy.
    Dropout theo khoảng cách (mốc DROPOUT_REF_RANGE), suy giảm intensity theo ATTENUATION.

    labels: nhãn đầu vào (N,) của points (khi nối sau weather khác); mặc định toàn LABEL_ORIGINAL.
    Returns augmented points (K, 5); nếu return_labels thì thêm labels (K,): điểm còn sống giữ nhãn
    đầu vào, giọt mưa = LABEL_NOISE.
    """
    if intensity_level not in config.RAIN_LEVELS:
        raise ValueError(f"intensity_level phải thuộc {list(config.RAIN_LEVELS)}")
    rng = rng or np.random.default_rng()
    p = config.RAIN_LEVELS[intensity_level]
    
    R_rate = RAIN_RATES.get(intensity_level, 10.0)
    alpha = ATTENUATION[intensity_level]

    dist = np.linalg.norm(points[:, :3], axis=1)

    # (1) Dropout phụ thuộc khoảng cách (điểm xa dễ bị hấp thụ/tán xạ hơn)
    # Xác suất sống sót (survival) giảm theo e^(-beta * d)
    # Chọn beta sao cho tỷ lệ dropout ở DROPOUT_REF_RANGE tương đương với p["dropout"] trong config
    beta = -np.log(max(1e-5, 1.0 - p["dropout"])) / DROPOUT_REF_RANGE
    survival_prob = np.exp(-beta * dist)
    
    kept_mask = rng.random(len(points)) < survival_prob
    kept = points[kept_mask].copy()
    kept_dist = dist[kept_mask]

    # (2) Noise gần sensor: số điểm = noise_ratio * N, trong cầu bán kính 10 m
    n_noise = int(p["noise_ratio"] * len(points))
    r = rng.uniform(0.5, NOISE_MAX_RANGE, n_noise)
    az = rng.uniform(-np.pi, np.pi, n_noise)
    el = np.radians(rng.uniform(-30.67, 10.67, n_noise))  # FOV dọc HDL-32E
    noise = np.c_[r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el),
                  rng.uniform(0, 5, n_noise), rng.integers(0, 32, n_noise)].astype(points.dtype)

    # (3) Suy giảm intensity theo khoảng cách (2 chiều): I' = I * exp(-2 * alpha * r)
    kept[:, 3] *= np.exp(-2 * alpha * kept_dist)
    
    # Range noise (jitter dọc tia) tăng theo khoảng cách và rain rate
    jitter = rng.normal(0, 0.01 * (R_rate ** 0.5) * (kept_dist / 30.0), len(kept))
    # Dịch chuyển xyz dọc theo tia quét
    kept[:, :3] *= (1.0 + jitter / np.maximum(kept_dist, 1e-3))[:, None]

    out = np.concatenate([kept, noise], axis=0)
    if return_labels:
        kept_labels = (np.full(len(kept), config.LABEL_ORIGINAL, np.int8) if labels is None
                       else np.asarray(labels, np.int8)[kept_mask])
        return out, np.r_[kept_labels, np.full(n_noise, config.LABEL_NOISE, np.int8)]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    pts = np.c_[rng.uniform(-50, 50, (30000, 2)), rng.uniform(-2, 1, 30000),
                rng.uniform(0, 255, 30000), rng.integers(0, 32, 30000)].astype(np.float32)
    for level in config.RAIN_LEVELS:
        out, lab = simulate_rain(pts, level, rng, return_labels=True)
        print(f"{level:9s}: {len(pts)} -> {len(out)} pts (noise {(lab == config.LABEL_NOISE).sum()}), "
              f"mean intensity {pts[:, 3].mean():.1f} -> {out[lab == config.LABEL_ORIGINAL, 3].mean():.1f}")
