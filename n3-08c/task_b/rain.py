"""Mô phỏng mưa trên LiDAR point cloud.

3 bước: (1) dropout (tia bị giọt mưa chặn / hấp thụ) -> (2) noise gần sensor (tia phản xạ từ giọt mưa, range < 10 m)
-> (3) suy giảm intensity theo khoảng cách.

config.WEATHER_MODEL:
    v2 (mặc định)  (1) p_mất = 1 - exp(-r / drop_range): tia đi càng xa qua càng nhiều giọt nước (Beer-Lambert);
                   drop_range hiệu chỉnh để tỉ lệ mất trung bình bằng dropout của level.
                   (2) noise nằm TRÊN TIA THẬT: chọn ngẫu nhiên noise_ratio * N tia, điểm của tia đó dời về
                   r ~ U(0.5, min(10, r_gốc)) dọc chính tia (giữ ring + azimuth); điểm gốc mất vì mỗi tia chỉ trả về
                   1 điểm (điểm gần nhất là giọt mưa).
    v1             bản cũ: dropout đều, noise rải đều trong cầu 10 m và cộng thêm vào (giữ để tái lập số đo cũ).
Chạy demo: python task_b/rain.py
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# Hệ số suy giảm intensity (1/m) theo level. TODO: thay bằng mô hình vật lý alpha = 0.01 * R^0.6 (R mm/h).
ATTENUATION = {"light": 0.005, "moderate": 0.01, "heavy": 0.02}
NOISE_MIN_RANGE, NOISE_MAX_RANGE = 0.5, 10.0


def _noise_v1(points: np.ndarray, n_noise: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(NOISE_MIN_RANGE, NOISE_MAX_RANGE, n_noise)
    az = rng.uniform(-np.pi, np.pi, n_noise)
    el = np.radians(rng.uniform(-30.67, 10.67, n_noise))  # FOV dọc HDL-32E
    return np.c_[r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el),
                 rng.uniform(0, 5, n_noise), rng.integers(0, 32, n_noise)].astype(points.dtype)


def beam_noise(points: np.ndarray, idx: np.ndarray, rng: np.random.Generator,
               r_max: float = NOISE_MAX_RANGE) -> np.ndarray:
    """Điểm nhiễu trên đúng tia của points[idx]: dời dọc tia về r ~ U(0.5, min(r_max, r_gốc)), intensity thấp."""
    beams = points[idx].copy()
    d = np.maximum(np.linalg.norm(beams[:, :3], axis=1), 1e-3)
    r = rng.uniform(np.minimum(NOISE_MIN_RANGE, d), np.minimum(r_max, d))
    beams[:, :3] *= (r / d)[:, None]
    beams[:, 3] = rng.uniform(0, 5, len(idx))
    return beams


def simulate_rain(points: np.ndarray, intensity_level: str = "moderate",
                  rng: Optional[np.random.Generator] = None, return_labels: bool = False,
                  model: Optional[str] = None):
    """Mưa theo level trong config.RAIN_LEVELS: light / moderate / heavy. model: "v2" | "v1" (mặc định config).

    Returns augmented points (K, 5); nếu return_labels thì thêm labels (K,) 0 = gốc, 1 = noise.
    """
    if intensity_level not in config.RAIN_LEVELS:
        raise ValueError(f"intensity_level phải thuộc {list(config.RAIN_LEVELS)}")
    model = model or config.WEATHER_MODEL
    rng = rng or np.random.default_rng()
    p = config.RAIN_LEVELS[intensity_level]
    n_noise = int(p["noise_ratio"] * len(points))

    if model == "v1":
        kept = points[rng.random(len(points)) >= p["dropout"]].copy()
        noise = _noise_v1(points, n_noise, rng)
    elif model == "v2":
        dist = np.linalg.norm(points[:, :3], axis=1)
        hit = np.zeros(len(points), bool)  # tia trúng giọt mưa gần sensor: điểm gốc thay bằng điểm nhiễu
        hit[rng.choice(len(points), min(n_noise, len(points)), replace=False)] = True
        noise = beam_noise(points, np.flatnonzero(hit), rng)
        lost = rng.random(len(points)) < 1 - np.exp(-dist / p["drop_range"])
        kept = points[~hit & ~lost].copy()
    else:
        raise ValueError(f"model phải là 'v1' hoặc 'v2', nhận {model!r}")

    # (3) Suy giảm intensity theo khoảng cách (2 chiều): I' = I * exp(-2 * alpha * r)
    kept[:, 3] *= np.exp(-2 * ATTENUATION[intensity_level] * np.linalg.norm(kept[:, :3], axis=1))
    # TODO: range noise (jitter dọc tia) tăng theo khoảng cách và rain rate.

    out = np.concatenate([kept, noise], axis=0)
    if return_labels:
        return out, np.r_[np.zeros(len(kept), np.int8), np.ones(len(noise), np.int8)]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    pts = np.c_[rng.uniform(-50, 50, (30000, 2)), rng.uniform(-2, 1, 30000),
                rng.uniform(0, 255, 30000), rng.integers(0, 32, 30000)].astype(np.float32)
    for model in ("v1", "v2"):
        for level in config.RAIN_LEVELS:
            out, lab = simulate_rain(pts, level, rng, return_labels=True, model=model)
            print(f"{model} {level:9s}: {len(pts)} -> {len(out)} pts (noise {lab.sum()}), "
                  f"mean intensity {pts[:, 3].mean():.1f} -> {out[lab == 0, 3].mean():.1f}")
