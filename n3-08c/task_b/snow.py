"""Mô phỏng tuyết rơi trên LiDAR point cloud.

3 bước: (1) một phần beam trúng bông tuyết gần sensor -> điểm giả thay cho điểm gốc của beam đó
-> (2) dropout theo khoảng cách -> (3) suy giảm intensity theo khoảng cách.

GIẢ ĐỊNH (mô hình kinh nghiệm, CHƯA hiệu chỉnh với dữ liệu tuyết thật — nuScenes không có cảnh tuyết):
  - Mỗi beam chỉ có 1 return: beam trúng bông tuyết thì mất điểm gốc (giống backscatter ở fog.py).
  - Khoảng cách bông tuyết dọc tia ~ 0.5 + Gamma(2, FLAKE_RANGE_SCALE) m, tức tập trung trong vài mét
    quanh sensor, và không xa hơn điểm gốc.
  - Bông tuyết phản xạ mạnh hơn giọt mưa: intensity đều trong [0, FLAKE_INTENSITY_MAX].
  - Dropout và suy giảm intensity dùng cùng dạng hàm mũ với rain.py; tham số trong config.SNOW_LEVELS
    là giá trị tự chọn để 3 level tách biệt rõ.
Chạy demo: python task_b/snow.py
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

DROPOUT_REF_RANGE = 30.0     # khoảng cách (m) tại đó tỉ lệ mất điểm = dropout trong config
FLAKE_RANGE_SCALE = 2.0      # scale phân bố Gamma của khoảng cách bông tuyết (m)
FLAKE_INTENSITY_MAX = 15.0


def simulate_snow(points: np.ndarray, level: str = "moderate",
                  rng: Optional[np.random.Generator] = None, return_labels: bool = False,
                  labels: Optional[np.ndarray] = None):
    """Tuyết theo level trong config.SNOW_LEVELS: light / moderate / heavy.

    labels: nhãn đầu vào (N,) của points (khi nối sau weather khác); mặc định toàn LABEL_ORIGINAL.
    Returns augmented points (K, 5); nếu return_labels thì thêm labels (K,): điểm còn sống giữ nhãn
    đầu vào, bông tuyết = LABEL_NOISE.
    """
    if level not in config.SNOW_LEVELS:
        raise ValueError(f"level phải thuộc {list(config.SNOW_LEVELS)}")
    rng = rng or np.random.default_rng()
    p = config.SNOW_LEVELS[level]
    labels = (np.full(len(points), config.LABEL_ORIGINAL, np.int8) if labels is None
              else np.asarray(labels, np.int8))
    dist = np.linalg.norm(points[:, :3], axis=1)

    # (1) Beam trúng bông tuyết: điểm trả về nằm trên cùng tia, gần sensor (giữ nguyên ring)
    is_flake = rng.random(len(points)) < p["noise_ratio"]
    flakes = points[is_flake].copy()
    d_beam = np.maximum(dist[is_flake], 1e-3)
    r_flake = np.minimum(0.5 + rng.gamma(2.0, FLAKE_RANGE_SCALE, len(flakes)), d_beam)
    flakes[:, :3] *= (r_flake / d_beam)[:, None]
    flakes[:, 3] = rng.uniform(0, FLAKE_INTENSITY_MAX, len(flakes))

    # (2) Dropout theo khoảng cách cho các beam còn lại: survival = exp(-beta * d)
    beta = -np.log(1.0 - p["dropout"]) / DROPOUT_REF_RANGE
    keep = ~is_flake & (rng.random(len(points)) < np.exp(-beta * dist))
    kept = points[keep].copy()

    # (3) Suy giảm intensity 2 chiều: I' = I * exp(-2 * alpha * d)
    kept[:, 3] *= np.exp(-2 * p["attenuation"] * dist[keep])

    out = np.concatenate([kept, flakes], axis=0)
    if return_labels:
        return out, np.r_[labels[keep], np.full(len(flakes), config.LABEL_NOISE, np.int8)]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    pts = np.c_[rng.uniform(-50, 50, (30000, 2)), rng.uniform(-2, 1, 30000),
                rng.uniform(0, 255, 30000), rng.integers(0, 32, 30000)].astype(np.float32)
    for level in config.SNOW_LEVELS:
        out, lab = simulate_snow(pts, level, rng, return_labels=True)
        flake = lab == config.LABEL_NOISE
        print(f"{level:9s}: {len(pts)} -> {len(out)} pts (bông tuyết {flake.sum()}, "
              f"range TB {np.linalg.norm(out[flake, :3], axis=1).mean():.1f} m), "
              f"mean intensity {pts[:, 3].mean():.1f} -> {out[~flake, 3].mean():.1f}")
