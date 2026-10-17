"""Mô phỏng sương mù trên LiDAR point cloud.

2 bước: (1) exp-decay dropout, survival = exp(-d / visibility) -> (2) backscatter (tia phản xạ từ hạt sương gần sensor).
visibility (m): 200 = light, 50 = moderate, 20 = dense.

config.WEATHER_MODEL:
    v2 (mặc định)  backscatter THAY điểm gốc của cùng tia (mỗi tia chỉ trả về 1 điểm, điểm gần nhất là hạt sương)
    v1             bản cũ: backscatter cộng thêm, tia có 2 điểm (giữ để tái lập số đo cũ)
Chạy demo: python task_b/fog.py
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

BACKSCATTER_REF = 0.05  # tỉ lệ beam bị backscatter khi visibility = 20 m. TODO: hiệu chỉnh theo dữ liệu thật.


def simulate_fog(points: np.ndarray, visibility: float = 50.0,
                 rng: Optional[np.random.Generator] = None, return_labels: bool = False,
                 model: Optional[str] = None):
    """Sương mù với tầm nhìn visibility (m). model: "v2" | "v1" (mặc định config.WEATHER_MODEL).

    Returns augmented points (K, 5); nếu return_labels thì thêm labels (K,) 0 = gốc, 1 = backscatter.
    """
    if visibility <= 0:
        raise ValueError("visibility phải > 0")
    model = model or config.WEATHER_MODEL
    if model not in ("v1", "v2"):
        raise ValueError(f"model phải là 'v1' hoặc 'v2', nhận {model!r}")
    rng = rng or np.random.default_rng()
    dist = np.linalg.norm(points[:, :3], axis=1)

    # (1) Exp-decay dropout: điểm càng xa càng dễ mất
    survive = rng.random(len(points)) < np.exp(-dist / visibility)
    # TODO: phụ thuộc reflectivity — điểm intensity cao (biển báo, vạch kẻ) sống sót lâu hơn.

    # (2) Backscatter: một phần beam phản xạ từ hạt sương ở gần sensor, dọc theo chính tia đó
    ratio = min(1.0, BACKSCATTER_REF * 20.0 / visibility)
    idx = rng.choice(len(points), int(ratio * len(points)), replace=False)
    beams = points[idx]
    d_beam = np.maximum(dist[idx], 1e-3)
    r_scatter = np.minimum(rng.gamma(2.0, visibility / 20.0, len(idx)) + 1.0, d_beam)  # dọc tia, gần sensor
    scatter = beams.copy()
    scatter[:, :3] = beams[:, :3] * (r_scatter / d_beam)[:, None]
    scatter[:, 3] = rng.uniform(0, 5, len(idx))
    if model == "v2":
        survive[idx] = False  # tia đã trả về điểm backscatter: điểm gốc phía sau không còn

    kept = points[survive].copy()
    kept[:, 3] *= np.exp(-np.linalg.norm(kept[:, :3], axis=1) / visibility)  # intensity suy giảm

    out = np.concatenate([kept, scatter.astype(points.dtype)], axis=0)
    if return_labels:
        return out, np.r_[np.zeros(len(kept), np.int8), np.ones(len(scatter), np.int8)]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    pts = np.c_[rng.uniform(-50, 50, (30000, 2)), rng.uniform(-2, 1, 30000),
                rng.uniform(0, 255, 30000), rng.integers(0, 32, 30000)].astype(np.float32)
    for model in ("v1", "v2"):
        for level, vis in config.FOG_VISIBILITY.items():
            out, lab = simulate_fog(pts, vis, rng, return_labels=True, model=model)
            far = np.linalg.norm(out[lab == 0, :3], axis=1).max()
            print(f"{model} {level:9s} (V={vis:.0f} m): {len(pts)} -> {len(out)} pts "
                  f"(backscatter {lab.sum()}), max range kept {far:.1f} m")
