"""Vật thể VN không có trong nuScenes (xe ba gác, xe máy chở hàng, động vật...): dựng hình từ khối cơ bản theo
kích thước công bố / giả định, rồi bắn tia theo đúng LiDAR nuScenes thay vì cắt mẫu.

Cảm biến: Velodyne HDL-32E (LIDAR_TOP) — 32 tia dọc, góc ngước theo thông số (HDL32E_ELEVATIONS, 1.33°),
~1084 bước azimuth / vòng. Mỗi tia lấy điểm chạm gần nhất => mật độ theo khoảng cách, mặt nhìn thấy, tự che
(vd người ngồi sau bị người trước che) tự đúng theo cảm biến. Che khuất với cảnh và xoá điểm cũ trong box dùng lại
merge.resolve_occlusion / points_in_new_box như object cắt mẫu.

Hình: danh sách primitive trong object frame (x hướng đầu, y sang trái, z lên, gốc = tâm box bao, đáy z = -H/2):
    ("box", tâm (3,), kích thước l, w, h, độ phản xạ)
    ("ell", tâm (3,), bán trục rx, ry, rz, độ phản xạ)
Độ phản xạ = intensity trung bình (thang nuScenes 0-255), thêm nhiễu khi bắn.

Chạy demo / kiểm chứng: python task_a/raycast.py   (số điểm theo khoảng cách: ô tô / xe máy ray-cast vs object thật)
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import project_z_to_ground  # noqa: E402
from task_a.merge import points_in_new_box, resolve_occlusion  # noqa: E402
from utils.transforms import yaw_to_quaternion  # noqa: E402

HDL32E_ELEVATIONS = np.radians(np.linspace(-30.67, 10.67, 32))  # 32 tia đều 1.33° (thông số Velodyne HDL-32E)
RANGE_NOISE = 0.02     # (m) sai số đo khoảng cách HDL-32E ~ 2 cm
DROPOUT = 0.03         # tỉ lệ tia không trả về (bề mặt xiên, vật liệu tối)
# Intensity tham chiếu (thang 0-255 nuScenes). Sơn xe ~ trung bình, vải / da người thấp hơn, lông động vật thấp.
REFL = {"metal": 30.0, "paint": 25.0, "cloth": 10.0, "fur": 6.0, "wood": 12.0, "tire": 4.0, "plastic": 15.0,
        "glass": 5.0}
# Tỉ lệ tia không trả về theo vật liệu: kính gần như trong suốt / phản xạ gương với laser 905 nm.
DROP = {"glass": 0.8, "tire": 0.3, "fur": 0.1}


def _box(c, l, w, h, mat):
    return ("box", np.asarray(c, float), np.array([l, w, h], float), REFL[mat], DROP.get(mat, 0.0))


def _ell(c, rx, ry, rz, mat):
    return ("ell", np.asarray(c, float), np.array([rx, ry, rz], float), REFL[mat], DROP.get(mat, 0.0))


# ----- hình các vật thể (z tính từ mặt đất, chuyển về gốc tâm box ở build_model) -----
def _wheels(x_front, x_back, r=0.3, y=0.0, thick=0.06):
    return [_ell((x, y, r), r, thick, r, "tire") for x in (x_front, x_back)]


def _rider(x, seat_z=0.75, mat="cloth"):
    return [_box((x + 0.15, 0, seat_z - 0.25), 0.35, 0.45, 0.45, mat),       # chân
            _ell((x, 0, seat_z + 0.32), 0.17, 0.22, 0.32, mat),              # thân
            _ell((x + 0.03, 0, seat_z + 0.78), 0.12, 0.11, 0.13, "plastic")]  # đầu + mũ bảo hiểm


def _moto_frame():  # xe số / tay ga ~ Honda Wave Alpha 1.914 x 0.688 x 1.075 m, Vision 1.871 x 0.686 x 1.101 m
    # Bản đầu thân rộng 0.34 m -> detector nhận 61% thành xe đạp (đối chứng ray-cast 34% vs cắt mẫu 94%).
    # Sửa theo thân nhựa thật của xe tay ga / xe số VN: ốp thân, yếm, sàn để chân rộng ~0.55-0.65 m.
    return (_wheels(0.66, -0.66, 0.28, 0.0, 0.08)
            + [_box((-0.15, 0, 0.58), 0.95, 0.58, 0.42, "plastic"),      # ốp thân + yên
               _box((0.18, 0, 0.30), 0.55, 0.45, 0.12, "plastic"),       # sàn để chân
               _box((0.48, 0, 0.62), 0.22, 0.62, 0.62, "plastic"),       # yếm (leg shield)
               _box((0.60, 0, 0.98), 0.25, 0.45, 0.18, "plastic"),       # đầu đèn
               _box((0.58, 0, 1.06), 0.08, 0.69, 0.06, "metal")])        # tay lái


def shape(name: str, rng: Optional[np.random.Generator] = None) -> List[tuple]:
    """Primitive (z tính từ mặt đất) của vật thể; rng cho biến thể kích thước hàng hoá / số người."""
    rng = rng or np.random.default_rng(0)
    if name == "car":  # ô tô 4-5 chỗ ~ Toyota Vios 4.425 x 1.730 x 1.475 m
        return (_wheels(1.35, -1.35, 0.31, 0.75, 0.10) + _wheels(1.35, -1.35, 0.31, -0.75, 0.10)
                + [_box((0, 0, 0.62), 4.40, 1.72, 0.66, "paint"),
                   _box((-0.25, 0, 1.18), 2.30, 1.50, 0.58, "glass"),       # cabin: phần lớn là kính
                   _box((-0.25, 0, 1.45), 2.00, 1.40, 0.05, "paint")])      # nóc
    if name == "moto_1":
        return _moto_frame() + _rider(-0.10)
    if name == "moto_multi":  # 2-3 người ngồi nối tiếp
        n = int(rng.integers(2, 4))
        return _moto_frame() + sum((_rider(x) for x in (0.0, -0.40, -0.75)[:n]), [])
    if name == "moto_cargo":  # hàng cồng kềnh buộc sau yên: rộng 1.0-1.8 m, cao tới ~2 m
        w, h, l = rng.uniform(1.0, 1.8), rng.uniform(0.7, 1.1), rng.uniform(0.7, 1.1)
        return _moto_frame() + _rider(0.05) + [_box((-0.55, 0, 0.85 + h / 2), l, w, h, "cloth")]
    if name == "ba_gac":  # ~ ô tô 4 chỗ (giả định nhóm): đầu xe máy + thùng hở 2.6 x 1.7, hàng chất trên thùng
        hh = rng.uniform(0.3, 0.9)
        return (_wheels(1.65, 1.65) + _wheels(-1.2, -1.2, 0.28, 0.75) + _wheels(-1.2, -1.2, 0.28, -0.75)
                + _rider(1.25) + [_box((1.45, 0, 0.55), 0.8, 0.34, 0.4, "plastic"),
                                  _box((-0.6, 0, 0.62), 2.6, 1.70, 0.12, "metal"),      # sàn thùng
                                  _box((-0.6, 0.83, 0.85), 2.6, 0.04, 0.35, "metal"),   # thành trái
                                  _box((-0.6, -0.83, 0.85), 2.6, 0.04, 0.35, "metal"),  # thành phải
                                  _box((-0.6, 0, 0.68 + hh / 2), 2.2, 1.5, hh, "cloth")])
    if name == "moto_tow":  # xe máy kéo thùng hàng 1.8 x 1.2 x 1.0 (1 box chung)
        return ([_box(p[1] + [1.0, 0, 0], *p[2], "plastic") if p[0] == "box" else
                 _ell(p[1] + [1.0, 0, 0], *p[2], "tire") for p in _moto_frame()]
                + _rider(0.9) + _wheels(-0.9, -0.9, 0.25, 0.55) + _wheels(-0.9, -0.9, 0.25, -0.55)
                + [_box((-0.9, 0, 0.75), 1.8, 1.2, 0.95, "wood")])
    if name == "vendor_cart":  # xe đẩy hàng rong 1.5 x 0.8 + mái che + người đẩy
        return (_wheels(0.5, -0.5, 0.2, 0.35) + _wheels(0.5, -0.5, 0.2, -0.35)
                + [_box((0, 0, 0.75), 1.5, 0.8, 0.5, "wood"), _box((0, 0, 1.75), 1.6, 1.0, 0.05, "plastic"),
                   _box((0.7, 0.45, 1.25), 0.04, 0.04, 1.0, "metal"), _box((-0.7, -0.45, 1.25), 0.04, 0.04, 1.0, "metal"),
                   _ell((-1.05, 0, 1.05), 0.17, 0.22, 0.32, "cloth"), _ell((-1.05, 0, 1.52), 0.11, 0.11, 0.12, "cloth"),
                   _box((-1.05, 0, 0.42), 0.25, 0.35, 0.84, "cloth")])
    if name == "dog":  # 0.9 x 0.3 x 0.6
        return ([_ell((0, 0, 0.42), 0.33, 0.13, 0.14, "fur"), _ell((0.40, 0, 0.52), 0.11, 0.09, 0.09, "fur")]
                + [_box((x, y, 0.16), 0.06, 0.06, 0.32, "fur") for x in (0.22, -0.22) for y in (0.08, -0.08)])
    if name == "cow":  # trâu / bò 2.2 x 0.7 x 1.4
        return ([_ell((0, 0, 1.0), 0.85, 0.33, 0.36, "fur"), _ell((0.95, 0, 1.15), 0.22, 0.14, 0.18, "fur")]
                + [_box((x, y, 0.33), 0.12, 0.12, 0.66, "fur") for x in (0.6, -0.6) for y in (0.18, -0.18)])
    if name == "ox_wagon":  # xe gỗ do bò kéo 2.5 x 1.5 x 1.2 (box riêng, như trailer của nuScenes)
        return (_wheels(0.0, 0.0, 0.5, 0.75, 0.08) + _wheels(0.0, 0.0, 0.5, -0.75, 0.08)
                + [_box((0, 0, 0.75), 2.5, 1.4, 0.10, "wood"), _box((0, 0, 0.95), 2.5, 1.5, 0.35, "wood"),
                   _box((1.6, 0, 0.75), 0.9, 0.08, 0.06, "wood")])
    raise KeyError(name)


SHAPES = ("car", "moto_1", "moto_multi", "moto_cargo", "ba_gac", "moto_tow", "vendor_cart", "dog", "cow", "ox_wagon")


def build_model(name: str, rng: Optional[np.random.Generator] = None, category: Optional[str] = None) -> dict:
    """Template ray-cast: primitive ở object frame gốc tâm box bao + size [w, l, h] như annotation nuScenes."""
    prims = shape(name, rng)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for kind, c, s, _, _ in prims:
        half = s / 2 if kind == "box" else s
        lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
    lo[2] = 0.0  # đáy box = mặt đất
    center = (lo + hi) / 2
    size_lwh = hi - lo
    prims = [(k, c - center, s, refl, drop) for k, c, s, refl, drop in prims]
    return {"name": name, "category": category or f"vn.{name}", "primitives": prims,
            "size": [float(size_lwh[1]), float(size_lwh[0]), float(size_lwh[2])],
            "ann_token": f"raycast_{name}_{id(prims)}", "points": np.zeros((0, 5), np.float32), "distance": None}


def _hit_box(o, d, size):
    """Slab method: t vào (hoặc inf) cho tia o + t d với box tâm 0, kích thước size (l, w, h)."""
    half = size / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmax > 0), np.where(tmin > 0, tmin, np.inf), np.inf)


def _hit_ell(o, d, radii):
    o, d = o / radii, d / radii
    a = (d * d).sum(1)
    b = 2 * (o * d).sum(1)
    c = (o * o).sum(-1) - 1  # o: gốc tia chung (3,)
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def raycast(model: dict, center: Sequence[float], yaw: float, ring_table: np.ndarray,
            rng: Optional[np.random.Generator] = None, n_az: int = config.LIDAR_AZIMUTH_BINS) -> np.ndarray:
    """Bắn tia từ gốc sensor vào model đặt tại center (tâm box, sensor frame), hướng yaw.

    Returns (K, 5) [x, y, z, intensity, ring] ở sensor frame.
    """
    rng = rng or np.random.default_rng()
    center = np.asarray(center, float)
    w, l, h = model["size"]
    # chỉ bắn các azimuth bao được object (bán kính bao + lề)
    r_xy = np.hypot(*center[:2])
    half_ang = np.arctan2(np.hypot(l, w) / 2 + 0.2, max(r_xy, 0.5))
    az0 = np.arctan2(center[1], center[0])
    step = 2 * np.pi / n_az
    az = (np.round(az0 / step) + np.arange(-int(half_ang / step) - 1, int(half_ang / step) + 2)) * step
    ring_idx, az_g = np.meshgrid(np.arange(len(ring_table)), az, indexing="ij")
    el = ring_table[ring_idx.ravel()]
    a = az_g.ravel()
    dirs = np.c_[np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]
    # tia sang object frame
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o_loc = (np.zeros(3) - center) @ R
    d_loc = dirs @ R
    best_t = np.full(len(dirs), np.inf)
    best_refl, best_drop = np.zeros(len(dirs)), np.zeros(len(dirs))
    for kind, pc, size, refl, drop in model["primitives"]:
        t = (_hit_box if kind == "box" else _hit_ell)(o_loc - pc, d_loc, size)
        closer = t < best_t
        best_t[closer], best_refl[closer], best_drop[closer] = t[closer], refl, drop
    # tia chạm kính có thể xuyên qua (không trả về); đơn giản hoá: bỏ tia, không bắn tiếp vào nội thất
    hit = np.isfinite(best_t) & (rng.random(len(dirs)) >= np.maximum(DROPOUT, best_drop))
    t = best_t[hit] + rng.normal(0, RANGE_NOISE, hit.sum())
    pts = dirs[hit] * t[:, None]
    inten = np.clip(best_refl[hit] * rng.lognormal(0, 0.35, hit.sum()), 0, 255)
    return np.c_[pts, inten, ring_idx.ravel()[hit]].astype(np.float32)


def insert_raycast(target_points: np.ndarray, model: dict, position: Tuple[float, float, float], ground,
                   rng: Optional[np.random.Generator] = None, **_) -> Tuple[np.ndarray, dict]:
    """Như merge.insert_object nhưng điểm object sinh bằng ray-cast tại đúng vị trí (không cần chỉnh mật độ)."""
    x, y, yaw = position
    h = model["size"][2]
    center = [x, y, project_z_to_ground(x, y, ground) + h / 2]
    # Góc tia theo thông số HDL-32E, KHÔNG suy từ scan: cột ring của nuScenes không sạch ở ~9% scan mini_val
    # (merge.ring_elevations lệch tới 16.7°, nhiều ring trùng góc) -> bắn dồn tia, vd chó 96 điểm thay vì ~12.
    placed = raycast(model, center, yaw, HDL32E_ELEVATIONS, rng).astype(target_points.dtype)
    box = {"translation": center, "size": model["size"], "rotation": yaw_to_quaternion(yaw)}
    in_box = points_in_new_box(target_points, box)
    target = target_points[~in_box]
    merged = np.concatenate([target, placed], axis=0)
    is_obj = np.r_[np.zeros(len(target), bool), np.ones(len(placed), bool)]
    keep = resolve_occlusion(merged, is_obj)
    info = {"n_template": int(len(placed)), "n_density_dropped": 0, "n_box_cleared": int(in_box.sum()),
            "n_target_occluded": int((~keep & ~is_obj).sum()), "n_object_occluded": int((~keep & is_obj).sum()),
            "n_visible": int((keep & is_obj).sum())}
    return merged[keep], info


def beam_table(n_rings: int, fov=(-30.67, 10.67)) -> np.ndarray:
    """Bảng góc ngước đều cho LiDAR giả định n_rings tia (so sánh 32 / 64 / 128 tia)."""
    return np.radians(np.linspace(fov[0], fov[1], n_rings))


def points_vs_distance(name: str, distances: Sequence[float], ring_table: np.ndarray, n_az: int = 1084,
                       sensor_height: float = 1.84, rng=None) -> List[int]:
    """Số điểm trên object (đứng thẳng trước sensor, nhìn nghiêng 45°) theo khoảng cách, mặt đất z = -sensor_height."""
    rng = rng or np.random.default_rng(0)
    m = build_model(name, np.random.default_rng(1))
    out = []
    for r in distances:
        pts = raycast(m, [r, 0, -sensor_height + m["size"][2] / 2], np.radians(45), ring_table, rng, n_az)
        out.append(int(len(pts)))
    return out


if __name__ == "__main__":
    import json
    import os

    # Kiểm chứng: số điểm ô tô / xe máy ray-cast vs object thật cùng khoảng cách (bank nuScenes mini)
    with open(os.path.join(config.TEMPLATE_DIR, "bank.json"), encoding="utf-8") as f:
        bank = json.load(f)
    hdl32 = np.radians(np.linspace(-30.67, 10.67, 32))
    bins = [(3, 7), (7, 12), (12, 20), (20, 30), (30, 45)]
    for name, cat in (("car", "vehicle.car"), ("moto_1", "vehicle.motorcycle")):
        real = [(m["distance"], m["num_points"]) for m in bank if m["category"] == cat
                and (cat != "vehicle.motorcycle" or "cycle.with_rider" in m["attributes"])]
        rc = points_vs_distance(name, [np.mean(b) for b in bins], hdl32)
        print(f"{name}: (khoảng cách | object thật: median / p75 (n) | ray-cast)")
        for (lo, hi), n_rc in zip(bins, rc):
            pts = [p for d, p in real if lo <= d < hi]
            med, p75 = (np.median(pts), np.percentile(pts, 75)) if pts else (float("nan"),) * 2
            print(f"  {lo:2d}-{hi:2d} m | {med:6.0f} / {p75:6.0f} ({len(pts):3d}) | {n_rc:5d}")
