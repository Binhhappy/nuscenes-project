"""T5 — Thống kê thiếu hụt của nuScenes so với giao thông Việt Nam (pain point của đề tài).

Chỉ dùng metadata (annotation, attribute, ego pose, map), không đọc point cloud, nên chạy được cả
trên v1.0-trainval khi chỉ tải gói metadata (--version v1.0-trainval).

Hệ toạ độ ego: x hướng đầu xe, y sang trái. "Đang di chuyển" = attribute vehicle.moving,
cycle.with_rider (xe 2 bánh có người lái) hoặc pedestrian.moving.

Các chỉ số tình huống nguy hiểm là PROXY từ vị trí + hướng box ở 1 khung hình (không có vận tốc):
    cut_in      xe / 2 bánh đang chạy, ở phía trước trong hành lang của ego (0 < x < 15, |y| < 2),
                hướng lệch 20-70° so với ego -> đang cắt ngang / chen vào làn
    wrong_way   xe đang chạy, phía trước trong làn của ego (0 < x < 40, |y| < 1.5), hướng ngược > 150°
    jaywalk     người đi bộ (adult/child) đang đi, trên drivable_area, không ở ped_crossing / walkway /
                carpark_area (cần map expansion). Là cận trên: polygon drivable_area sát mép vỉa hè
    moto_swarm  >= 5 xe 2 bánh có người lái trong bán kính 15 m

Chạy:
    python tools/dataset_gap.py                          # v1.0-mini
    python tools/dataset_gap.py --version v1.0-trainval  # cần v1.0-trainval metadata trong DATAROOT
    python tools/dataset_gap.py --no-map                 # bỏ jaywalk (nhanh hơn)
Output: <OUTPUT_DIR>/dataset_gap_<version>.json + bảng in ra màn hình (dán vào báo cáo).
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.nuscenes_loader import get_ego_pose, get_lidar_token, load_nuscenes  # noqa: E402

TWO_WHEEL = ("vehicle.motorcycle", "vehicle.bicycle")
ROAD_USER = ("vehicle.", "human.pedestrian.")
WALKERS = ("human.pedestrian.adult", "human.pedestrian.child")
NOT_ROAD = ("ped_crossing", "walkway", "carpark_area")


def _attrs(nusc, ann) -> set:
    return {nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]}


def _is_moving(cat: str, attrs: set) -> bool:
    if cat.startswith(TWO_WHEEL):
        return "cycle.with_rider" in attrs
    if cat.startswith("vehicle."):
        return "vehicle.moving" in attrs
    return "pedestrian.moving" in attrs


def sample_objects(nusc, sample) -> list:
    """Object của 1 keyframe trong hệ ego: dict category, x, y, dist, dyaw (độ, so với hướng ego), moving."""
    pose = get_ego_pose(nusc, get_lidar_token(nusc, sample["token"]))
    q_inv = Quaternion(pose["rotation"]).inverse
    ego_yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
    out = []
    for tok in sample["anns"]:
        a = nusc.get("sample_annotation", tok)
        cat = a["category_name"]
        if not cat.startswith(ROAD_USER):
            continue
        x, y, _ = q_inv.rotate(np.asarray(a["translation"]) - np.asarray(pose["translation"]))
        dyaw = np.degrees((Quaternion(a["rotation"]).yaw_pitch_roll[0] - ego_yaw + np.pi) % (2 * np.pi) - np.pi)
        out.append({"category": cat, "x": float(x), "y": float(y), "dist": float(np.hypot(x, y)),
                    "dyaw": float(dyaw), "moving": _is_moving(cat, _attrs(nusc, a)),
                    "global_xy": a["translation"][:2]})
    return out


def _jaywalk_checker(nusc, use_map: bool):
    """Hàm (sample, obj) -> True nếu người đi bộ đang đi trên lòng đường, ngoài vạch / vỉa hè / bãi đỗ."""
    if not use_map:
        return None
    from nuscenes.map_expansion.map_api import NuScenesMap

    maps = {}

    def check(sample, obj) -> bool:
        if not (obj["moving"] and obj["category"].startswith(WALKERS)):
            return False
        loc = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        if loc not in maps:
            maps[loc] = NuScenesMap(dataroot=nusc.dataroot, map_name=loc)
        on = maps[loc].layers_on_point(*obj["global_xy"], layer_names=list(NOT_ROAD) + ["drivable_area"])
        return bool(on["drivable_area"]) and not any(on[k] for k in NOT_ROAD)
    return check


def analyse(nusc, use_map: bool = True) -> dict:
    from tqdm import tqdm

    jaywalk_check = _jaywalk_checker(nusc, use_map)
    cat_count = Counter(a["category_name"] for a in nusc.sample_annotation)
    two_wheel_attr = Counter()
    for a in nusc.sample_annotation:
        if a["category_name"].startswith(TWO_WHEEL):
            two_wheel_attr["moving" if "cycle.with_rider" in _attrs(nusc, a) else "parked"] += 1

    per_sample = []
    for s in tqdm(nusc.sample, desc="samples"):
        objs = sample_objects(nusc, s)
        moving = [o for o in objs if o["moving"] and not o["category"].startswith("human.")]
        tw_moving = [o for o in moving if o["category"].startswith(TWO_WHEEL)]
        peds = [o for o in objs if o["category"].startswith("human.") and o["dist"] < 40]
        per_sample.append({
            "two_wheel_20m": sum(o["category"].startswith(TWO_WHEEL) and o["dist"] < 20 for o in objs),
            "two_wheel_moving_15m": sum(o["dist"] < 15 for o in tw_moving),
            "nearest_moving": min((o["dist"] for o in moving), default=np.inf),
            "cut_in": sum(0 < o["x"] < 15 and abs(o["y"]) < 2 and 20 <= abs(o["dyaw"]) <= 70 for o in moving),
            "wrong_way": sum(0 < o["x"] < 40 and abs(o["y"]) < 1.5 and abs(o["dyaw"]) > 150 for o in moving),
            "jaywalk": sum(jaywalk_check(s, o) for o in peds) if jaywalk_check else None,
        })

    n = len(per_sample)

    def frac(key, cond):
        return sum(cond(p[key]) for p in per_sample) / n

    scenes = nusc.scene
    desc = [sc["description"].lower() for sc in scenes]
    total = sum(cat_count.values())
    report = {
        "version": nusc.version,
        "n_scenes": len(scenes),
        "n_samples": n,
        "n_annotations": total,
        "class_share": {c: {"count": k, "share": k / total} for c, k in cat_count.most_common()},
        "two_wheelers": {
            "count": sum(cat_count[c] for c in TWO_WHEEL),
            "share": sum(cat_count[c] for c in TWO_WHEEL) / total,
            "moving_with_rider": two_wheel_attr["moving"],
            "parked_or_no_rider": two_wheel_attr["parked"],
            "samples_with_any_within_20m": frac("two_wheel_20m", lambda v: v >= 1),
            "samples_moto_swarm_ge5_moving_15m": frac("two_wheel_moving_15m", lambda v: v >= 5),
            "max_moving_within_15m": int(max(p["two_wheel_moving_15m"] for p in per_sample)),
        },
        "close_range": {
            "samples_nearest_moving_lt_3m": frac("nearest_moving", lambda v: v < 3),
            "samples_nearest_moving_lt_5m": frac("nearest_moving", lambda v: v < 5),
            "median_nearest_moving_m": float(np.median([p["nearest_moving"] for p in per_sample
                                                        if np.isfinite(p["nearest_moving"])])),
        },
        "risky_proxies": {
            "samples_with_cut_in": frac("cut_in", lambda v: v >= 1),
            "samples_with_wrong_way": frac("wrong_way", lambda v: v >= 1),
            "samples_with_jaywalk": frac("jaywalk", lambda v: bool(v)) if jaywalk_check else None,
        },
        "conditions": {
            "scenes_rain": sum("rain" in d for d in desc),
            "scenes_night": sum("night" in d for d in desc),
            "scenes_total": len(scenes),
        },
    }
    return report


def print_report(r: dict) -> None:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    print(f"\n## Thiếu hụt của nuScenes {r['version']}: {r['n_scenes']} scene, {r['n_samples']} keyframe, "
          f"{r['n_annotations']} box\n")
    print("| Lớp | Số box | Tỉ lệ |\n|---|---|---|")
    for c, v in list(r["class_share"].items())[:12]:
        print(f"| {c} | {v['count']} | {pct(v['share'])} |")
    tw, cr, rp, cd = r["two_wheelers"], r["close_range"], r["risky_proxies"], r["conditions"]
    print("\n| Chỉ số | Giá trị |\n|---|---|")
    rows = [
        ("Xe 2 bánh / tổng box", pct(tw["share"])),
        ("Xe 2 bánh có người lái / đỗ hoặc không người", f"{tw['moving_with_rider']} / {tw['parked_or_no_rider']}"),
        ("Keyframe có >= 1 xe 2 bánh trong 20 m", pct(tw["samples_with_any_within_20m"])),
        ("Keyframe có bầy >= 5 xe 2 bánh đang chạy trong 15 m", pct(tw["samples_moto_swarm_ge5_moving_15m"])),
        ("Nhiều nhất xe 2 bánh đang chạy trong 15 m (1 keyframe)", tw["max_moving_within_15m"]),
        ("Keyframe có xe đang chạy cách ego < 3 m", pct(cr["samples_nearest_moving_lt_3m"])),
        ("Keyframe có xe đang chạy cách ego < 5 m", pct(cr["samples_nearest_moving_lt_5m"])),
        ("Median khoảng cách tới xe đang chạy gần nhất", f"{cr['median_nearest_moving_m']:.1f} m"),
        ("Keyframe có xe cắt đầu (proxy)", pct(rp["samples_with_cut_in"])),
        ("Keyframe có xe ngược chiều trong làn ego (proxy, cần xem lại bằng mắt)", pct(rp["samples_with_wrong_way"])),
        ("Keyframe có người đi bộ giữa đường, ngoài vạch (proxy, cận trên)", pct(rp["samples_with_jaywalk"])),
        ("Scene mưa / đêm / tổng", f"{cd['scenes_rain']} / {cd['scenes_night']} / {cd['scenes_total']}"),
    ]
    for k, v in rows:
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=config.VERSION)
    ap.add_argument("--no-map", action="store_true", help="bỏ thống kê jaywalk (không cần map expansion)")
    args = ap.parse_args()

    rep = analyse(load_nuscenes(args.version), use_map=not args.no_map)
    print_report(rep)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, f"dataset_gap_{args.version}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("\nsaved", out)
