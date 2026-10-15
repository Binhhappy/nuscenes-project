"""Pipeline Task A end-to-end: source_token -> target_token -> augmented scene.

Output (config.AUGMENTED_DIR):
    <target_sample_token>.bin   point cloud (N+M, 5) float32, sensor frame (cùng format nuScenes)
    <target_sample_token>.json  annotations mới (sensor + global frame), quality (chỉ số vật lý của từng
                                object chèn), reject_stats + thống kê cho evaluate.py

Chạy:
    python task_a/run_insertion.py                                   # demo: sample[0] -> sample[10]
    python task_a/run_insertion.py --source-ann <ann_token> --target <sample_token>
    python task_a/run_insertion.py --source-sample <tok> --target <tok> --num-objects 3 --vis
    python task_a/run_insertion.py --from-bank --categories vehicle.motorcycle --num-objects 5
    python task_a/run_insertion.py --naive --out-dir ../output/augmented_naive   # pipeline cũ để so sánh
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.extract_object import (TemplateBank, extract_all_templates, extract_object_template,  # noqa: E402
                                   has_required_attributes)
from task_a.ground_plane import (GroundModel, get_map_area, layers_for_category,  # noqa: E402
                                 load_map_for_sample)
from task_a.merge import create_annotation, insert_object, points_in_new_box  # noqa: E402
from task_a.placement import sample_position_for_template, sample_valid_position  # noqa: E402
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose,  # noqa: E402
                                   get_lidar_token, get_pointcloud, load_nuscenes, save_bin)
from utils.transforms import box_sensor_to_global, to_local_frame  # noqa: E402


def collect_templates(nusc, source_sample: Optional[str], source_anns: Optional[List[str]],
                      num_objects: int, categories: Sequence[str] = ("vehicle",),
                      rng: Optional[np.random.Generator] = None) -> List[dict]:
    """Template từ annotation chỉ định, hoặc num_objects object ngẫu nhiên của source_sample.

    Chọn ngẫu nhiên (không lấy object nhiều điểm nhất): chọn theo số điểm thiên về object dày bất thường
    so với khoảng cách, làm mật độ object chèn cao hơn object thật. Bỏ xe 2 bánh không người lái.
    """
    if source_anns:
        return [extract_object_template(nusc, nusc.get("sample_annotation", a)["sample_token"], a)
                for a in source_anns]
    rng = rng or np.random.default_rng()
    templates = [t for t in extract_all_templates(nusc, source_sample, categories=list(categories))
                 if has_required_attributes(t)]
    idx = rng.permutation(len(templates))[:num_objects]
    return [templates[i] for i in idx]


def collect_from_bank(nusc, target_token: str, num_objects: int, categories: Sequence[str] = ("vehicle",),
                      min_points: int = 50, rng: Optional[np.random.Generator] = None) -> List[dict]:
    """num_objects template từ TemplateBank (cân bằng category), bỏ template thuộc chính scene đích
    và xe 2 bánh không người lái (config.TEMPLATE_REQUIRED_ATTRIBUTES)."""
    bank = TemplateBank().load_or_build(nusc)
    scene = nusc.get("sample", target_token)["scene_token"]
    return bank.sample(num_objects, rng, categories=list(categories), min_points=min_points,
                       exclude_scenes=[scene], required_attributes=config.TEMPLATE_REQUIRED_ATTRIBUTES)


def facing_cos(object_points: np.ndarray, box: dict) -> float:
    """cos giữa (tâm các điểm object) và (hướng về sensor), trong object frame.

    > 0: mặt có điểm quay về phía LiDAR (đúng vật lý). < 0: điểm nằm ở mặt LiDAR không thể thấy.
    """
    if len(object_points) == 0:
        return float("nan")
    loc = to_local_frame(object_points[:, :3], box["translation"], box["rotation"])[:, :2].mean(0)
    sensor = to_local_frame(np.zeros((1, 3)), box["translation"], box["rotation"])[0, :2]
    return float(loc @ sensor / (np.linalg.norm(loc) * np.linalg.norm(sensor) + 1e-9))


def run_insertion(nusc, target_token: str, templates: List[dict], n_attempts: int = 100,
                  rng: Optional[np.random.Generator] = None, physical: bool = True,
                  max_tries: int = 5) -> dict:
    """Chèn lần lượt các template vào target sample. Trả về dict points + annotations + thống kê.

    physical=True: đặt bằng sample_position_for_template (giữ khoảng cách + góc nhìn), yêu cầu thấy mặt đất,
    xoá điểm trong box, xử lý che khuất; object còn < MIN_VISIBLE_POINTS điểm thì thử vị trí khác.
    physical=False: pipeline cũ (r, yaw ngẫu nhiên, chỉ concat) để so sánh.
    """
    rng = rng or np.random.default_rng(config.SEED)
    t0 = time.perf_counter()

    points = get_pointcloud(nusc, target_token)
    existing = get_annotations(nusc, target_token)
    ground = GroundModel.from_points(points, rng=rng)
    nusc_map = load_map_for_sample(nusc, target_token)
    areas = {}  # cache vùng đặt object theo bộ layer
    rejects = {}

    new_anns, quality = [], []
    for tpl in templates:
        layers = layers_for_category(tpl["category"])
        if layers not in areas:
            areas[layers] = get_map_area(nusc, nusc_map, target_token, layers)
        for _ in range(max_tries if physical else 1):
            if physical:
                pos = sample_position_for_template(areas[layers], existing + new_anns, tpl, ground, points,
                                                   n_attempts, rng, stats=rejects)
            else:
                pos = sample_valid_position(areas[layers], existing + new_anns, tpl["size"], n_attempts, rng)
            if pos is None:
                break
            aug, info = insert_object(points, tpl, pos, ground, rng, physical=physical)
            if physical and info["n_visible"] < config.MIN_VISIBLE_POINTS:
                rejects["occluded"] = rejects.get("occluded", 0) + 1
                continue
            points = aug
            z, support = ground.height_at(pos[0], pos[1])
            ann = create_annotation(tpl, pos, float(z[0]), info["n_visible"])
            new_anns.append(ann)
            quality.append({
                "source_distance": tpl.get("distance"),
                "distance": float(np.hypot(pos[0], pos[1])),
                "ground_support": int(support[0]),
                "facing_cos": facing_cos(points[len(points) - info["n_visible"]:], ann),
                **info,
            })
            break
    if physical:  # object chèn sau có thể che object chèn trước: đếm lại điểm trong từng box
        for ann, q in zip(new_anns, quality):
            ann["num_lidar_pts"] = q["n_visible_final"] = int(points_in_new_box(points, ann).sum())
    drivable = areas.get(config.DEFAULT_LAYERS) or get_map_area(nusc, nusc_map, target_token)

    sd_token = get_lidar_token(nusc, target_token)
    calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)
    return {
        "points": points,
        "target_sample_token": target_token,
        "physical": physical,
        "source_ann_tokens": [t.get("ann_token") for t in templates],
        "n_requested": len(templates),
        "n_inserted": len(new_anns),
        "elapsed_s": time.perf_counter() - t0,
        "reject_stats": rejects,
        "annotations_sensor": new_anns,
        "annotations_global": [box_sensor_to_global(a, calib, pose) for a in new_anns],
        "quality": quality,
        "existing_annotations": existing,
        "drivable": drivable,
    }


def save_result(result: dict, out_dir: str = config.AUGMENTED_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, result["target_sample_token"])
    save_bin(stem + ".bin", result["points"])
    meta = {k: v for k, v in result.items() if k not in ("points", "drivable", "existing_annotations")}
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return stem


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-sample", help="sample lấy object (mặc định sample[0])")
    ap.add_argument("--source-ann", nargs="*", help="annotation token cụ thể cần chèn")
    ap.add_argument("--target", help="sample token đích (mặc định sample[10])")
    ap.add_argument("--num-objects", type=int, default=3)
    ap.add_argument("--n-attempts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--categories", nargs="*", default=["vehicle"], help="prefix category của template")
    ap.add_argument("--from-bank", action="store_true", help="lấy template từ TemplateBank thay vì source sample")
    ap.add_argument("--naive", action="store_true", help="pipeline cũ (r, yaw ngẫu nhiên, không che khuất) để so sánh")
    ap.add_argument("--out-dir", default=config.AUGMENTED_DIR)
    ap.add_argument("--vis", action="store_true", help="mở Open3D sau khi chèn")
    args = ap.parse_args()

    nusc = load_nuscenes()
    source = args.source_sample or nusc.sample[0]["token"]
    target = args.target or nusc.sample[10]["token"]

    rng = np.random.default_rng(args.seed)
    if args.from_bank and not args.source_ann:
        templates = collect_from_bank(nusc, target, args.num_objects, args.categories, rng=rng)
    else:
        templates = collect_templates(nusc, source, args.source_ann, args.num_objects, args.categories, rng)
    result = run_insertion(nusc, target, templates, args.n_attempts, rng, physical=not args.naive)
    stem = save_result(result, args.out_dir)

    print(f"target {target}: chèn {result['n_inserted']}/{result['n_requested']} objects "
          f"trong {result['elapsed_s']:.2f}s")
    for a, q in zip(result["annotations_sensor"], result["quality"]):
        print(f"  {a['category']:25s} at ({a['translation'][0]:.1f}, {a['translation'][1]:.1f}) "
              f"{a['num_lidar_pts']} pts (template {q['n_template']}, r {q['source_distance']:.1f} -> "
              f"{q['distance']:.1f} m, facing_cos {q['facing_cos']:+.2f}, ground support {q['ground_support']})")
    print("reject:", result["reject_stats"])
    print("saved", stem + ".bin/.json")

    from utils.visualization import plot_bev, show_open3d
    png = os.path.join(config.VIS_DIR, f"insertion_{target}.png")
    plot_bev(result["points"], result["existing_annotations"], result["annotations_sensor"],
             region=result["drivable"], title=f"insertion {target}", save_path=png)
    print("saved", png)
    if args.vis:
        show_open3d(result["points"], result["existing_annotations"], result["annotations_sensor"])
