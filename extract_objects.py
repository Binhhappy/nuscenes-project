"""Bước 2: Object extraction — crop points trong cuboid và lưu template ở object-local frame.

Template: (M, 5) float32 [x, y, z, intensity, ring], gốc toạ độ = tâm cuboid,
trục x = hướng đầu xe (yaw = 0), z hướng lên. Đáy box ở z = -h/2.

Chạy:
  python extract_objects.py                      # trích toàn bộ v1.0-mini
  python extract_objects.py --show 5             # trích xong, xem 5 template nhiều điểm nhất
  python extract_objects.py --categories vehicle.car human.pedestrian.adult
"""
import argparse
import json
import os
import time

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, points_in_box
from tqdm import tqdm

DATAROOT = r"C:\nuscenes-project\datasets\nuscenes"
OUT_DIR = r"C:\nuscenes-project\output\templates"


def load_lidar(nusc, lidar_token):
    """Point cloud (N, 5) trong lidar sensor frame."""
    path = nusc.get_sample_data_path(lidar_token)
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def to_local(points, box):
    """Chuyển points (sensor frame) về object-local frame của box."""
    local = points.copy()
    xyz = points[:, :3] - box.center
    local[:, :3] = xyz @ box.orientation.rotation_matrix  # = R^T @ xyz cho từng điểm
    return local


def extract_sample(nusc, sample, categories, min_points):
    lidar_token = sample["data"]["LIDAR_TOP"]
    points = load_lidar(nusc, lidar_token)
    # Boxes đã được chuyển về lidar sensor frame
    _, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE)

    results = []
    for box in boxes:
        if categories and not any(box.name.startswith(c) for c in categories):
            continue
        mask = points_in_box(box, points[:, :3].T)
        if mask.sum() < min_points:
            continue
        template = to_local(points[mask], box).astype(np.float32)
        w, l, h = box.wlh
        results.append({
            "ann_token": box.token,
            "category": box.name,
            "num_points": int(mask.sum()),
            "size_wlh": [float(w), float(l), float(h)],
            "sensor_center": box.center.tolist(),
            "sensor_distance": float(np.linalg.norm(box.center[:2])),
            "sample_token": sample["token"],
            "scene_token": sample["scene_token"],
            "lidar_token": lidar_token,
            "template": template,
        })
    return results


def show_template(item):
    import open3d as o3d

    pts = item["template"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    pcd.paint_uniform_color([0.1, 0.4, 0.9])

    w, l, h = item["size_wlh"]
    obb = o3d.geometry.OrientedBoundingBox(np.zeros(3), np.eye(3), np.array([l, w, h]))
    obb.color = (1.0, 0.2, 0.2)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(l, w) * 0.5)

    title = f"{item['category']} | {item['num_points']} pts | {item['ann_token'][:8]}"
    o3d.visualization.draw_geometries([pcd, obb, axes], window_name=title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=DATAROOT)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--categories", nargs="*", default=["vehicle", "human.pedestrian", "movable_object"],
                    help="prefix category cần trích (rỗng = tất cả)")
    ap.add_argument("--min-points", type=int, default=20, help="bỏ vật thể có ít points hơn")
    ap.add_argument("--show", type=int, default=0, help="xem N template nhiều điểm nhất sau khi trích")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    index = []
    for sample in tqdm(nusc.sample, desc="Extracting"):
        for item in extract_sample(nusc, sample, args.categories, args.min_points):
            cat_dir = os.path.join(args.out, item["category"])
            os.makedirs(cat_dir, exist_ok=True)
            rel = os.path.join(item["category"], f"{item['ann_token']}.npy")
            np.save(os.path.join(args.out, rel), item.pop("template"))
            item["file"] = rel
            index.append(item)
    elapsed = time.time() - t0

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)

    # Thống kê
    counts = {}
    for it in index:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    print(f"\nĐã trích {len(index)} template từ {len(nusc.sample)} samples trong {elapsed:.1f}s")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:40s} {n}")
    print(f"Lưu tại: {args.out}")

    if args.show:
        for it in sorted(index, key=lambda x: -x["num_points"])[:args.show]:
            it["template"] = np.load(os.path.join(args.out, it["file"]))
            show_template(it)


if __name__ == "__main__":
    main()
