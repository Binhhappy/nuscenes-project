"""Kiểm tra setup: load nuScenes mini, in thống kê, load 1 point cloud, list annotations, visualize Open3D.

Chạy: python n3-08c/verify_setup.py [--no-vis] [--sample-index 0]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from utils.nuscenes_loader import (get_annotations, get_lidar_token, get_map_name,  # noqa: E402
                                   get_pointcloud, load_nuscenes)


def check_paths() -> bool:
    ok = True
    for rel in (os.path.join(config.VERSION, "sample.json"), "samples", "sweeps",
                os.path.join("maps", "expansion")):
        exists = os.path.exists(os.path.join(config.DATAROOT, rel))
        ok &= exists or rel.startswith("maps")  # map expansion chỉ cần cho drivable area
        print(f"  [{'OK' if exists else 'MISSING'}] {os.path.join(config.DATAROOT, rel)}")
    return ok


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vis", action="store_true", help="bỏ qua cửa sổ Open3D")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    print(f"DATAROOT = {config.DATAROOT}  VERSION = {config.VERSION}")
    if not check_paths():
        print("FAIL: thiếu dữ liệu nuScenes (sửa config.DATAROOT hoặc đặt NUSCENES_DATAROOT)")
        return 1

    # 1. Load nuScenes + stats
    nusc = load_nuscenes(verbose=True)
    print(f"\nScenes: {len(nusc.scene)}  Samples: {len(nusc.sample)}  "
          f"Annotations: {len(nusc.sample_annotation)}")

    # 2. Load 1 point cloud
    sample = nusc.sample[args.sample_index]
    tok = sample["token"]
    points = get_pointcloud(nusc, tok)
    print(f"\nSample {tok} (map {get_map_name(nusc, tok)}, lidar {get_lidar_token(nusc, tok)})")
    print(f"Point cloud shape: {points.shape}  # (N, 5): x, y, z, intensity, ring")

    # 3. List annotations
    anns = get_annotations(nusc, tok)
    print(f"\n--- {len(anns)} annotations (sensor frame) ---")
    for a in anns:
        x, y, z = a["translation"]
        print(f"  {a['category']:35s} size={[round(v, 2) for v in a['size']]} pos=({x:.1f}, {y:.1f}, {z:.1f})")

    # 4. Visualize
    if not args.no_vis:
        from utils.visualization import show_open3d
        show_open3d(points, boxes=anns, window_name=f"nuScenes LiDAR - {tok}")

    print("\nverify_setup: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
