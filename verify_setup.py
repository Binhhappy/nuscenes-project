import os
import sys

from nuscenes.nuscenes import NuScenes
import numpy as np
import open3d as o3d

# ===== 0. Check data =====
DATAROOT = r"C:\nuscenes-project\datasets\nuscenes"  # sửa lại đường dẫn của bạn
if not os.path.isfile(os.path.join(DATAROOT, "v1.0-mini", "sample.json")):
    sys.exit(f"Chưa có dữ liệu: giải nén nuScenes Mini vào {DATAROOT} (cần thư mục v1.0-mini/, samples/, sweeps/)")

# ===== 1. Load nuScenes =====
nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)

print(f"Scenes: {len(nusc.scene)}")
print(f"Samples: {len(nusc.sample)}")
print(f"Annotations: {len(nusc.sample_annotation)}")

# ===== 2. Load 1 point cloud =====
sample = nusc.sample[0]
lidar_token = sample["data"]["LIDAR_TOP"]
pc_path = nusc.get_sample_data_path(lidar_token)

points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
print(f"Point cloud shape: {points.shape}")  # (N, 5): x, y, z, intensity, ring

# ===== 3. List annotations in this sample =====
print("\n--- Annotations ---")
for ann_token in sample["anns"]:
    ann = nusc.get("sample_annotation", ann_token)
    print(f"  {ann['category_name']:30s} size={ann['size']} "
          f"pos=({ann['translation'][0]:.1f}, {ann['translation'][1]:.1f}, {ann['translation'][2]:.1f})")

# ===== 4. Visualize =====
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# Color by intensity
intensities = points[:, 3]
intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
colors = np.stack([intensities, intensities * 0.5, 1 - intensities], axis=1)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name="nuScenes LiDAR")
