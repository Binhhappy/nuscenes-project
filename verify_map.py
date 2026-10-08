import os
import sys

from nuscenes.map_expansion.map_api import NuScenesMap

DATAROOT = r"C:\nuscenes-project\datasets\nuscenes"  # sửa lại đường dẫn của bạn
if not os.path.isdir(os.path.join(DATAROOT, "maps", "expansion")):
    sys.exit(f"Chưa có Map expansion: giải nén vào {DATAROOT}\\maps (cần maps\\expansion\\*.json)")

nusc_map = NuScenesMap(dataroot=DATAROOT, map_name="singapore-onenorth")

# Lấy tất cả drivable area polygons
drivable = nusc_map.get_records_in_patch(
    box_coords=(300, 1000, 400, 1100),  # (x_min, y_min, x_max, y_max)
    layer_names=["drivable_area"],
    mode="intersect",
)
print(f"Drivable area polygons found: {len(drivable['drivable_area'])}")
