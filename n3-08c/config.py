"""Cấu hình chung cho N3-08C — Synthetic Point Cloud Augmentation.

DATAROOT có thể override bằng biến môi trường NUSCENES_DATAROOT.
"""
import os

# Thư mục gốc repo (cha của n3-08c\), không phụ thuộc vị trí clone
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== Dataset =====
DATAROOT: str = os.environ.get("NUSCENES_DATAROOT", os.path.join(PROJECT_ROOT, "datasets", "nuscenes"))

VERSION: str = "v1.0-mini"
LIDAR_CHANNEL: str = "LIDAR_TOP"

# ===== Output paths =====
OUTPUT_DIR: str = os.environ.get("N308C_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
TEMPLATE_DIR: str = os.path.join(OUTPUT_DIR, "templates")    # object templates (object-local frame)
AUGMENTED_DIR: str = os.path.join(OUTPUT_DIR, "augmented")   # Task A: scene sau khi chèn object
WEATHER_DIR: str = os.path.join(OUTPUT_DIR, "weather")       # Task B: scene sau khi mô phỏng thời tiết
VIS_DIR: str = os.path.join(OUTPUT_DIR, "vis")               # ảnh BEV

SEED: int = 42

# ===== Task A: thresholds =====
MIN_DISTANCE: float = 3.0           # khoảng cách tối thiểu từ sensor tới object chèn (m)
MAX_DISTANCE: float = 70.0          # khoảng cách tối đa (m)
COLLISION_IOU_THRESH: float = 0.0   # BEV IoU > ngưỡng này = va chạm
MIN_TEMPLATE_POINTS: int = 20       # template ít điểm hơn bị bỏ
TEMPLATE_GROUND_MARGIN: float = 0.1  # bỏ điểm trong dải này phía trên đáy box (mặt đường lẫn vào); 0 = tắt
TEMPLATE_CATEGORIES: tuple = ("vehicle", "human.pedestrian", "movable_object")  # prefix category cho bank

# Ground model (tham số chọn theo sai số đáy box xe thật, xem task_a/ground_plane.py)
GROUND: dict = {
    "z_threshold": -1.5,   # RANSAC chỉ xét điểm thấp hơn (LIDAR_TOP cao ~1.84 m)
    "band": 0.6,           # ground candidate: |khoảng cách tới plane| < band (m)
    "cell": 1.0,           # lưới BEV (m) để lấy điểm thấp nhất mỗi cell
    "cell_tol": 0.2,       # giữ điểm trong cell có z <= z_min_cell + cell_tol
    "radius": 3.0,         # bán kính lấy median z cục bộ (m)
    "min_support": 5,      # ít điểm hơn -> fallback về plane, coi là vùng không thấy mặt đất
}

# Map layer hợp lệ để đặt object, theo prefix category (prefix dài nhất khớp được dùng)
PLACEMENT_LAYERS: dict = {
    "vehicle": ("drivable_area",),
    "movable_object": ("drivable_area",),
    "human.pedestrian": ("walkway", "ped_crossing"),
}
DEFAULT_LAYERS: tuple = ("drivable_area",)

# ===== Task B: weather levels =====
RAIN_LEVELS: dict = {
    "light": {"dropout": 0.1, "noise_ratio": 0.03},
    "moderate": {"dropout": 0.2, "noise_ratio": 0.07},
    "heavy": {"dropout": 0.3, "noise_ratio": 0.12},
}
FOG_VISIBILITY: dict = {"light": 200.0, "moderate": 50.0, "dense": 20.0}  # (m)
# Đường ướt: num_puddles vũng ellipse bán kính [min_radius, max_radius] (m) trên mặt đường;
# điểm mặt đường trong vũng mất với xác suất dropout (phản xạ gương), điểm còn lại intensity *= intensity_scale.
WET_ROAD_LEVELS: dict = {
    "light": {"num_puddles": 5, "min_radius": 0.5, "max_radius": 2.0, "dropout": 0.6, "intensity_scale": 0.5},
    "moderate": {"num_puddles": 15, "min_radius": 0.5, "max_radius": 4.0, "dropout": 0.85, "intensity_scale": 0.3},
    "heavy": {"num_puddles": 30, "min_radius": 0.5, "max_radius": 6.0, "dropout": 0.95, "intensity_scale": 0.2},
}
WET_ROAD_GROUND_TOL: float = 0.2  # điểm cao hơn mặt đất cục bộ quá ngưỡng này (m) không tính là mặt đường
# Tuyết: dropout = tỉ lệ mất điểm ở 30 m, noise_ratio = tỉ lệ beam trúng bông tuyết,
# attenuation = hệ số suy giảm intensity (1/m). Giá trị tự chọn, CHƯA hiệu chỉnh với dữ liệu thật.
SNOW_LEVELS: dict = {
    "light": {"dropout": 0.1, "noise_ratio": 0.05, "attenuation": 0.005},
    "moderate": {"dropout": 0.2, "noise_ratio": 0.12, "attenuation": 0.01},
    "heavy": {"dropout": 0.35, "noise_ratio": 0.25, "attenuation": 0.02},
}

# Nhãn điểm trong <name>_labels.npy (dùng chung cho mọi weather)
LABEL_ORIGINAL: int = 0   # điểm gốc
LABEL_NOISE: int = 1      # điểm do thời tiết sinh ra (giọt mưa, backscatter sương, bông tuyết)
LABEL_PUDDLE: int = 2     # điểm gốc nằm trong vũng nước (còn sống sót, intensity đã giảm)
LABEL_NAMES: dict = {LABEL_ORIGINAL: "original", LABEL_NOISE: "noise", LABEL_PUDDLE: "puddle"}
