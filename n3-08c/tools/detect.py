"""T6 — Detector pretrained nuScenes (OpenPCDet) chạy trên point cloud gốc / đã chèn.

Chỉ chạy trong WSL env `pcdet` (OpenPCDet build CUDA op, không chạy trên Windows):
    ~/miniconda3/envs/pcdet/bin/python tools/detect.py --sample 10
    ~/miniconda3/envs/pcdet/bin/python tools/detect.py --sample 10 --aug-dir ../output/scenarios/cut_in

Model mặc định: CenterPoint-PointPillar (cbgs_dyn_pp_centerpoint, mAP 50.0 / NDS 60.7 trên nuScenes val).
Input = load_multisweep (keyframe + 9 sweep, [x, y, z, intensity, time_lag] ở LIDAR_TOP sensor frame),
cùng format OpenPCDet NuScenesDataset dùng khi train, nên dùng thẳng checkpoint pretrained.

Box OpenPCDet: [x, y, z, dx (dài), dy (rộng), dz (cao), heading] ở sensor frame.
Khớp prediction - GT theo kiểu nuScenes: cùng lớp, khoảng cách tâm BEV < MATCH_DIST.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # torch phải load trước shapely (task_a.placement): ngược lại CUDA init lỗi
    import torch  # noqa: F401   "random_device could not be read: File exists" (xung đột thư viện C++)
except ImportError:  # Windows env không có torch: vẫn import được gt_boxes / match
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.transforms import quaternion_to_yaw  # noqa: E402

PCDET_ROOT = os.environ.get("PCDET_ROOT", os.path.expanduser("~/OpenPCDet"))
DEFAULT_CFG = "cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml"
DEFAULT_CKPT = os.path.join(PCDET_ROOT, "ckpt", "cbgs_dyn_pp_centerpoint.pth")
MATCH_DIST = 2.0       # (m) ngưỡng khớp tâm BEV, ngưỡng giữa của nuScenes detection (0.5/1/2/4)
SCORE_THRESH = 0.3     # bỏ prediction điểm thấp hơn khi đếm phát hiện

# category nuScenes -> lớp detection nuScenes (10 lớp)
CLASS_OF = {
    "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus", "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier", "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian", "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


# Vật thể VN ray-cast (task_a/raycast.py, category "vn.<tên>"): xe máy chở nhiều người / chở hàng vẫn là
# motorcycle; loại detector nuScenes không có lớp -> "obstacle:<tên>", khớp với prediction lớp bất kỳ.
VN_CLASS_OF = {"vn.car": "car", "vn.moto_1": "motorcycle", "vn.moto_multi": "motorcycle",
               "vn.moto_cargo": "motorcycle"}


def detection_class(category: str) -> Optional[str]:
    if category.startswith("vn."):
        return VN_CLASS_OF.get(category, "obstacle:" + category[3:])
    return CLASS_OF.get(category)


def class_ok(gt_cls: str, pred_label: str) -> bool:
    """Prediction có tính là phát hiện đúng GT không: cùng lớp, hoặc GT là vật cản không lớp (lớp nào cũng được)."""
    return gt_cls == pred_label or gt_cls.startswith("obstacle")


class Detector:
    """Model OpenPCDet + checkpoint, predict trên (N, 5) [x, y, z, intensity, time_lag]."""

    def __init__(self, cfg_file: str = DEFAULT_CFG, ckpt: str = DEFAULT_CKPT):
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        cwd = os.getcwd()
        os.chdir(os.path.join(PCDET_ROOT, "tools"))  # _BASE_CONFIG_ trong yaml là đường dẫn tương đối
        try:
            cfg_from_yaml_file(cfg_file, cfg)
        finally:
            os.chdir(cwd)
        self.cfg, self.torch, self._to_gpu = cfg, torch, load_data_to_gpu
        self.class_names = list(cfg.CLASS_NAMES)
        self.dataset = DatasetTemplate(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                       training=False, root_path=Path(PCDET_ROOT), logger=common_utils.create_logger())
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset)
        self.model.load_params_from_file(filename=ckpt, logger=common_utils.create_logger(), to_cpu=False)
        self.model.cuda().eval()

    def predict(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns boxes (K, 9) [x, y, z, dx, dy, dz, heading, vx, vy], scores (K,), labels (K,) tên lớp."""
        data = self.dataset.prepare_data({"points": points.astype(np.float32), "frame_id": 0})
        batch = self.dataset.collate_batch([data])
        self._to_gpu(batch)
        with self.torch.no_grad():
            pred = self.model(batch)[0][0]
        labels = pred["pred_labels"].cpu().numpy() - 1  # OpenPCDet đánh lớp từ 1
        return {"boxes": pred["pred_boxes"].cpu().numpy(), "scores": pred["pred_scores"].cpu().numpy(),
                "labels": np.array([self.class_names[i] for i in labels], dtype=object)}


def gt_boxes(annotations: Sequence[dict]) -> List[dict]:
    """Annotation sensor frame (size [w, l, h]) -> dict có class detection, center, yaw. Bỏ category ngoài 10 lớp."""
    out = []
    for a in annotations:
        cls = detection_class(a["category"])
        if cls:
            out.append({**a, "cls": cls, "center": np.asarray(a["translation"][:3], float),
                        "yaw": quaternion_to_yaw(a["rotation"])})
    return out


def match(gts: Sequence[dict], pred: Dict[str, np.ndarray], score_thresh: float = SCORE_THRESH,
          dist: float = MATCH_DIST) -> List[Optional[int]]:
    """Khớp tham lam theo score giảm dần: mỗi prediction khớp GT cùng lớp gần nhất chưa khớp (BEV < dist).

    Returns list theo gts: chỉ số prediction khớp, hoặc None nếu bị bỏ sót.
    """
    keep = np.where(pred["scores"] >= score_thresh)[0]
    keep = keep[np.argsort(-pred["scores"][keep])]
    result: List[Optional[int]] = [None] * len(gts)
    for k in keep:
        best, best_d = None, dist
        for i, g in enumerate(gts):
            if result[i] is None and class_ok(g["cls"], pred["labels"][k]):
                d = float(np.hypot(*(g["center"][:2] - pred["boxes"][k, :2])))
                if d < best_d:
                    best, best_d = i, d
        if best is not None:
            result[best] = int(k)
    return result


def main():
    from task_a.sweeps import load_multisweep
    from utils.nuscenes_loader import get_annotations, load_nuscenes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=10, help="chỉ số sample trong nusc.sample")
    ap.add_argument("--aug-dir", default=None, help="thư mục output đã chèn (scenarios/<sc>); bỏ trống = dữ liệu gốc")
    ap.add_argument("--max-range", type=float, default=50.0, help="chỉ tính GT trong bán kính này (m)")
    args = ap.parse_args()

    nusc = load_nuscenes()
    token = nusc.sample[args.sample]["token"]
    points = load_multisweep(nusc, token, args.aug_dir)
    det = Detector()
    pred = det.predict(points)

    anns = get_annotations(nusc, token)
    npts = {a["token"]: a["num_lidar_pts"] for a in (nusc.get("sample_annotation", t)
                                                     for t in nusc.get("sample", token)["anns"])}
    gts = [g for g in gt_boxes(anns)
           if npts.get(g["token"], 1) > 0 and np.hypot(*g["center"][:2]) < args.max_range]
    m = match(gts, pred)
    print(f"sample {args.sample} {token}: {len(points)} điểm, {int((pred['scores'] >= SCORE_THRESH).sum())} "
          f"prediction (score >= {SCORE_THRESH}), {len(gts)} GT trong {args.max_range:.0f} m")
    for cls in sorted({g["cls"] for g in gts}):
        idx = [i for i, g in enumerate(gts) if g["cls"] == cls]
        hit = sum(m[i] is not None for i in idx)
        print(f"  {cls:22s} recall {hit}/{len(idx)}")


if __name__ == "__main__":
    main()
