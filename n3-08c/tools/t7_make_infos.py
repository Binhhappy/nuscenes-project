"""T7 — Tạo info train OpenPCDet cho fine-tune: nuScenes mini_train gốc vs gốc + kịch bản VN.

Hai biến thể cùng số mẫu (cùng số bước train), chỉ khác nội dung, để tách tác dụng của dữ liệu VN khỏi
tác dụng "fine-tune thêm trên mini":
    nuscenes_infos_10sweeps_train_plain.pkl   info gốc x 2
    nuscenes_infos_10sweeps_train_vn.pkl      info gốc + info đã chèn (keyframe + sweep lấy từ aug_dir)

Info đã chèn: lidar_path / sweeps[*].lidar_path trỏ tới file .bin đã chèn (đường dẫn tuyệt đối, pathlib
nối root_path / tuyệt đối = tuyệt đối), gt thêm object chèn: box [x, y, z, dx, dy, dz, yaw, vx, vy] sensor frame
(như nuscenes_utils.fill_trainval_infos của OpenPCDet).

Chạy trong WSL env pcdet (sau khi sinh dữ liệu train: t6_eval.generate trên mini_train, --templates train):
    ~/miniconda3/envs/pcdet/bin/python tools/t7_make_infos.py --aug-dir ../output/t7/train/seed0/moto_swarm
"""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.detect import PCDET_ROOT, detection_class  # noqa: E402
from utils.transforms import quaternion_to_yaw  # noqa: E402

INFO_DIR = os.path.join(PCDET_ROOT, "data", "nuscenes", "v1.0-mini")
BASE_INFO = "nuscenes_infos_10sweeps_train.pkl"
BASE_DB = "nuscenes_dbinfos_10sweeps_withvelo.pkl"
T7_DB = "nuscenes_dbinfos_10sweeps_withvelo_t7.pkl"
MODEL_CFG = os.path.join(PCDET_ROOT, "tools", "cfgs", "nuscenes_models", "cbgs_dyn_pp_centerpoint.yaml")
DATASET_CFG = os.path.join(PCDET_ROOT, "tools", "cfgs", "dataset_configs", "nuscenes_dataset.yaml")
FT = {"BATCH_SIZE_PER_GPU": 2, "NUM_EPOCHS": 6, "LR": 0.0005}  # fine-tune ngắn, LR nhỏ hơn train gốc (0.001)


def filter_db(db: dict, base_infos: list, nusc, heldout: frozenset) -> dict:
    """Bỏ object thuộc instance held-out khỏi gt database (gt_sampling không được dán xe test vào cảnh train)."""
    def inst(e):
        ann = base_infos[e["image_idx"]]["gt_boxes_token"][e["gt_idx"]]
        return nusc.get("sample_annotation", ann)["instance_token"]
    return {k: [e for e in v if inst(e) not in heldout] for k, v in db.items()}


def write_cfgs(info_dir: str, ft: dict = FT, cbgs: bool = False, suffix: str = "") -> list:
    """Config fine-tune cho 2 biến thể, DB đã lọc, LR / epoch fine-tune.

    cbgs=False giữ đúng số mẫu 2 biến thể; nhưng trên mini (323 khung) tắt CBGS làm model quên lớp hiếm
    (lần 1: AP xe máy 0.55 -> 0.14 ở cả 2 biến thể), nên lần 2 bật lại như lúc train gốc.
    """
    import yaml
    with open(MODEL_CFG, encoding="utf-8") as f:
        model = yaml.safe_load(f)
    with open(DATASET_CFG, encoding="utf-8") as f:
        aug = yaml.safe_load(f)["DATA_AUGMENTOR"]
    for a in aug["AUG_CONFIG_LIST"]:
        if a["NAME"] == "gt_sampling":
            a["DB_INFO_PATH"] = [T7_DB]
    paths = []
    for name in ("plain", "vn"):
        cfg = dict(model)
        cfg["DATA_CONFIG"] = {**model["DATA_CONFIG"], "VERSION": "v1.0-mini", "BALANCED_RESAMPLING": cbgs,
                              "INFO_PATH": {"train": [f"nuscenes_infos_10sweeps_train_{name}.pkl"],
                                            "test": ["nuscenes_infos_10sweeps_val.pkl"]},
                              "DATA_AUGMENTOR": aug}
        cfg["OPTIMIZATION"] = {**model["OPTIMIZATION"], **ft}
        path = os.path.join(os.path.dirname(MODEL_CFG), f"t7_ft_{name}{suffix}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        paths.append(path)
    return paths


def augmented_info(info: dict, result: dict, aug_dir: str) -> dict:
    """Bản sao info trỏ tới point cloud đã chèn + gt có thêm object chèn."""
    out = copy.deepcopy(info)
    out["lidar_path"] = os.path.abspath(os.path.join(aug_dir, f"{info['token']}.bin"))
    for sw in out["sweeps"]:
        p = os.path.join(aug_dir, "sweeps", f"{sw['sample_data_token']}.bin")
        if os.path.isfile(p):
            sw["lidar_path"] = os.path.abspath(p)
    boxes, names, n_pts = [], [], []
    for a in result["annotations_sensor"]:
        cls = detection_class(a["category"])
        if cls is None or a["num_lidar_pts"] <= 0:
            continue
        w, l, h = a["size"]
        vx, vy = a.get("velocity_sensor", (0.0, 0.0))
        boxes.append([*a["translation"][:3], l, w, h, quaternion_to_yaw(a["rotation"]), vx, vy])
        names.append(cls)
        n_pts.append(a["num_lidar_pts"])
    if boxes:
        b = np.array(boxes, dtype=out["gt_boxes"].dtype)
        out["gt_boxes"] = np.concatenate([out["gt_boxes"], b], axis=0)
        out["gt_boxes_velocity"] = np.concatenate(
            [out["gt_boxes_velocity"], np.c_[b[:, 7:9], np.zeros(len(b))]], axis=0)
        out["gt_names"] = np.concatenate([out["gt_names"], np.array(names)])
        out["gt_boxes_token"] = np.concatenate([out["gt_boxes_token"],
                                                np.array([f"inserted_{i}" for i in range(len(b))])])
        out["num_lidar_pts"] = np.concatenate([out["num_lidar_pts"], np.array(n_pts)])
        out["num_radar_pts"] = np.concatenate([out["num_radar_pts"], np.zeros(len(b), dtype=int)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug-dir", nargs="+", required=True, help="thư mục output kịch bản (mỗi thư mục 1 bản chèn)")
    ap.add_argument("--info-dir", default=INFO_DIR)
    ap.add_argument("--lr", type=float, default=FT["LR"])
    ap.add_argument("--epochs", type=int, default=FT["NUM_EPOCHS"])
    ap.add_argument("--cbgs", action="store_true", help="bật class-balanced resampling (CBGS)")
    ap.add_argument("--suffix", default="", help="hậu tố tên cfg, vd _v2")
    args = ap.parse_args()

    with open(os.path.join(args.info_dir, BASE_INFO), "rb") as f:
        base = pickle.load(f)
    aug, n_obj = [], 0
    for d in args.aug_dir:
        for info in base:
            js = os.path.join(d, f"{info['token']}.json")
            if not os.path.isfile(js):
                continue
            with open(js, encoding="utf-8") as f:
                result = json.load(f)
            aug.append(augmented_info(info, result, d))
            n_obj += len(aug[-1]["gt_boxes"]) - len(info["gt_boxes"])
    # cùng số mẫu: plain lặp lại info gốc cho đủ len(aug)
    rng = np.random.default_rng(0)
    extra = [base[i] for i in rng.choice(len(base), len(aug), replace=len(aug) > len(base))]
    for name, infos in (("plain", base + extra), ("vn", base + aug)):
        path = os.path.join(args.info_dir, f"nuscenes_infos_10sweeps_train_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(infos, f)
        print(f"{name}: {len(infos)} info -> {path}")
    print(f"info đã chèn: {len(aug)}, object chèn: {n_obj}")

    from tools.t6_eval import HELDOUT_INSTANCES
    from utils.nuscenes_loader import load_nuscenes
    with open(os.path.join(args.info_dir, BASE_DB), "rb") as f:
        db = pickle.load(f)
    db_t7 = filter_db(db, base, load_nuscenes(), HELDOUT_INSTANCES)
    with open(os.path.join(args.info_dir, T7_DB), "wb") as f:
        pickle.dump(db_t7, f)
    print("gt database bỏ object held-out:", {k: len(db[k]) - len(db_t7[k]) for k in db if len(db[k]) != len(db_t7[k])})
    ft = {**FT, "LR": args.lr, "NUM_EPOCHS": args.epochs}
    for p in write_cfgs(args.info_dir, ft, args.cbgs, args.suffix):
        print("cfg ->", p)


if __name__ == "__main__":
    main()
