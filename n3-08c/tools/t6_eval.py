"""T6 — Detector pretrained nuScenes có xử lý được tình huống giao thông VN không?

3 nhóm, cùng detector (tools/detect.py, CenterPoint-PointPillar pretrained), cùng split mini_val (scene model
chưa thấy khi train):
    A  real        object thật của nuScenes                          -> mốc chuẩn
    B  control_*   object chèn, tư thế giao thông bình thường        -> chèn có "giả" không (B ~ A là tốt)
    C  kịch bản VN cut_in / wrong_way / moto_swarm / jaywalk / lane_block
Kết luận hợp lệ khi B ~ A và C < B (cùng lớp object): detector yếu do tình huống, không do dữ liệu chèn.

Mỗi object GT ghi: detected (khớp cùng lớp, tâm BEV < 2 m, score >= 0.3), và nếu bỏ sót thì lý do:
    confused   có prediction lớp khác ở gần (vd xe máy -> bicycle / pedestrian)
    merged     có prediction cùng lớp ở gần nhưng đã khớp object khác (bầy xe bị gộp)
    missed     không có gì
Với xe: heading_flip (hướng lệch > 90°), vel_flip (vận tốc dự đoán ngược chiều thật, chỉ object đang chạy).

Chạy trong WSL env pcdet (cần GPU):
    ~/miniconda3/envs/pcdet/bin/python tools/t6_eval.py --seeds 0 1 2      # sinh dữ liệu + đánh giá
    ~/miniconda3/envs/pcdet/bin/python tools/t6_eval.py --seeds 0 1 2 --skip-gen
Output: <OUTPUT_DIR>/t6/seed<k>/<scenario>/ (dữ liệu chèn + sweep), t6/records.json (từng object),
        t6/report.json + bảng in ra màn hình.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.detect  # noqa: E402,F401  (load torch trước shapely, xem tools/detect.py)
import config  # noqa: E402
from tools.detect import MATCH_DIST, SCORE_THRESH, class_ok, gt_boxes, match  # noqa: E402
from utils.transforms import wrap_angle  # noqa: E402

VN_SCENARIOS = ("cut_in", "wrong_way", "moto_swarm", "jaywalk", "lane_block")
CONTROLS = ("control_2w", "control_car", "control_ped")
MAX_RANGE = 50.0                       # (m) như nuScenes eval với phần lớn lớp
DIST_BINS = (0, 10, 20, 35, MAX_RANGE)
VEHICLES = ("car", "truck", "bus", "trailer", "construction_vehicle", "motorcycle", "bicycle")
OUT_ROOT = os.path.join(config.OUTPUT_DIR, "t6")
# T7: instance xe 2 bánh giữ riêng cho test (mini chỉ có 6 xe máy + 2 xe đạp có người lái). Dữ liệu fine-tune chỉ
# dùng instance còn lại, để model sau fine-tune được đo trên xe nó chưa thấy hình dạng.
HELDOUT_INSTANCES = frozenset({"21cdc9f24c614a6197fd044379697197", "c9944f24807c478d85e2daa4bf02ea3c",
                               "ec0e6425eae2442698ff7b43d405d279"})


def template_filter(nusc, which: str):
    """which: all | train (bỏ instance held-out) | test (chỉ instance held-out, với xe 2 bánh)."""
    if which == "all":
        return None
    two_wheel = ("vehicle.motorcycle", "vehicle.bicycle")

    def allow(meta: dict) -> bool:
        if not meta["category"].startswith(two_wheel):
            return True
        held = nusc.get("sample_annotation", meta["ann_token"])["instance_token"] in HELDOUT_INSTANCES
        return held if which == "test" else not held
    return allow


def real_gts(nusc, sample_token: str, max_range: float = MAX_RANGE) -> List[dict]:
    """GT object thật trong max_range, có >= 1 điểm LiDAR (như nuScenes eval)."""
    from utils.nuscenes_loader import get_annotations
    npts = {t: nusc.get("sample_annotation", t)["num_lidar_pts"] for t in nusc.get("sample", sample_token)["anns"]}
    return [{**g, "num_lidar_pts": npts.get(g["token"], 0)} for g in gt_boxes(get_annotations(nusc, sample_token))
            if npts.get(g["token"], 0) > 0 and np.hypot(*g["center"][:2]) < max_range]


def inserted_gts(result: dict, max_range: float = MAX_RANGE) -> List[dict]:
    gts = gt_boxes(result["annotations_sensor"])
    return [g for g in gts if g["num_lidar_pts"] > 0 and np.hypot(*g["center"][:2]) < max_range]


def object_records(gts: Sequence[dict], pred: Dict[str, np.ndarray], group: str, sample_token: str,
                   seed: Optional[int]) -> List[dict]:
    """Khớp gts với pred, trả về 1 record / GT (xem docstring module)."""
    m = match(gts, pred)
    keep = pred["scores"] >= SCORE_THRESH
    taken = {k for k in m if k is not None}
    out = []
    for g, k in zip(gts, m):
        r = {"group": group, "seed": seed, "sample": sample_token, "cls": g["cls"], "category": g["category"],
             "dist": float(np.hypot(*g["center"][:2])), "npts": int(g.get("num_lidar_pts", -1)),
             "detected": k is not None, "miss": None, "score": None, "heading_flip": None, "vel_flip": None}
        if k is not None:
            r["score"] = float(pred["scores"][k])
            if g["cls"] in VEHICLES:
                r["heading_flip"] = bool(abs(wrap_angle(pred["boxes"][k, 6] - g["yaw"])) > np.pi / 2)
                v_gt = np.asarray(g.get("velocity_sensor", (0.0, 0.0)), float)
                v_pr = pred["boxes"][k, 7:9]
                if np.linalg.norm(v_gt) > 1.0 and np.linalg.norm(v_pr) > 0.5:
                    r["vel_flip"] = bool(v_gt @ v_pr < 0)
        else:
            near = keep & (np.hypot(*(pred["boxes"][:, :2] - g["center"][:2]).T) < MATCH_DIST)
            other = [str(pred["labels"][j]) for j in np.flatnonzero(near) if not class_ok(g["cls"], pred["labels"][j])]
            same = [j for j in np.flatnonzero(near) if class_ok(g["cls"], pred["labels"][j]) and j in taken]
            r["miss"] = f"confused:{other[0]}" if other else ("merged" if same else "missed")
        out.append(r)
    return out


def generate(nusc, samples: Sequence[str], seeds: Sequence[int], names: Sequence[str], out_root: str,
             templates: str = "all", n_attempts: int = 300) -> None:
    from tqdm import tqdm

    from task_a.extract_object import TemplateBank
    from task_a.scenarios import SCENARIOS, run_scenario, save_scenario
    from task_a.vn_scenarios import VN_SCENARIOS, run_vn_scenario
    bank = TemplateBank().load_or_build(nusc)
    allow = template_filter(nusc, templates)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for name in names:
            out_dir = os.path.join(out_root, f"seed{seed}", name)
            n_obj = 0
            for tok in tqdm(samples, desc=f"gen seed{seed} {name}"):
                if name in VN_SCENARIOS:  # vật thể VN ray-cast (task_a/vn_scenarios.py)
                    r = run_vn_scenario(nusc, tok, VN_SCENARIOS[name], bank, rng, n_attempts, allow)
                else:
                    r = run_scenario(nusc, tok, SCENARIOS[name], bank, rng, n_attempts, allow_template=allow)
                if r["n_inserted"]:
                    save_scenario(nusc, r, out_dir, sweeps=True, rng=rng)
                    n_obj += r["n_inserted"]
            print(f"seed{seed} {name}: {n_obj} object -> {out_dir}")


def evaluate(nusc, samples: Sequence[str], seeds: Sequence[int], names: Sequence[str], out_root: str,
             ckpt: Optional[str] = None) -> List[dict]:
    from tqdm import tqdm

    from task_a.sweeps import load_multisweep
    from tools.detect import Detector
    det = Detector(ckpt=ckpt) if ckpt else Detector()
    records = []
    for tok in tqdm(samples, desc="eval real"):
        records += object_records(real_gts(nusc, tok), det.predict(load_multisweep(nusc, tok)), "real", tok, None)
    for seed in seeds:
        for name in names:
            aug_dir = os.path.join(out_root, f"seed{seed}", name)
            if not os.path.isdir(aug_dir):
                continue
            for tok in tqdm([t for t in samples if os.path.isfile(os.path.join(aug_dir, f"{t}.json"))],
                            desc=f"eval seed{seed} {name}"):
                with open(os.path.join(aug_dir, f"{tok}.json"), encoding="utf-8") as f:
                    result = json.load(f)
                gts = real_gts(nusc, tok) + inserted_gts(result)
                n_real = len(gts) - len(inserted_gts(result))
                pred = det.predict(load_multisweep(nusc, tok, aug_dir))
                # chỉ ghi object chèn (object thật đã đo ở nhóm real), nhưng khớp chung để cạnh tranh công bằng
                records += object_records(gts, pred, name, tok, seed)[n_real:]
    return records


def _rate(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(records: Sequence[dict]) -> List[dict]:
    """1 dòng / (group, cls): recall (+ std giữa seed), cự ly, kiểu bỏ sót, lật hướng / vận tốc."""
    by = defaultdict(list)
    for r in records:
        by[(r["group"], r["cls"])].append(r)
    rows = []
    for (group, cls), rs in by.items():
        seeds = sorted({r["seed"] for r in rs if r["seed"] is not None})
        per_seed = [np.mean([r["detected"] for r in rs if r["seed"] == s]) for s in seeds]
        misses = defaultdict(int)
        for r in rs:
            if r["miss"]:
                misses[r["miss"].split(":")[0]] += 1
        bins = {}
        for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
            d = [r["detected"] for r in rs if lo <= r["dist"] < hi]
            bins[f"{lo}-{hi}m"] = {"n": len(d), "recall": float(np.mean(d)) if d else None}
        rows.append({
            "group": group, "cls": cls, "n": len(rs), "recall": float(np.mean([r["detected"] for r in rs])),
            "recall_std_seeds": float(np.std(per_seed)) if len(per_seed) > 1 else None,
            "median_dist": float(np.median([r["dist"] for r in rs])),
            "median_npts": float(np.median([r["npts"] for r in rs])),
            "missed": misses["missed"] / len(rs), "confused": misses["confused"] / len(rs),
            "merged": misses["merged"] / len(rs),
            "confused_as": dict(sorted(_count(r["miss"].split(":")[1] for r in rs
                                              if r["miss"] and r["miss"].startswith("confused")).items())),
            "heading_flip": _rate(r["heading_flip"] for r in rs), "vel_flip": _rate(r["vel_flip"] for r in rs),
            "recall_by_dist": bins,
        })
    order = {g: i for i, g in enumerate(("real", *CONTROLS, "rc_control_car", "rc_control_moto", *VN_SCENARIOS))}
    return sorted(rows, key=lambda r: (r["cls"], order.get(r["group"], 99)))


def _count(xs) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for x in xs:
        out[x] += 1
    return out


def print_report(rows: Sequence[dict], min_n: int = 5) -> None:
    pct = lambda x: "   -" if x is None else f"{100 * x:4.0f}"  # noqa: E731
    print(f"\n{'lớp':14s} {'nhóm':12s} {'n':>5s} {'recall':>7s} {'±seed':>6s} {'r_med':>6s} {'pts':>5s} "
          f"{'sót':>5s} {'nhầm':>5s} {'gộp':>5s} {'lật hướng':>9s} {'ngược v':>8s}")
    for r in rows:
        if r["n"] < min_n:
            continue
        std = "     -" if r["recall_std_seeds"] is None else f"{100 * r['recall_std_seeds']:6.1f}"
        print(f"{r['cls']:14s} {r['group']:12s} {r['n']:5d} {pct(r['recall']):>6s}% {std} {r['median_dist']:6.1f} "
              f"{r['median_npts']:5.0f} {pct(r['missed']):>4s}% {pct(r['confused']):>4s}% {pct(r['merged']):>4s}% "
              f"{pct(r['heading_flip']):>8s}% {pct(r['vel_flip']):>7s}%")
    print("(sót/nhầm/gộp: % trên tổng object; lật hướng / ngược v: % trên object đã phát hiện)")


def main():
    from task_a.scenarios import split_samples
    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scenarios", nargs="+", default=[*CONTROLS, *VN_SCENARIOS])
    ap.add_argument("--split", default="mini_val")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--skip-gen", action="store_true", help="dùng dữ liệu đã sinh trong out-root")
    ap.add_argument("--gen-only", action="store_true", help="chỉ sinh dữ liệu (T7: dữ liệu fine-tune trên mini_train)")
    ap.add_argument("--attempts", type=int, default=300,
                    help="số tư thế thử mỗi sample (ít template -> phần lớn bị loại no_template, cần tăng)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N sample đầu (chạy thử)")
    ap.add_argument("--templates", choices=["all", "train", "test"], default="all",
                    help="xe 2 bánh: all | train (bỏ HELDOUT_INSTANCES) | test (chỉ HELDOUT_INSTANCES)")
    ap.add_argument("--ckpt", default=None, help="checkpoint detector (mặc định pretrained, T7: model fine-tune)")
    ap.add_argument("--tag", default="", help="hậu tố tên file kết quả, vd _ft_vn")
    args = ap.parse_args()

    nusc = load_nuscenes()
    samples = split_samples(nusc, args.split)[:args.limit]
    if not args.skip_gen:
        generate(nusc, samples, args.seeds, args.scenarios, args.out_root, args.templates, args.attempts)
    if args.gen_only:
        return
    records = evaluate(nusc, samples, args.seeds, args.scenarios, args.out_root, args.ckpt)
    rows = summarize(records)
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, f"records{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)
    with open(os.path.join(args.out_root, f"report{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "seeds": args.seeds, "n_samples": len(samples), "score_thresh": SCORE_THRESH,
                   "match_dist": MATCH_DIST, "templates": args.templates, "ckpt": args.ckpt, "rows": rows},
                  f, indent=1, ensure_ascii=False)
    print_report(rows)


if __name__ == "__main__":
    main()
