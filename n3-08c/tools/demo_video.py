"""Video demo: detector trên nuScenes gốc (trái) vs cùng cảnh có kịch bản VN nhiều khung (phải).

Màu box GT: xanh lá = detector phát hiện, đỏ = bỏ sót, cam = nhận nhầm lớp (ghi lớp bị nhận thành).
Box GT chèn vẽ đậm, GT thật vẽ mảnh. Prediction (score >= 0.3) vẽ nét đứt xanh dương.

Chạy trong WSL env pcdet (cần GPU):
    ~/miniconda3/envs/pcdet/bin/python tools/demo_video.py --scene scene-0103 --scenario swarm
    ~/miniconda3/envs/pcdet/bin/python tools/demo_video.py --scenario swarm --ckpt <ckpt T7> --label "sau fine-tune"
Output: <OUTPUT_DIR>/videos/<scene>_<scenario><tag>.mp4 (+ .png khung giữa để dán báo cáo).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.detect  # noqa: E402,F401  (load torch trước shapely)
import config  # noqa: E402
from tools.detect import SCORE_THRESH  # noqa: E402
from tools.t6_eval import inserted_gts, object_records, real_gts  # noqa: E402

STATUS_COLOR = {"ok": "#1a9e3a", "missed": "#d62728", "merged": "#d62728", "confused": "#ff8c00"}
LABEL_VI = {"car": "ô tô", "truck": "xe tải", "bicycle": "xe đạp", "pedestrian": "người", "barrier": "rào",
            "motorcycle": "xe máy", "traffic_cone": "cọc", "bus": "xe buýt", "construction_vehicle": "xe công trình",
            "trailer": "rơ-moóc"}
TITLE = {"swarm": "Kịch bản VN: bầy xe máy quanh xe", "cut_in": "Kịch bản VN: xe máy cắt đầu",
         "cargo": "Kịch bản VN: xe máy chở hàng cồng kềnh", "dog": "Kịch bản VN: chó lao ngang trước xe"}


def _rect(x, y, l, w, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


OBJ_POINTS = "#6250d6"  # điểm LiDAR của object chèn (khác màu trạng thái box)


def draw_panel(ax, points, gts, recs, pred, xlim, title, inserted_from: int, highlight: bool = False,
               center=(0.0, 0.0)):
    """highlight: tô màu điểm LiDAR nằm trong box object chèn (để thấy object được chèn trông thế nào).
    center: tâm khung nhìn (mặc định xe ego), nửa cạnh = xlim."""
    cx, cy = center
    keyframe = points[:, 4] == 0  # chỉ vẽ keyframe cho dễ nhìn (input model vẫn là 10 sweep)
    p = points[keyframe]
    m = (np.abs(p[:, 0] - cx) < xlim) & (np.abs(p[:, 1] - cy) < xlim)
    ax.scatter(p[m, 0], p[m, 1], s=0.4, c="#9aa0a6", linewidths=0)
    if highlight and len(gts) > inserted_from:
        from task_a.merge import points_in_new_box
        obj = np.zeros(len(p), bool)
        for g in gts[inserted_from:]:
            obj |= points_in_new_box(p, g)
        ax.scatter(p[obj, 0], p[obj, 1], s=2.5, c=OBJ_POINTS, linewidths=0, zorder=3)
    keep = pred["scores"] >= SCORE_THRESH
    for b in pred["boxes"][keep]:
        r = _rect(b[0], b[1], b[3], b[4], b[6])
        ax.plot(r[:, 0], r[:, 1], color="#1f6fd1", lw=0.9, ls="--")
    for i, (g, r) in enumerate(zip(gts, recs)):
        status = "ok" if r["detected"] else r["miss"].split(":")[0]
        color = STATUS_COLOR[status]
        inserted = i >= inserted_from  # object chèn: nét đậm, đủ màu; object thật nuScenes: nét mảnh, mờ
        w, l, _ = g["size"]
        rr = _rect(g["center"][0], g["center"][1], l, w, g["yaw"])
        ax.plot(rr[:, 0], rr[:, 1], color=color, lw=2.0 if inserted else 0.7, alpha=1.0 if inserted else 0.35)
        if status == "confused" and i >= inserted_from:
            ax.text(g["center"][0], g["center"][1] + 1.2, "→" + LABEL_VI.get(r["miss"].split(":")[1], ""),
                    color=color, fontsize=8, ha="center", weight="bold")
    ax.plot(0, 0, marker="^", color="black", ms=9)
    ax.set(xlim=(cx - xlim, cx + xlim), ylim=(cy - xlim, cy + xlim), aspect="equal", xticks=[], yticks=[])
    ax.set_title(title, fontsize=11)


def summary_line(recs, two_wheel_only=True):
    rs = [r for r in recs if not two_wheel_only or r["cls"] in ("motorcycle", "bicycle")]
    ok = sum(r["detected"] for r in rs)
    conf = sum(1 for r in rs if r["miss"] and r["miss"].startswith("confused"))
    return ok, len(rs), conf, len(rs) - ok - conf


def main():
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from tqdm import tqdm

    from task_a.extract_object import TemplateBank
    from task_a.multiframe import run_multiframe, scene_samples
    from task_a.sweeps import load_multisweep
    from tools.detect import Detector
    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="scene-0103", help="nên dùng scene mini_val: scene-0103 / scene-0916")
    ap.add_argument("--scenario", choices=["swarm", "cut_in", "cargo", "dog"], default="swarm")
    ap.add_argument("--n-objects", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--label", default="CenterPoint-PointPillar pretrained nuScenes")
    ap.add_argument("--tag", default="")
    ap.add_argument("--xlim", type=float, default=30.0)
    ap.add_argument("--regen", action="store_true", help="sinh lại dữ liệu nhiều khung")
    ap.add_argument("--weather", choices=["rain", "fog", "storm"], default=None,
                    help="so kịch bản VN trời trong (trái) với cùng kịch bản dưới thời tiết xấu (phải)")
    args = ap.parse_args()

    nusc = load_nuscenes()
    toks = scene_samples(nusc, args.scene)[:args.frames]
    aug_dir = os.path.join(config.OUTPUT_DIR, "multiframe", f"{args.scene}_{args.scenario}")
    if args.regen or not all(os.path.isfile(os.path.join(aug_dir, f"{t}.json")) for t in toks):
        n = args.n_objects or {"swarm": 15, "cargo": 2}.get(args.scenario, 1)
        run_multiframe(nusc, toks, args.scenario, TemplateBank().load_or_build(nusc), aug_dir, n,
                       np.random.default_rng(args.seed))
    det = Detector(ckpt=args.ckpt) if args.ckpt else Detector()

    out_dir = os.path.join(config.OUTPUT_DIR, "videos")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{args.scene}_{args.scenario}{args.tag}")
    frames, totals = [], np.zeros(4, int)
    legend = [Line2D([], [], color=STATUS_COLOR["ok"], lw=2, label="phát hiện đúng"),
              Line2D([], [], color=STATUS_COLOR["missed"], lw=2, label="bỏ sót"),
              Line2D([], [], color=STATUS_COLOR["confused"], lw=2, label="nhận nhầm lớp"),
              Line2D([], [], color="#1f6fd1", lw=1, ls="--", label="prediction của model"),
              Line2D([], [], color=OBJ_POINTS, marker="o", ls="", ms=4, label="điểm LiDAR object chèn")]
    weather_name = {"rain": "mưa to", "fog": "sương dày", "storm": "giông / bão"}.get(args.weather)
    level = {"rain": "heavy", "fog": "dense", "storm": "storm"}.get(args.weather)
    for k, tok in enumerate(tqdm(toks, desc="render")):
        with open(os.path.join(aug_dir, f"{tok}.json"), encoding="utf-8") as f:
            result = json.load(f)
        real = real_gts(nusc, tok)
        gts_a = real + inserted_gts(result)
        if args.weather:  # trái: kịch bản VN trời trong; phải: cùng kịch bản dưới thời tiết xấu
            from task_b.run_weather import apply_weather
            rng = np.random.default_rng(args.seed + k)
            vehicles = [g for g in gts_a if g["cls"] in ("car", "truck", "bus", "motorcycle", "trailer")]
            pts_o = load_multisweep(nusc, tok, aug_dir)
            gts_o = gts_a
            pts_a = load_multisweep(nusc, tok, aug_dir, per_sweep=lambda p, i: apply_weather(
                p, args.weather, level, rng, vehicles if i == 0 else ())[0])
        else:  # trái: nuScenes gốc; phải: có kịch bản VN
            pts_o, gts_o = load_multisweep(nusc, tok), real
            pts_a = load_multisweep(nusc, tok, aug_dir)
        pred_o = det.predict(pts_o)
        recs_o = object_records(gts_o, pred_o, "left", tok, None)
        pred_a = det.predict(pts_a)
        recs_a = object_records(gts_a, pred_a, args.scenario, tok, None)
        ins = recs_a[len(real):]
        s = summary_line(ins, two_wheel_only=False)
        totals += s

        fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), dpi=100)
        if args.weather:
            so = summary_line(recs_o[len(real):], two_wheel_only=False)
            left = f"{TITLE[args.scenario]}, trời trong: đúng {so[0]}/{so[1]} · nhầm {so[2]} · sót {so[3]}"
            right = f"Cùng cảnh, {weather_name}: đúng {s[0]}/{s[1]} · nhầm {s[2]} · sót {s[3]}"
        else:
            ok_o, n_o, _, _ = summary_line(recs_o, two_wheel_only=False)
            left = f"nuScenes gốc (Singapore/Boston): phát hiện {ok_o}/{n_o} object"
            right = f"{TITLE[args.scenario]}: object chèn đúng {s[0]}/{s[1]} · nhầm lớp {s[2]} · bỏ sót {s[3]}"
        draw_panel(axes[0], pts_o, gts_o, recs_o, pred_o, args.xlim, left, len(real), highlight=bool(args.weather))
        draw_panel(axes[1], pts_a, gts_a, recs_a, pred_a, args.xlim, right, len(real), highlight=True)
        fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=11, frameon=False)
        fig.suptitle(f"{args.label}   |   {args.scene}   t = {result['t']:.1f} s", fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if k == len(toks) // 2:
            fig.savefig(stem + ".png", dpi=100)
        plt.close(fig)

    imageio.mimsave(stem + ".mp4", frames, fps=2, macro_block_size=1)
    print(f"{len(frames)} khung -> {stem}.mp4")
    print(f"tổng object chèn: phát hiện {totals[0]}/{totals[1]} ({100 * totals[0] / max(totals[1], 1):.0f}%), "
          f"nhầm lớp {totals[2]}, bỏ sót {totals[3]}")


if __name__ == "__main__":
    main()
