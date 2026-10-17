"""Xuất dữ liệu từng khung cho các chuỗi video (task_a/multiframe.py) để trang web phát lại khung-theo-khung:
mỗi khung = point cloud 3D + box màu theo kết quả detector + 6 ảnh camera đúng thời điểm + số đúng / nhầm / sót.

Dùng lại dữ liệu nhiều khung đã sinh cho video (output/multiframe/<scene>_<kind>, tools/demo_video.py); chạy
detector thật trên từng khung (WSL env pcdet, GPU).
    ~/miniconda3/envs/pcdet/bin/python tools/sequence_export.py
Output: output/sequences/index.json + output/sequences/<chuỗi>/<k>.json + output/sequences/<chuỗi>/<k>/<CAM>.jpg
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.detect  # noqa: E402,F401  (load torch trước shapely)
import config  # noqa: E402
from tools.case_gallery import camera_tiles, export_3d, frame_line  # noqa: E402
from tools.t6_eval import inserted_gts, object_records, real_gts  # noqa: E402

OUT = os.path.join(config.OUTPUT_DIR, "sequences")
SCENE = "scene-0103"
SEQS = [  # (tên, kind multiframe, số khung, thời tiết, mức, nhãn, video)
    ("swarm", "swarm", None, None, None, "Bầy xe máy quanh xe", "scene-0103_swarm_pretrained.mp4"),
    ("cut_in", "cut_in", None, None, None, "Xe máy cắt đầu", "scene-0103_cut_in_pretrained.mp4"),
    ("cargo", "cargo", None, None, None, "Xe máy chở hàng cồng kềnh", "scene-0103_cargo_pretrained.mp4"),
    ("dog", "dog", 12, None, None, "Chó lao ngang trước xe", "scene-0103_dog_pretrained.mp4"),
    ("swarm_storm", "swarm", None, "storm", "storm", "Bầy xe máy trong giông bão", "scene-0103_swarm_storm.mp4"),
]


def main():
    from tqdm import tqdm

    from task_a.multiframe import scene_samples
    from task_a.sweeps import load_multisweep
    from task_b.run_weather import apply_weather
    from tools.detect import Detector
    from utils.nuscenes_loader import load_nuscenes

    nusc = load_nuscenes()
    det = Detector()
    index = {}
    for name, kind, n_frames, weather, level, label, video in SEQS:
        aug_dir = os.path.join(config.OUTPUT_DIR, "multiframe", f"{SCENE}_{kind}")
        toks = [t for t in scene_samples(nusc, SCENE)[:n_frames] if os.path.isfile(os.path.join(aug_dir, f"{t}.json"))]
        if not toks:
            print("bỏ qua (chưa có dữ liệu nhiều khung, chạy tools/demo_video.py trước):", name)
            continue
        out_dir = os.path.join(OUT, name)
        os.makedirs(out_dir, exist_ok=True)
        frames, tot = [], np.zeros(3, int)
        for k, tok in enumerate(tqdm(toks, desc=name)):
            with open(os.path.join(aug_dir, f"{tok}.json"), encoding="utf-8") as f:
                result = json.load(f)
            real = real_gts(nusc, tok)
            gts = real + inserted_gts(result)
            rng = np.random.default_rng(config.SEED + k)
            vehicles = [g for g in gts if g["cls"] in ("car", "truck", "bus", "motorcycle", "trailer")]

            def per_sweep(p, i):
                return p if weather is None else apply_weather(p, weather, level, rng, vehicles if i == 0 else ())[0]
            pts = load_multisweep(nusc, tok, aug_dir, per_sweep=per_sweep)
            pred = det.predict(pts)
            recs = object_records(gts, pred, name, tok, None)
            ins = recs[len(real):]
            ok = sum(r["detected"] for r in ins)
            conf = sum(1 for r in ins if r["miss"] and r["miss"].startswith("confused"))
            tot += [ok, conf, len(ins) - ok - conf]
            export_3d(os.path.join(out_dir, f"{k}.json"), pts, gts, recs, pred, len(real), max_points=14000)
            cams = camera_tiles(os.path.join(out_dir, str(k)), nusc, tok, pts, gts, recs, len(real), weather, level)
            frames.append({"t": round(float(result.get("t", 0.0)), 2), "n": len(ins), "ok": ok, "confused": conf,
                           "missed": len(ins) - ok - conf, "line": frame_line(ins), "cams": cams})
        index[name] = {"label": label, "scene": SCENE, "weather": weather, "video": video, "frames": frames,
                       "totals": {"ok": int(tot[0]), "confused": int(tot[1]), "missed": int(tot[2]),
                                  "objects": int(tot.sum())}}
        print(name, index[name]["totals"])
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
