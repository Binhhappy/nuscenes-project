"""Task B — Detector pretrained giảm bao nhiêu khi trời xấu (mưa / sương / giông bão), trên mini_val.

Thời tiết áp riêng lên TỪNG sweep trước khi gộp 10 sweep (load_multisweep(per_sweep=...)), như cảm biến quét
trong mưa thật. Spray của giông chỉ sinh ở keyframe (box xe đúng frame keyframe).
--aug-dir: chạy trên dữ liệu đã chèn kịch bản (vd bầy xe máy) để đo tác động kép kịch bản VN + thời tiết;
GT = object thật + object chèn (đo trên object chèn).

Chạy trong WSL env pcdet (GPU):
    ~/miniconda3/envs/pcdet/bin/python tools/weather_eval.py
    ~/miniconda3/envs/pcdet/bin/python tools/weather_eval.py --aug-dir ../output/t6/seed0/moto_swarm --tag _swarm
Output: <OUTPUT_DIR>/weather_eval/report<tag>.json + bảng.
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
from tools.t6_eval import inserted_gts, object_records, print_report, real_gts, summarize  # noqa: E402

CONDITIONS = [("clear", None, None), ("rain_light", "rain", "light"), ("rain_moderate", "rain", "moderate"),
              ("rain_heavy", "rain", "heavy"), ("fog_light", "fog", "light"), ("fog_moderate", "fog", "moderate"),
              ("fog_dense", "fog", "dense"), ("storm", "storm", "storm")]


def main():
    from tqdm import tqdm

    from task_a.scenarios import split_samples
    from task_a.sweeps import load_multisweep
    from task_b.run_weather import apply_weather
    from tools.detect import Detector
    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug-dir", default=None)
    ap.add_argument("--every", type=int, default=1, help="lấy 1 sample mỗi N (giảm tải máy)")
    ap.add_argument("--conditions", nargs="+", default=[c[0] for c in CONDITIONS])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    nusc = load_nuscenes()
    samples = split_samples(nusc, "mini_val")[::args.every]
    if args.aug_dir:
        samples = [t for t in samples if os.path.isfile(os.path.join(args.aug_dir, f"{t}.json"))]
    det = Detector()
    rng = np.random.default_rng(args.seed)
    records = []
    for cond, weather, level in CONDITIONS:
        if cond not in args.conditions:
            continue
        for tok in tqdm(samples, desc=cond):
            real = real_gts(nusc, tok)
            gts, n_real = real, len(real)
            if args.aug_dir:
                with open(os.path.join(args.aug_dir, f"{tok}.json"), encoding="utf-8") as f:
                    gts = real + inserted_gts(json.load(f))
            vehicles = [g for g in gts if g["cls"] in ("car", "truck", "bus", "motorcycle", "trailer")]

            def per_sweep(pts, i):
                if weather is None:
                    return pts
                return apply_weather(pts, weather, level, rng, vehicles if i == 0 else ())[0]
            pred = det.predict(load_multisweep(nusc, tok, args.aug_dir, per_sweep=per_sweep))
            recs = object_records(gts, pred, cond, tok, None)
            records += recs[n_real:] if args.aug_dir else recs
    rows = summarize(records)
    out_dir = os.path.join(config.OUTPUT_DIR, "weather_eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"report{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"aug_dir": args.aug_dir, "n_samples": len(samples), "rows": rows}, f, indent=1, ensure_ascii=False)
    order = {c[0]: i for i, c in enumerate(CONDITIONS)}
    print_report(sorted(rows, key=lambda r: (r["cls"], order.get(r["group"], 99))))


if __name__ == "__main__":
    main()
