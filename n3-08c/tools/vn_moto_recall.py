"""Bước 0 — recall xe máy của detector camera pretrained (YOLO, COCO) theo độ đông, trên ảnh Việt Nam thật.

Chạy trong env có ultralytics (không cài vào nuscenes-env):
    python tools/vn_moto_recall.py --images ../output/step0/cvat/obj_train_data --labels <thư mục .txt export YOLO 1.1>

Nhãn = YOLO 1.1 export từ CVAT (file vn_moto_frames.py tạo), obj.names nằm ở --labels hoặc thư mục cha.
Detector: ultralytics YOLO pretrained COCO, chỉ lớp motorcycle (id 3), không tinh chỉnh.

Video VnTraffic chỉ dài 17 s nên mật độ cả khung gần như không đổi. Vì vậy đo độ đông theo TỪNG XE:
- crowd: số xe máy khác có tâm cách tâm xe này < CROWD_RADIUS x chiều cao box xe này;
- occl:  phần diện tích box bị box xe khác gần camera hơn (đáy box thấp hơn trong ảnh) che.
Xe đông thường ở xa (box nhỏ), nên recall theo crowd được chuẩn hoá theo cỡ box (size-adjusted):
recall từng ô (crowd, cỡ) lấy trọng số theo phân bố cỡ của toàn bộ xe. Khác biệt còn lại = do đông, không do nhỏ.

Khớp: IoU >= --iou (mặc định 0.3, vì box nhãn bao cả người ngồi còn box COCO motorcycle thường chỉ bao xe;
0.5 được báo cáo kèm). CI 95% = bootstrap theo khung. Các khung cùng 17 s video nên tương quan:
CI thực tế rộng hơn số in ra.

Đầu ra (mặc định ../output/step0/eval/): per_object.csv, summary.json, recall_vs_crowd.png,
recall_vs_occlusion.png, misses/<khung>.jpg (xanh = phát hiện, đỏ = bỏ sót).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COCO_MOTORCYCLE = 3
MIN_H = 24             # (px) xe máy thấp hơn bị loại khỏi recall (quá nhỏ để chắc nhãn đủ)
CROWD_RADIUS = 1.5     # bán kính đếm xe lân cận, theo chiều cao box của chính xe đó
SIZE_EDGES = [24, 48, 96, np.inf]          # chiều cao box (px)
CROWD_EDGES = [0, 1, 3, 6, np.inf]         # số xe lân cận: 0 | 1-2 | 3-5 | >= 6
OCCL_EDGES = [0, 0.1, 0.3, 0.5, 1.01]      # phần bị che
MIN_CELL = 10          # ô (crowd, cỡ) ít hơn số xe này không dùng khi chuẩn hoá


def bin_labels(edges, unit=""):
    return [f">={lo:g}{unit}" if np.isinf(hi) else f"{lo:g}-{hi:g}{unit}" for lo, hi in zip(edges[:-1], edges[1:])]


def read_names(labels_dir):
    for d in (labels_dir, labels_dir.parent):
        p = d / "obj.names"
        if p.exists():
            return [n.strip() for n in p.read_text(encoding="utf-8").splitlines() if n.strip()]
    raise SystemExit(f"không thấy obj.names trong {labels_dir} hoặc thư mục cha")


def read_yolo(path, w, h, keep_id):
    """YOLO txt -> box xyxy (px) của lớp keep_id, kèm số box lớp khác."""
    boxes, others = [], 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            if int(f[0]) != keep_id:
                others += 1
                continue
            cx, cy, bw, bh = (float(v) for v in f[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=float).reshape(-1, 4), others


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def greedy_match(iou, thresh):
    """Ghép tham lam theo IoU giảm dần -> mảng: chỉ số prediction của từng GT (-1 = bỏ sót)."""
    match = -np.ones(iou.shape[0], dtype=int)
    if iou.size == 0:
        return match
    used = set()
    for flat in np.argsort(-iou, axis=None):
        g, p = np.unravel_index(flat, iou.shape)
        if iou[g, p] < thresh:
            break
        if match[g] < 0 and p not in used:
            match[g], _ = p, used.add(p)
    return match


def crowd_and_occlusion(gt):
    """Mỗi xe: số xe lân cận trong CROWD_RADIUS x chiều cao, và phần bị xe gần camera hơn che."""
    n = len(gt)
    ctr = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], 1) if n else np.zeros((0, 2))
    hgt = gt[:, 3] - gt[:, 1] if n else np.zeros(0)
    crowd, occl = np.zeros(n, dtype=int), np.zeros(n)
    g = (np.arange(8) + 0.5) / 8  # lưới 8x8 điểm mẫu trong box
    for i in range(n):
        d = np.hypot(*(ctr - ctr[i]).T)
        crowd[i] = int(((d < CROWD_RADIUS * hgt[i]) & (np.arange(n) != i)).sum())
        front = gt[(gt[:, 3] > gt[i, 3]) & (np.arange(n) != i)]
        if len(front):
            px = gt[i, 0] + g[None, :] * (gt[i, 2] - gt[i, 0])
            py = gt[i, 1] + g[:, None] * (gt[i, 3] - gt[i, 1])
            px, py = np.broadcast_to(px, (8, 8)).ravel(), np.broadcast_to(py, (8, 8)).ravel()
            inside = ((px[:, None] >= front[:, 0]) & (px[:, None] <= front[:, 2])
                      & (py[:, None] >= front[:, 1]) & (py[:, None] <= front[:, 3])).any(1)
            occl[i] = inside.mean()
    return crowd, occl


rows_by_frame_cache = {}  # khung -> các xe được đánh giá; dùng khi bootstrap theo khung


def recall_by_bin(rows, key, edges, frames=None):
    """Recall theo bin của key; frames = danh sách khung (có lặp) khi bootstrap."""
    pool = rows if frames is None else [r for f in frames for r in rows_by_frame_cache[f]]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r["hit"] for r in pool if lo <= r[key] < hi]
        out.append((sum(sel) / len(sel) if sel else np.nan, len(sel)))
    return out


def size_adjusted(rows, frames=None):
    """Recall theo crowd, chuẩn hoá theo phân bố cỡ box của toàn bộ xe."""
    pool = rows if frames is None else [r for f in frames for r in rows_by_frame_cache[f]]
    sizes = np.array([r["h"] for r in pool])
    w = np.histogram(sizes, SIZE_EDGES)[0].astype(float)
    w /= max(w.sum(), 1)
    out = []
    for clo, chi in zip(CROWD_EDGES[:-1], CROWD_EDGES[1:]):
        num = den = 0.0
        for k, (slo, shi) in enumerate(zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])):
            cell = [r["hit"] for r in pool if clo <= r["crowd"] < chi and slo <= r["h"] < shi]
            if len(cell) >= MIN_CELL:
                num += w[k] * np.mean(cell)
                den += w[k]
        out.append(num / den if den > 0 else np.nan)
    return out


def bootstrap(fn, frames, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    samples = np.array([fn(list(rng.choice(frames, len(frames)))) for _ in range(n_boot)], dtype=float)
    return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="thư mục ảnh khung (.jpg)")
    ap.add_argument("--labels", required=True, help="thư mục .txt YOLO 1.1 đã gán xong")
    ap.add_argument("--model", default="yolo11m.pt", help="checkpoint ultralytics pretrained COCO")
    ap.add_argument("--imgsz", type=int, default=1280, help="cạnh ảnh khi suy luận (xe nhỏ cần >= 1280)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3, help="ngưỡng IoU khớp (0.5 báo cáo kèm)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "output" / "step0" / "eval"))
    ap.add_argument("--n-vis", type=int, default=6, help="số khung vẽ minh hoạ bỏ sót")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    images, labels, out = Path(args.images), Path(args.labels), Path(args.out)
    (out / "misses").mkdir(parents=True, exist_ok=True)
    names = read_names(labels)
    if "motorbike" not in names:
        raise SystemExit(f"obj.names không có lớp motorbike: {names}")
    moto_id = names.index("motorbike")
    model = YOLO(args.model)

    rows, frames_info, leftover = [], [], 0
    img_paths = sorted(images.glob("*.jpg"))
    if not img_paths:
        raise SystemExit(f"không có ảnh .jpg trong {images}")
    for ip in img_paths:
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gt, others = read_yolo(labels / f"{ip.stem}.txt", w, h, moto_id)
        leftover += others
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=[COCO_MOTORCYCLE], verbose=False)[0]
        pred = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        score = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        iou = iou_matrix(gt, pred)
        match = greedy_match(iou, args.iou)
        match50 = greedy_match(iou, 0.5)
        crowd, occl = crowd_and_occlusion(gt)
        frame_rows = []
        for i, b in enumerate(gt):
            bh = b[3] - b[1]
            r = dict(frame=ip.stem, x0=round(b[0], 1), y0=round(b[1], 1), x1=round(b[2], 1), y1=round(b[3], 1),
                     h=round(bh, 1), crowd=int(crowd[i]), occl=round(float(occl[i]), 3),
                     hit=int(match[i] >= 0), hit50=int(match50[i] >= 0),
                     score=round(float(score[match[i]]), 3) if match[i] >= 0 else "",
                     eval=int(bh >= MIN_H))
            frame_rows.append(r)
        rows += frame_rows
        ev = [r for r in frame_rows if r["eval"]]
        frames_info.append(dict(frame=ip.stem, n_gt=len(ev), n_pred=len(pred),
                                recall=sum(r["hit"] for r in ev) / len(ev) if ev else None))
        if len(frames_info) <= args.n_vis:
            vis = img.copy()
            for r in frame_rows:
                c = (0, 200, 0) if r["hit"] else (0, 0, 255)
                cv2.rectangle(vis, (int(r["x0"]), int(r["y0"])), (int(r["x1"]), int(r["y1"])), c, 2)
            for p in pred:
                cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 160, 0), 1)
            cv2.imwrite(str(out / "misses" / f"{ip.stem}.jpg"), vis)

    if leftover:
        print(f"CẢNH BÁO: còn {leftover} box không phải motorbike (vd `review`) trong nhãn: chưa gán xong?")
    with open(out / "per_object.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    ev = [r for r in rows if r["eval"]]
    frames = sorted({r["frame"] for r in ev})
    rows_by_frame_cache.update({fr: [r for r in ev if r["frame"] == fr] for fr in frames})
    crowd_lbl = ["0", "1-2", "3-5", ">=6"]
    occl_lbl = bin_labels(OCCL_EDGES)
    size_lbl = bin_labels(SIZE_EDGES, " px")

    raw = recall_by_bin(ev, "crowd", CROWD_EDGES)
    raw_ci = bootstrap(lambda fs: [v for v, _ in recall_by_bin(ev, "crowd", CROWD_EDGES, fs)], frames, args.boot)
    adj = size_adjusted(ev)
    adj_ci = bootstrap(lambda fs: size_adjusted(ev, fs), frames, args.boot)
    occ = recall_by_bin(ev, "occl", OCCL_EDGES)
    occ_ci = bootstrap(lambda fs: [v for v, _ in recall_by_bin(ev, "occl", OCCL_EDGES, fs)], frames, args.boot)
    size = recall_by_bin(ev, "h", SIZE_EDGES)

    total = sum(r["hit"] for r in ev) / len(ev)
    summary = dict(
        source="VnTraffic (Zenodo 10.5281/zenodo.18195750, CC BY 4.0)", model=args.model, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, n_frames=len(frames), n_motorbikes=len(ev),
        n_excluded_small=len(rows) - len(ev),
        recall_all=round(total, 3), recall_all_iou50=round(sum(r["hit50"] for r in ev) / len(ev), 3),
        by_crowd=[dict(bin=b, n=n, recall=round(v, 3), ci95=[float(round(lo, 3)), float(round(hi, 3))],
                       size_adjusted=None if np.isnan(a) else float(round(a, 3)),
                       size_adjusted_ci95=[None if np.isnan(x) else float(round(x, 3)) for x in (alo, ahi)])
                  for b, (v, n), lo, hi, a, alo, ahi in zip(crowd_lbl, raw, *raw_ci, adj, *adj_ci)],
        by_occlusion=[dict(bin=b, n=n, recall=round(v, 3), ci95=[float(round(lo, 3)), float(round(hi, 3))])
                      for b, (v, n), lo, hi in zip(occl_lbl, occ, *occ_ci)],
        by_size=[dict(bin=b, n=n, recall=None if np.isnan(v) else round(v, 3)) for b, (v, n) in zip(size_lbl, size)],
        frames=frames_info,
        caveat="Các khung cùng 17 s video, tương quan mạnh: CI bootstrap theo khung là cận dưới của độ bất định.",
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(labels_x, series, title, xlabel, path):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        x = np.arange(len(labels_x))
        for k, (name, vals, lo, hi) in enumerate(series):
            v = np.array(vals, dtype=float)
            ax.errorbar(x + 0.08 * (k - (len(series) - 1) / 2), v * 100,
                        yerr=[np.clip(v - lo, 0, None) * 100, np.clip(hi - v, 0, None) * 100],
                        marker="o", capsize=4, label=name)
        ax.set_xticks(x, labels_x)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Recall xe máy (%)")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    ns = [n for _, n in raw]
    plot([f"{b}\n(n={n})" for b, n in zip(crowd_lbl, ns)],
         [("Thô", [v for v, _ in raw], *raw_ci), ("Chuẩn hoá theo cỡ box", adj, *adj_ci)],
         f"Recall theo độ đông ({args.model}, {len(frames)} khung VnTraffic)",
         f"Số xe máy lân cận (trong {CROWD_RADIUS:g} x chiều cao box)", out / "recall_vs_crowd.png")
    plot([f"{b}\n(n={n})" for b, (_, n) in zip(occl_lbl, occ)],
         [("Thô", [v for v, _ in occ], *occ_ci)],
         "Recall theo phần bị che", "Phần box bị xe gần hơn che", out / "recall_vs_occlusion.png")

    print(f"{len(frames)} khung, {len(ev)} xe máy (loại {len(rows) - len(ev)} xe cao < {MIN_H} px)")
    print(f"recall chung {total:.1%} (IoU {args.iou}), {summary['recall_all_iou50']:.1%} (IoU 0.5)")
    for d in summary["by_crowd"]:
        print(f"  lân cận {d['bin']:>4}: n={d['n']:4d} recall {d['recall']:.1%} CI {d['ci95']}"
              f" | chuẩn hoá cỡ {d['size_adjusted']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
