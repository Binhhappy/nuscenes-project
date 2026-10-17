"""Tổng hợp số liệu cho báo cáo: khoảng tin cậy 95% (bootstrap theo khung hình) + biểu đồ PNG.

Object trong cùng 1 khung không độc lập (cùng cảnh, cùng che khuất), nên bootstrap lấy mẫu lại theo KHUNG
(seed, sample), không theo object. Đọc kết quả đã có, thiếu file nào thì bỏ phần đó:
    output/t6/records.json               T6: thật / đối chứng / kịch bản VN (xe máy, ô tô, người)
    output/vn/records.json               vật thể VN ray-cast (recall vật cản không phân lớp)
    output/weather_eval/report*.json     Task B: detector theo thời tiết
Output: output/report/summary.json + output/report/*.png. Chạy được trên Windows (không cần GPU):
    python tools/stats_report.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

OUT = os.path.join(config.OUTPUT_DIR, "report")
# Bảng màu tham chiếu (categorical theo thứ tự cố định, đã kiểm định CVD), chữ dùng màu chữ, không dùng màu series
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#86b6ef", "#2a78d6", "#104281"]  # 1 hue nhạt -> đậm (số tia LiDAR tăng dần)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def bootstrap_recall(recs, n_boot=2000, seed=0):
    """Recall + CI 95% bootstrap theo khung (seed, sample)."""
    by = defaultdict(list)
    for r in recs:
        by[(r.get("seed"), r["sample"])].append(r["detected"])
    frames = [np.array(v, float) for v in by.values()]
    hits, ns = np.array([f.sum() for f in frames]), np.array([len(f) for f in frames])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(frames), (n_boot, len(frames)))
    boot = hits[idx].sum(1) / ns[idx].sum(1)
    return {"recall": float(hits.sum() / ns.sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_objects": int(ns.sum()), "n_frames": len(frames)}


def bootstrap_diff(recs_a, recs_b, n_boot=2000, seed=0):
    """Hiệu recall a - b (2 nhóm khung độc lập) + CI 95%; CI không chứa 0 = khác biệt có ý nghĩa."""
    def frames(recs):
        by = defaultdict(list)
        for r in recs:
            by[(r.get("seed"), r["sample"])].append(r["detected"])
        return np.array([sum(v) for v in by.values()], float), np.array([len(v) for v in by.values()], float)
    (ha, na), (hb, nb) = frames(recs_a), frames(recs_b)
    rng = np.random.default_rng(seed)
    ia, ib = rng.integers(0, len(ha), (n_boot, len(ha))), rng.integers(0, len(hb), (n_boot, len(hb)))
    d = ha[ia].sum(1) / na[ia].sum(1) - hb[ib].sum(1) / nb[ib].sum(1)
    return {"diff": float(ha.sum() / na.sum() - hb.sum() / nb.sum()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bar_ci(rows, title, path, xlabel="Recall (%)"):
    """Thanh ngang 1 series: recall + CI 95%, nhãn giá trị bằng màu chữ."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(rows))[::-1]
    v = [100 * r["recall"] for r in rows]
    lo = [100 * (r["recall"] - r["ci95"][0]) for r in rows]
    hi = [100 * (r["ci95"][1] - r["recall"]) for r in rows]
    ax.barh(y, v, color=C1, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(v, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
    for yi, vi, r in zip(y, v, rows):
        ax.text(min(vi + hi[list(y).index(yi)] + 1.5, 101), yi, f"{vi:.0f}%  (n={r['n_objects']})",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r["label"] for r in rows], color=INK)
    ax.set_xlim(0, 115)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def grouped_bars(cats, series, title, path, ylabel="Recall (%)"):
    """Cột nhóm, <= 3 series, legend + nhãn giá trị."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(cats))
    w = 0.8 / len(series)
    for k, (name, vals, color) in enumerate(series):
        pos = x - 0.4 + w / 2 + k * w
        ax.bar(pos, [np.nan if v is None else 100 * v for v in vals], width=w - 0.03, color=color, label=name,
               edgecolor=SURFACE, linewidth=2)
        for p, v in zip(pos, vals):
            if v is not None:
                ax.text(p, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, cats, color=INK, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def beam_chart(path):
    """Số điểm LiDAR trên vật thể theo khoảng cách: 32 tia (nuScenes) vs 64 / 128 tia giả định, cùng FOV dọc."""
    import matplotlib.pyplot as plt

    from task_a.raycast import beam_table, points_vs_distance
    dist = [5, 10, 15, 20, 30, 40, 50]
    objs = [("dog", "Chó (cao 0.6 m)"), ("moto_1", "Xe máy có người lái"), ("car", "Ô tô 4 chỗ")]
    table = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (name, label) in zip(axes, objs):
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        table[name] = {}
        for n, color in zip((32, 64, 128), SEQ):
            pts = points_vs_distance(name, dist, beam_table(n))
            table[name][n] = dict(zip(dist, pts))
            ax.plot(dist, np.maximum(pts, 0.8), color=color, lw=2, marker="o", ms=5, label=f"{n} tia")
        ax.axhline(5, color=INK2, lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Khoảng cách (m)", color=INK2)
    axes[0].set_ylabel("Số điểm trên vật thể (log)", color=INK2)
    axes[0].text(50, 5.6, "5 điểm", ha="right", fontsize=8, color=INK2)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Vật thể nhỏ gần như vô hình với LiDAR 32 tia ở xa (ray-cast, cùng FOV dọc -30.7°..+10.7°)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return table


def gallery(path, root=os.path.join(config.OUTPUT_DIR, "vn", "seed0"), xlim=25.0):
    """Mỗi kịch bản VN 1 khung BEV (khung có nhiều object chèn nhất): điểm keyframe + box chèn (cam) + tên."""
    import glob

    import matplotlib.pyplot as plt

    from utils.nuscenes_loader import load_bin
    from utils.visualization import to_box
    names = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))] if os.path.isdir(root) else []
    if not names:
        return
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        files = glob.glob(os.path.join(root, name, "*.json"))
        if not files:
            continue
        best = max(files, key=lambda f: json.load(open(f, encoding="utf-8"))["n_inserted"])
        with open(best, encoding="utf-8") as f:
            r = json.load(f)
        pts = load_bin(best[:-5] + ".bin")
        m = (np.abs(pts[:, 0]) < xlim) & (np.abs(pts[:, 1]) < xlim)
        ax.scatter(pts[m, 0], pts[m, 1], s=0.3, c="#9aa0a6", linewidths=0)
        for a in r["annotations_sensor"]:
            xy = to_box(a).bottom_corners()[:2]
            ax.plot(np.append(xy[0], xy[0, 0]), np.append(xy[1], xy[1, 0]), color=C2, lw=1.6)
            if a["category"].startswith("vn."):
                ax.text(a["translation"][0], a["translation"][1] + 1.3, a["category"][3:], fontsize=7,
                        color=INK, ha="center")
        ax.plot(0, 0, marker="^", color=INK, ms=7)
        ax.set(xlim=(-xlim, xlim), ylim=(-xlim, xlim), aspect="equal")
        ax.set_title(f"{name}: {r.get('scenario_description', '')}", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    # ---- T6 ----
    t6 = _load(os.path.join(config.OUTPUT_DIR, "t6", "records.json"))
    if t6:
        moto = lambda g: [r for r in t6 if r["group"] == g and r["cls"] == "motorcycle"]  # noqa: E731
        groups = [("real", "Xe máy thật (nuScenes)"), ("control_2w", "Đối chứng: chèn, đi cùng chiều"),
                  ("moto_swarm", "Bầy 10-20 xe máy"), ("cut_in", "Xe máy cắt đầu")]
        rows = [{**bootstrap_recall(moto(g)), "label": lab, "group": g} for g, lab in groups if moto(g)]
        summary["t6_motorcycle"] = rows
        summary["t6_swarm_minus_control"] = bootstrap_diff(moto("moto_swarm"), moto("control_2w"))
        bar_ci(rows, "Recall xe máy — detector pretrained nuScenes\n(CI 95%, bootstrap theo khung hình)",
               os.path.join(OUT, "t6_motorcycle_recall.png"))
        bins = [(1, 20), (20, 50), (50, 100), (100, 10 ** 6)]
        cats = ["1-19 điểm", "20-49", "50-99", "≥100"]
        def by_pts(g):
            rs = moto(g)
            return [np.mean([r["detected"] for r in rs if lo <= r["npts"] < hi]) if any(lo <= r["npts"] < hi for r in rs)
                    else None for lo, hi in bins]
        summary["t6_recall_by_points"] = {"bins": cats, "control_2w": by_pts("control_2w"), "moto_swarm": by_pts("moto_swarm")}
        grouped_bars(cats, [("Đối chứng", by_pts("control_2w"), C1), ("Trong bầy xe", by_pts("moto_swarm"), C2)],
                     "Cùng số điểm LiDAR, xe máy trong bầy vẫn bị bỏ sót nhiều hơn",
                     os.path.join(OUT, "t6_swarm_by_points.png"))
    # ---- vật thể VN ray-cast ----
    vn = _load(os.path.join(config.OUTPUT_DIR, "vn", "records.json"))
    import glob
    for extra in sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "vn", "records_*.json"))):
        redo = _load(extra)  # kịch bản chạy bổ sung / chạy lại (vd _ped): thay / thêm nhóm tương ứng
        if vn is not None and redo:
            groups = {r["group"] for r in redo if r["group"] != "real"}
            vn = [r for r in vn if r["group"] not in groups] + [r for r in redo if r["group"] in groups]
    if vn:
        labels = {"rc_control_car": "Đối chứng: ô tô ray-cast", "rc_control_moto": "Đối chứng: xe máy ray-cast",
                  "multi_rider": "Xe máy chở 2-3 người", "cargo_moto": "Xe máy chở hàng cồng kềnh",
                  "ba_gac_slow": "Xe ba gác", "tow_moto": "Xe máy kéo thùng hàng", "vendor_curb": "Xe hàng rong",
                  "dog_dart": "Chó lao ra đường", "cow_road": "Trâu / bò trên đường",
                  "ox_cart_road": "Xe bò (bò + xe)", "rush_hour": "Giờ cao điểm (mix)",
                  "market_street": "Phố chợ (mix)", "animal_in_traffic": "Chó giữa bầy xe (mix)",
                  "ped_group_cross": "Nhóm 3-8 người băng ngang", "ped_crowd": "Đám đông 10-20 người"}
        rows = []
        for g, lab in labels.items():
            rs = [r for r in vn if r["group"] == g]  # mọi object chèn của kịch bản (mix: cả xe máy cắt mẫu)
            if rs:
                rows.append({**bootstrap_recall(rs), "label": lab, "group": g,
                             "median_points": float(np.median([r["npts"] for r in rs]))})
        summary["vn_objects"] = rows
        # Người đi bộ: 1 người (jaywalk, T6) -> nhóm -> đám đông; recall theo số người chèn trong khung
        peds = {}
        jay = [r for r in (t6 or []) if r["group"] == "jaywalk"]
        if jay:
            peds["1 người (jaywalk)"] = bootstrap_recall(jay)
        for g, lab in (("ped_group_cross", "nhóm 3-8 người"), ("ped_crowd", "đám đông 10-20")):
            rs = [r for r in vn if r["group"] == g and r["cls"] == "pedestrian"]
            if rs:
                peds[lab] = bootstrap_recall(rs)
        by_n = defaultdict(list)
        for g in ("ped_group_cross", "ped_crowd"):
            frames = defaultdict(list)
            for r in vn:
                if r["group"] == g and r["cls"] == "pedestrian":
                    frames[(r["seed"], r["sample"])].append(r["detected"])
            for v in frames.values():
                by_n[min(len(v) // 4 * 4, 16)].append(np.mean(v))
        if peds:
            summary["pedestrians"] = {"recall": peds, "by_count": {f"{k}-{k + 3}" if k < 16 else "16+":
                                                                   {"mean_recall": float(np.mean(v)), "frames": len(v)}
                                                                   for k, v in sorted(by_n.items())}}
            bar_ci([{**v, "label": k} for k, v in peds.items()],
                   "Người đi bộ: 1 người / nhóm / đám đông\n(recall, CI 95% bootstrap theo khung)",
                   os.path.join(OUT, "pedestrians_recall.png"))
        # Xe máy ray-cast chưa giống xe máy cắt mẫu với detector (đối chứng ray-cast 64% vs cắt mẫu 94%), nên kịch bản
        # xe máy ray-cast chỉ so với CHÍNH đối chứng ray-cast; thêm recall "thấy vật cản" (không xét tên lớp).
        rc_ctrl = [r for r in vn if r["group"] == "rc_control_moto"]
        summary["vn_moto_vs_rc_control"] = {}
        for g in ("multi_rider", "cargo_moto"):
            rs = [r for r in vn if r["group"] == g]
            if rs and rc_ctrl:
                summary["vn_moto_vs_rc_control"][g] = {
                    "recall": bootstrap_recall(rs), "diff_vs_rc_control": bootstrap_diff(rs, rc_ctrl),
                    "obstacle_recall": float(np.mean([r["miss"] in (None,) or r["miss"].startswith("confused")
                                                      for r in rs]))}
        bar_ci(rows, "Detector pretrained trên vật thể VN ray-cast\n"
                     "(vật thể không có lớp nuScenes: có box bất kỳ tại chỗ = phát hiện)",
               os.path.join(OUT, "vn_objects_recall.png"))
    # ---- thời tiết ----
    for tag in ("", "_swarm"):
        rep = _load(os.path.join(config.OUTPUT_DIR, "weather_eval", f"report{tag}.json"))
        if not rep:
            continue
        conds = ["clear", "rain_light", "rain_moderate", "rain_heavy", "fog_light", "fog_moderate", "fog_dense", "storm"]
        names = {"clear": "trời trong", "rain_light": "mưa nhẹ", "rain_moderate": "mưa vừa", "rain_heavy": "mưa to",
                 "fog_light": "sương nhẹ", "fog_moderate": "sương vừa", "fog_dense": "sương dày", "storm": "giông/bão"}
        present = [c for c in conds if any(r["group"] == c for r in rep["rows"])]
        val = lambda c, cls: next((r["recall"] for r in rep["rows"] if r["group"] == c and r["cls"] == cls), None)  # noqa
        classes = ["car", "pedestrian", "motorcycle"] if not tag else ["motorcycle"]
        summary[f"weather{tag}"] = {cls: {c: val(c, cls) for c in present} for cls in classes}
        series = [(lab, [val(c, cls) for c in present], col) for cls, lab, col in
                  zip(classes, ["Ô tô", "Người đi bộ", "Xe máy"] if not tag else ["Xe máy trong bầy"], (C1, C2, C3))]
        grouped_bars([names[c] for c in present], series,
                     "Recall theo thời tiết" + (" — bầy xe máy" if tag else " — object thật mini_val"),
                     os.path.join(OUT, f"weather{tag}.png"))
    gallery(os.path.join(OUT, "vn_scenarios_gallery.png"))
    # ---- số tia LiDAR ----
    summary["beam_points"] = beam_chart(os.path.join(OUT, "lidar_beams.png"))
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "beam_points"}, indent=1, ensure_ascii=False)[:3000])
    print("->", OUT)


if __name__ == "__main__":
    main()
