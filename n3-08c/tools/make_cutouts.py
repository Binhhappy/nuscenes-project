"""Tách nền ảnh tải về (assets/vn_objects/_raw) thành PNG nền trong suốt cho utils/camera_paste.py.

OpenCV GrabCut khởi tạo bằng khung chữ nhật quanh vật thể + khung gợi ý chắc chắn vật thể / nền (toạ độ tỉ lệ 0-1
theo ảnh, chọn tay theo ảnh xem trước),
giữ thành phần liên thông lớn nhất, làm mềm mép alpha, cắt sát, lật để đầu vật thể quay sang phải (quy ước side.png).
Ghi assets/vn_objects/<loại>/<mặt>.png và SOURCES.md (nguồn + giấy phép từ _raw/sources.json).

Chạy: python tools/make_cutouts.py            (ảnh kiểm tra: output/vis/cutouts.png)
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

ASSETS = Path(config.PROJECT_ROOT) / "assets" / "vn_objects"
RAW = ASSETS / "_raw"
# raw key -> (thư mục loại, mặt, khung (x0, y0, x1, y1) tỉ lệ, lật ngang để đầu quay sang phải,
#             gợi ý {"fg": [khung chắc chắn là vật thể], "bg": [khung chắc chắn là nền]})
CUTS = {
    "cow": ("cow", "side", (0.17, 0.10, 0.84, 0.80), False, {}),
    "dog": ("dog", "side", (0.07, 0.08, 0.93, 0.98), True,
            {"fg": [(0.20, 0.20, 0.85, 0.55)], "bg": [(0.37, 0.72, 0.70, 0.98), (0.0, 0.80, 0.22, 1.0)]}),
    "vendor_cart": ("vendor_cart", "side", (0.04, 0.40, 0.50, 0.97), False,
                    {"fg": [(0.12, 0.50, 0.45, 0.70)], "bg": [(0.04, 0.93, 0.50, 1.0)]}),
    "moto_multi": ("moto_multi", "side", (0.54, 0.18, 0.87, 0.94), False,
                   {"fg": [(0.62, 0.32, 0.76, 0.80)], "bg": [(0.79, 0.18, 0.87, 0.45)]}),
    "ox_wagon_alt": ("ox_wagon", "side", (0.18, 0.36, 0.52, 0.87), False,
                     {"fg": [(0.22, 0.42, 0.48, 0.68)], "bg": [(0.18, 0.84, 0.52, 0.87)]}),
    "ba_gac": ("ba_gac", "side", (0.0, 0.55, 0.88, 1.0), True,
               {"fg": [(0.30, 0.62, 0.84, 0.84)], "bg": [(0.0, 0.55, 0.06, 0.92), (0.15, 0.95, 0.75, 1.0)]}),
}
# Lớp dùng chung ảnh với lớp khác: xe máy nuScenes được chèn luôn có người lái (cycle.with_rider), ảnh xe máy
# không người lái (_raw/motorcycle.png) dính nền nên không dùng; dùng ảnh xe máy có người lái.
ALIASES = {"motorcycle": "moto_multi"}
MAX_SIDE = 1000  # px: GrabCut chạy trên ảnh thu nhỏ cho nhanh


def cutout(img: np.ndarray, rect_frac, iters: int = 6, hints: dict = None) -> np.ndarray:
    """BGR -> BGRA, alpha từ GrabCut trong khung rect_frac; hints: khung tỉ lệ chắc chắn fg / bg."""
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    x0, y0, x1, y1 = rect_frac
    rect = (int(x0 * sw), int(y0 * sh), max(1, int((x1 - x0) * sw) - 1), max(1, int((y1 - y0) * sh) - 1))
    mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_PR_FGD
    for kind, val in (("bg", cv2.GC_BGD), ("fg", cv2.GC_FGD)):
        for bx0, by0, bx1, by1 in (hints or {}).get(kind, []):
            mask[int(by0 * sh):int(by1 * sh), int(bx0 * sw):int(bx1 * sw)] = val
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:  # giữ thành phần lớn nhất + thành phần >= 15% của nó (vd bánh xe rời thân)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= 0.15 * areas.max())
        fg = np.isin(lab, keep).astype(np.uint8)
    alpha = cv2.resize(fg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 1.0 / s))
    return np.dstack([img, (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])


def crop_alpha(bgra: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(bgra[..., 3] > 10)
    return bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    meta = json.loads((RAW / "sources.json").read_text(encoding="utf-8"))
    tiles, lines = [], ["# Nguồn ảnh (assets/vn_objects)", "",
                        "Ảnh gốc tải từ Wikimedia Commons, tách nền bằng `n3-08c/tools/make_cutouts.py` (OpenCV GrabCut).",
                        "Ảnh đã tách nền là tác phẩm phái sinh: giữ nguyên giấy phép của ảnh gốc (CC BY-SA -> CC BY-SA).", "",
                        "| File | Ảnh gốc | Tác giả | Giấy phép |", "|---|---|---|---|"]
    for raw_key, (folder, view, rect, flip, hints) in CUTS.items():
        src = next((RAW / f"{raw_key}{e}" for e in (".jpg", ".png") if (RAW / f"{raw_key}{e}").is_file()), None)
        if src is None:
            print("thiếu", raw_key)
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        out = crop_alpha(cutout(img, rect, hints=hints))
        if flip:
            out = out[:, ::-1]
        (ASSETS / folder).mkdir(parents=True, exist_ok=True)
        dst = ASSETS / folder / f"{view}.png"
        cv2.imwrite(str(dst), out)
        m = meta[raw_key]
        lines.append(f"| `{folder}/{view}.png` | [{m['title']}]({m['page']}) | {m['artist']} | {m['license']} |")
        print(f"{dst.relative_to(ASSETS)}  {out.shape[1]}x{out.shape[0]}  <- {src.name}")
        # ô kiểm tra: ảnh tách nền trên nền ô cờ
        k = 260 / max(out.shape[:2])
        t = cv2.resize(out, (max(1, int(out.shape[1] * k)), max(1, int(out.shape[0] * k))))
        board = ((np.indices(t.shape[:2]).sum(0) // 16) % 2 * 60 + 150).astype(np.uint8)[..., None].repeat(3, 2)
        a = t[..., 3:4] / 255.0
        cell = np.full((290, 270, 3), 255, np.uint8)
        cell[:t.shape[0], :t.shape[1]] = (t[..., :3] * a + board * (1 - a)).astype(np.uint8)
        cv2.putText(cell, f"{folder}/{view}", (4, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(cell)
    for alias, folder in ALIASES.items():
        for f in (ASSETS / folder).glob("*.png"):
            (ASSETS / alias).mkdir(parents=True, exist_ok=True)
            (ASSETS / alias / f.name).write_bytes(f.read_bytes())
            lines.append(f"| `{alias}/{f.name}` | bản sao `{folder}/{f.name}` | | |")
            print(f"{alias}/{f.name}  <- bản sao {folder}/{f.name}")
    (ASSETS / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    while len(tiles) % 4:
        tiles.append(np.full((290, 270, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    os.makedirs(config.VIS_DIR, exist_ok=True)
    png = os.path.join(config.VIS_DIR, "cutouts.png")
    cv2.imwrite(png, grid)
    print("saved", png)


if __name__ == "__main__":
    main()
