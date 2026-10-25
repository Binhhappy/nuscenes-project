"""Bước 0 — cắt khung từ video giao thông Việt Nam để tự gán nhãn xe máy (CVAT, định dạng YOLO 1.1).

Chạy trong env có opencv (env ultralytics cũng được):
    python tools/vn_moto_frames.py --video VNTraffic_Original-video.mp4 --gt VNTraffic_GroundTruth.txt

Nguồn: VnTraffic trong "Vehicle Tracking Dataset" (Zenodo 10.5281/zenodo.18195750, CC BY 4.0), camera
ngã tư Quang Trung - Lê Trọng Tấn, Hà Nội, 1920x1080, 30 fps, 510 khung (17 s).

Đầu ra (mặc định ../output/step0/cvat/ + cvat_import.zip):
    obj.names, obj.data, train.txt, obj_train_data/<khung>.jpg + <khung>.txt
Box gợi ý lấy từ nhãn gốc VnTraffic (người gán, MOT, không có lớp) với lớp `review`: người gán đổi
thành `motorbike` hoặc xoá (ô tô, xe buýt, xe tải), rồi vẽ thêm mọi xe máy còn thiếu. Không gợi ý
bằng YOLO: nhãn sinh từ chính detector cần đo sẽ làm recall cao giả.

Luật gán nhãn (xem LABEL_RULES bên dưới, cũng ghi vào README.txt của thư mục đầu ra).
"""
import argparse
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

NAMES = ["motorbike", "review"]  # thứ tự lớp = id lớp YOLO
LABEL_RULES = """Luật gán nhãn bước 0 (chỉ 1 lớp cần giữ: motorbike)
1. Gán MỌI xe máy / xe máy điện nhìn thấy, kể cả bị che tới ~70%, kể cả trong cụm đông.
2. Box bao cả xe VÀ người ngồi trên xe (người lái + người ngồi sau + hàng chở).
3. Bị che: box chỉ bao phần nhìn thấy, không đoán phần bị che.
4. Xe quá nhỏ (cao < 24 px) vẫn gán nếu thấy rõ là xe máy; đánh giá sẽ tự loại khỏi recall.
5. Box `review` (gợi ý từ nhãn gốc): đổi sang motorbike nếu là xe máy, còn lại XOÁ.
   Khi export không được còn box `review` nào.
6. Không gán: xe đạp, ô tô, xe buýt, xe tải, người đi bộ, xe máy đang dựng đỗ trên vỉa hè.
"""


def read_mot(path):
    """Nhãn MOT: frame,id,x,y,w,h,conf,... -> {frame: [(x, y, w, h)]}."""
    boxes = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        boxes[int(f[0])].append(tuple(float(v) for v in f[2:6]))
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", help="nhãn MOT gốc để gợi ý box `review` (bỏ trống = không gợi ý)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "output" / "step0" / "cvat"))
    ap.add_argument("--stride", type=int, default=15, help="lấy 1 khung mỗi N khung (15 = 0.5 s ở 30 fps)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="khung cuối (-1 = hết video)")
    ap.add_argument("--prefix", default="vntraffic", help="tiền tố tên file khung")
    args = ap.parse_args()

    out = Path(args.out)
    data_dir = out / "obj_train_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    gt = read_mot(args.gt) if args.gt else {}

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = n_frames - 1 if args.end < 0 else min(args.end, n_frames - 1)
    names, n_review = [], 0
    for idx in range(args.start, end + 1, args.stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            print(f"  bỏ khung {idx}: không đọc được")
            continue
        h, w = img.shape[:2]
        stem = f"{args.prefix}_{idx:06d}"
        cv2.imwrite(str(data_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for x, y, bw, bh in gt.get(idx, []):
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
            if x1 <= x0 or y1 <= y0:
                continue
            lines.append(f"1 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        n_review += len(lines)
        (data_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        names.append(stem)
    cap.release()

    (out / "obj.names").write_text("\n".join(NAMES) + "\n")
    (out / "obj.data").write_text(f"classes = {len(NAMES)}\nnames = obj.names\ntrain = train.txt\n")
    (out / "train.txt").write_text("".join(f"obj_train_data/{s}.jpg\n" for s in names))
    (out / "README.txt").write_text(LABEL_RULES, encoding="utf-8")

    zip_path = out.parent / "cvat_import.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["obj.names", "obj.data", "train.txt"]:
            z.write(out / f, f)
        for s in names:
            z.write(data_dir / f"{s}.txt", f"obj_train_data/{s}.txt")
    print(f"{len(names)} khung -> {data_dir}")
    print(f"{n_review} box gợi ý `review` (cần đổi thành motorbike hoặc xoá)")
    print(f"CVAT: tạo task, upload ảnh trong {data_dir}, rồi Upload annotations > YOLO 1.1 > {zip_path}")


if __name__ == "__main__":
    main()
