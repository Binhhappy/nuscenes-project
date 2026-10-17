"""Dán ảnh 2D của object chèn lên ảnh camera đích (cut-paste), cùng ý tưởng Task A trên LiDAR: ảnh object lấy từ
chính ảnh camera của annotation gốc trong nuScenes, không vẽ / không sinh bằng AI.

Luật:
    1. Chọn ảnh nguồn rõ nhất cho đúng point cloud đó (source_candidates + pick_source):
       ứng viên = mọi camera của sample nguồn (cùng thời điểm với điểm LiDAR đã cắt) và, dự phòng, cùng instance ở
       NEIGHBOR_FRAMES keyframe trước / sau (0.5 s mỗi keyframe).
       Loại (cổng bắt buộc):
         - box không nằm trọn trong ảnh (bị cắt mép)
         - visibility nuScenes < 60% (MIN_VISIBILITY)
         - bị che > MAX_SRC_OCCLUSION, lấy max của 2 cách đo:
             box: diện tích box nguồn bị box có nhãn khác (gần camera hơn) phủ
             LiDAR: điểm LiDAR cùng khung chiếu vào vùng box mà gần camera hơn mặt trước box > OCC_DEPTH_MARGIN
                    (vật chắn KHÔNG có nhãn: cây, cột, biển báo)
         - góc nhìn lệch so với camera đích > MAX_VIEW_DIFF (mặt object thấy được khác nhau, vd đầu xe vs đuôi xe)
         - box quá nhỏ trên ảnh (< MIN_SRC_PX px chiều cao: ít chi tiết)
         - ảnh nhoè: độ nét < MIN_SHARP (chuyển động, thiếu sáng)
         - ánh sáng không khớp: độ sáng trung bình ảnh nguồn / ảnh đích lệch > MAX_LUM_RATIO lần (vd ảnh đêm, đèn
           pha chói dán vào cảnh ban ngày)
         - điều kiện cảnh khác: mô tả scene nuScenes của nguồn và đích không cùng có / không có "night", "rain"
         Hai cổng ánh sáng + điều kiện cảnh là cổng mềm (RELAX_LIGHTING): không ứng viên nào đạt thì vẫn dán ứng viên
         tốt nhất qua các cổng còn lại (trạng thái pasted_relaxed, màu kéo mạnh hơn về nền đích: COLOR_MATCH_RELAXED).
         Lý do: nuScenes mini có ít mẫu hiếm (vd xe máy có người lái, gần như chỉ ở cảnh đêm / mưa); có ảnh rõ của
         chính object đó vẫn minh hoạ tốt hơn chỉ có box.
       Xếp hạng: ưu tiên đúng khung của point cloud (lệch khung nhỏ nhất), rồi điểm
           Q = sqrt(diện tích box px) * (1 - tỉ lệ bị che) * độ nét * cos(lệch góc nhìn)
       độ nét = min(1, phương sai Laplacian vùng box / SHARP_REF): ảnh nhoè do chuyển động / thiếu sáng bị trừ điểm.
    2. Góc nhìn: hướng camera nhìn object được đo trong object frame ở nguồn và đích. Task A giữ góc nhìn của LiDAR
       (view_angle) nên phần lớn object có ứng viên đạt.
    3. Hình học: 8 góc box nguồn (pixel camera nguồn) <-> 8 góc box đích (pixel camera đích), cùng thứ tự góc
       (nuscenes Box.corners). Homography (bình phương tối thiểu) biến ảnh nguồn sang ảnh đích: xử lý tỉ lệ theo khoảng
       cách và phối cảnh. Box 3D không phẳng nên homography là xấp xỉ; sai số chiếu lại góc > MAX_REPROJ (theo đường
       chéo box) thì bỏ.
    4. Mask: bao lồi 8 góc chiếu (gồm cả nền trong box), mềm mép. Trừ vùng object thật gần camera hơn (che khuất).
    5. Màu: kéo mean / std từng kênh của vùng dán nửa đường về phía nền quanh vị trí đích (khác giờ / thời tiết).
    6. Thứ tự: dán object xa trước, gần sau (object gần che object xa).
    7. Vật thể không có trong nuScenes (ray-cast: vn.ba_gac, vn.cow, ...) hoặc ảnh gốc không đạt: dán ảnh PNG nền
       trong suốt tải về, ASSET_DIR/<loại>/front.png | rear.png | side.png (side: đầu quay sang phải). Chọn mặt theo
       góc camera nhìn object (|az| <= 45 độ: front, >= 135: rear, còn lại: side, lật ngang theo hướng đầu), cao bằng
       chiều cao box chiếu, đáy chạm tâm đáy box, giữ tỉ lệ ảnh, thêm bóng đổ mờ; che khuất + màu như trên.
       Thứ tự nguồn: ảnh nuScenes gốc qua mọi cổng -> ảnh tải về -> ảnh nuScenes gốc khác điều kiện sáng.
Giới hạn: ảnh nuScenes mang theo nền quanh object; ảnh tải về chỉ có 3 mặt (góc chéo dùng mặt gần nhất) và không
khớp từng chi tiết với point cloud ray-cast; ánh sáng, bóng đổ không khớp hẳn. Chỉ để minh hoạ, các số đo của đề tài
là trên LiDAR.

Chạy demo: python utils/camera_paste.py   (lưu output/vis/camera_paste_<token>.png: trước / sau cho 6 camera)
"""
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_VISIBILITY = 3      # visibility_token nuScenes: 1 = 0-40%, 2 = 40-60%, 3 = 60-80%, 4 = 80-100%
MAX_VIEW_DIFF = 30.0    # (độ) lệch góc nhìn object tối đa giữa camera nguồn và camera đích
MAX_SRC_OCCLUSION = 0.1  # tỉ lệ box nguồn bị che tối đa (box có nhãn khác hoặc điểm LiDAR phía trước)
OCC_DEPTH_MARGIN = 0.5  # (m) điểm LiDAR gần camera hơn góc gần nhất của box quá ngưỡng này = vật chắn
MIN_SRC_PX = 24         # (px) chiều cao box nguồn tối thiểu trên ảnh 1600x900
SHARP_REF = 100.0       # phương sai Laplacian coi là đủ nét
MIN_SHARP = 0.25        # độ nét tối thiểu (theo thang SHARP_REF)
MAX_LUM_RATIO = 1.8     # độ sáng trung bình ảnh nguồn / ảnh đích lệch tối đa (lần)
CONDITION_TAGS = ("night", "rain")  # từ khoá trong scene description nuScenes phải khớp giữa nguồn và đích
NEIGHBOR_FRAMES = 2     # số keyframe trước / sau của cùng instance dùng làm ứng viên dự phòng
MAX_REPROJ = 0.15       # sai số chiếu lại góc box tối đa (tỉ lệ đường chéo box đích)
FEATHER = 5             # (px, ảnh 1600x900) độ mềm mép mask
COLOR_MATCH = 0.5       # 0 = giữ màu nguồn, 1 = khớp hẳn thống kê màu nền đích
COLOR_MATCH_RELAXED = 0.8  # khi dán ảnh khác điều kiện sáng (pasted_relaxed)
RELAX_LIGHTING = True   # không có ảnh nguồn cùng điều kiện sáng thì vẫn dán (cổng mềm)
W, H = 1600, 900


def _view_azimuth(center: np.ndarray, rot: np.ndarray, cam_pos: np.ndarray) -> float:
    """Azimuth (rad) của camera nhìn từ object, trong object frame (rot: ma trận xoay object -> frame chung)."""
    d = rot.T @ (cam_pos - center)
    return float(np.arctan2(d[1], d[0]))


def scene_conditions(nusc, sample_token: str) -> frozenset:
    """Tập điều kiện (CONDITION_TAGS) có trong mô tả scene của sample."""
    desc = nusc.get("scene", nusc.get("sample", sample_token)["scene_token"])["description"].lower()
    return frozenset(t for t in CONDITION_TAGS if t in desc)


def _instance_neighbors(nusc, ann: dict, n: int) -> List[Tuple[int, dict]]:
    """[(lệch khung, annotation)] của cùng instance: 0 = chính nó, ±1..±n keyframe trước / sau."""
    out = [(0, ann)]
    for key, sign in (("prev", -1), ("next", 1)):
        a, k = ann, 0
        while a[key] and k < n:
            a, k = nusc.get("sample_annotation", a[key]), k + 1
            out.append((sign * k, a))
    return out


def _lidar_occlusion(nusc, sample_token: str, cam: str, mask: np.ndarray, near_depth: float,
                     cache: Optional[dict] = None) -> float:
    """Tỉ lệ điểm LiDAR (cùng khung) chiếu vào mask mà nằm trước box: depth < near_depth - OCC_DEPTH_MARGIN."""
    from utils.camera_proj import lidar_to_cam
    from utils.nuscenes_loader import get_pointcloud
    key = ("lidar_cam", sample_token, cam)
    if cache is None or key not in cache:
        T, K, _ = lidar_to_cam(nusc, sample_token, cam)
        c = get_pointcloud(nusc, sample_token)[:, :3] @ T[:3, :3].T + T[:3, 3]
        c = c[c[:, 2] > 0.5]
        uv = np.round((c @ K.T)[:, :2] / c[:, 2:3]).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        val = (uv[ok], c[ok, 2])
        if cache is not None:
            cache[key] = val
    else:
        val = cache[key]
    uv, depth = val
    inside = mask[uv[:, 1], uv[:, 0]]
    if inside.sum() < 5:
        return 0.0
    return float((depth[inside] < near_depth - OCC_DEPTH_MARGIN).mean())


def source_candidates(nusc, ann_token: str, cache: Optional[dict] = None):
    """Ứng viên ảnh nguồn đã qua các cổng không phụ thuộc camera đích (trọn trong ảnh, visibility, bị che, kích
    thước). Mỗi ứng viên: dict path, uv (8, 2) góc box pixel, az (azimuth camera trong object frame), offset (lệch
    khung so với annotation gốc), camera, area, occ, sharp. Trả về (list, lý do nếu rỗng)."""
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    if cache is not None and ann_token in cache:
        return cache[ann_token]
    try:
        ann0 = nusc.get("sample_annotation", ann_token)
    except KeyError:  # object ray-cast: không có annotation / ảnh thật
        return [], "no_source"
    cands, reason = [], "source_not_in_camera"
    for offset, ann in _instance_neighbors(nusc, ann0, NEIGHBOR_FRAMES):
        if int(ann["visibility_token"] or 0) < MIN_VISIBILITY:
            reason = "source_occluded"
            continue
        for ch, sd in nusc.get("sample", ann["sample_token"])["data"].items():
            if not ch.startswith("CAM_"):
                continue
            path, boxes, K = nusc.get_sample_data(sd, box_vis_level=BoxVisibility.ANY)
            me = [b for b in boxes if b.token == ann["token"]]
            if not me:
                continue
            b = me[0]  # camera frame: camera ở gốc
            uv = view_points(b.corners(), K, normalize=True)[:2].T.astype(np.float32)
            if uv.min() < 0 or uv[:, 0].max() >= W or uv[:, 1].max() >= H:
                continue  # box bị cắt mép ảnh
            if uv[:, 1].max() - uv[:, 1].min() < MIN_SRC_PX:
                reason = "source_too_small"
                continue
            mask = _hull_mask(uv) > 0
            covered = np.zeros_like(mask)
            for o in boxes:  # box thật khác, gần camera hơn
                if o.token != b.token and np.linalg.norm(o.center) < np.linalg.norm(b.center):
                    ouv = view_points(o.corners(), K, normalize=True)[:2].T
                    if (o.corners()[2] > 0.1).all():
                        covered |= _hull_mask(ouv) > 0
            occ = max(float((mask & covered).sum() / max(mask.sum(), 1)),
                      _lidar_occlusion(nusc, ann["sample_token"], ch, mask, float(b.corners()[2].min()), cache))
            if occ > MAX_SRC_OCCLUSION:
                reason = "source_occluded"
                continue
            x0, y0 = np.floor(uv.min(0)).astype(int)
            x1, y1 = np.ceil(uv.max(0)).astype(int)
            full = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            sharp = float(min(1.0, cv2.Laplacian(full[y0:y1 + 1, x0:x1 + 1], cv2.CV_64F).var() / SHARP_REF))
            if sharp < MIN_SHARP:
                reason = "source_blurry"
                continue
            cands.append({"path": path, "uv": uv, "az": _view_azimuth(b.center, b.rotation_matrix, np.zeros(3)),
                          "offset": offset, "camera": ch, "area": float(mask.sum()), "occ": occ, "sharp": sharp,
                          "lum": float(full.mean()), "cond": scene_conditions(nusc, ann["sample_token"])})
    out = (cands, None if cands else reason)
    if cache is not None:
        cache[ann_token] = out
    return out


def pick_source(cands: Sequence[dict], target_az: float, target_lum: Optional[float] = None,
                target_cond: Optional[frozenset] = None, relax: bool = RELAX_LIGHTING
                ) -> Tuple[Optional[dict], Optional[str]]:
    """Ứng viên rõ nhất cho camera đích: qua cổng điều kiện cảnh + ánh sáng + góc nhìn, ưu tiên đúng khung
    (|offset| nhỏ), rồi điểm Q lớn nhất. target_lum: độ sáng trung bình ảnh đích (xám, 0-255).
    target_cond: scene_conditions của sample đích. relax: không ai qua cổng ánh sáng / điều kiện cảnh thì chọn lại
    bỏ 2 cổng đó (kết quả có relaxed=True)."""
    best, why = _pick(cands, target_az, target_lum, target_cond)
    if best is None and relax and why in ("condition_mismatch", "lighting_mismatch"):
        best, why = _pick(cands, target_az, None, None)
        if best is not None:
            best["relaxed"] = True
    return (best, None) if best else (None, why)


def _pick(cands, target_az, target_lum, target_cond):
    best, best_key, why = None, None, "view_mismatch"
    for c in cands:
        if target_cond is not None and c["cond"] != target_cond:
            why = "condition_mismatch"
            continue
        if target_lum and abs(np.log(max(c["lum"], 1) / max(target_lum, 1))) > np.log(MAX_LUM_RATIO):
            why = "lighting_mismatch"
            continue
        diff = abs(np.degrees(np.angle(np.exp(1j * (target_az - c["az"])))))
        if diff > MAX_VIEW_DIFF:
            why = "view_mismatch"
            continue
        q = np.sqrt(c["area"]) * (1 - c["occ"]) * c["sharp"] * np.cos(np.radians(diff))
        key = (-abs(c["offset"]), q)
        if best_key is None or key > best_key:
            best, best_key = dict(c, view_diff=float(diff), score=float(q), relaxed=False), key
    return best, why


def _hull_mask(uv: np.ndarray) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    return m


def _match_color(patch: np.ndarray, ref: np.ndarray, k: float = COLOR_MATCH) -> np.ndarray:
    """patch, ref: (N, 3) float. Kéo mean / std của patch về ref với hệ số k."""
    mp, sp = patch.mean(0), patch.std(0) + 1e-6
    mr, sr = ref.mean(0), ref.std(0) + 1e-6
    mt, st = mp + k * (mr - mp), sp + k * (sr - sp)
    return np.clip((patch - mp) / sp * st + mt, 0, 255)


# ảnh tải về: <repo>/assets/vn_objects/<loại>/{front,rear,side}.png; đổi thư mục bằng env N308C_ASSET_DIR
ASSET_DIR = Path(os.environ.get("N308C_ASSET_DIR", Path(__file__).resolve().parents[2] / "assets" / "vn_objects"))
# mặt -> góc nhìn tâm (độ, |azimuth camera trong object frame|); side / oblique_*: đầu object quay sang PHẢI ảnh.
# Ý tưởng 5 mặt + nhiều biến thể / mặt lấy từ repo nhóm (buildPhasev1.2 nhánh v3, utils/asset_manager.py).
ASSET_VIEWS = {"front": 0.0, "oblique_front": 45.0, "side": 90.0, "oblique_rear": 135.0, "rear": 180.0}
SHADOW = 0.35           # độ tối bóng đổ dưới ảnh tải về (0 = tắt)


def asset_key(category: str) -> str:
    """Thư mục ảnh của category: vn.ba_gac -> ba_gac, vehicle.motorcycle -> motorcycle,
    human.pedestrian.adult -> pedestrian."""
    parts = category.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def load_asset(key: str, cache: Optional[dict] = None) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """{view: [(tên file, ảnh RGBA cắt sát vùng alpha > 0), ...]} từ ASSET_DIR/<key>/<view>.png và các biến thể
    <view>_<bất kỳ>.png (vd side_2.png). Rỗng nếu chưa có ảnh."""
    ck = ("asset", key)
    if cache is not None and ck in cache:
        return cache[ck]
    views = {}
    for v in ASSET_VIEWS:
        files = [ASSET_DIR / key / f"{v}.png"] + sorted((ASSET_DIR / key).glob(f"{v}_*.png"))
        for f in files:
            im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.is_file() else None
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue  # cần PNG có kênh alpha (nền trong suốt)
            ys, xs = np.nonzero(im[..., 3] > 10)
            if len(xs) < 10:
                continue
            views.setdefault(v, []).append(
                (f.name, cv2.cvtColor(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1], cv2.COLOR_BGRA2RGBA)))
    if cache is not None:
        cache[ck] = views
    return views


def _asset_view(az: float, views) -> Optional[str]:
    """Mặt object camera thấy: mặt có góc tâm (ASSET_VIEWS) gần |azimuth| nhất trong số mặt có ảnh
    (azimuth camera trong object frame, 0 = camera ở phía đầu object)."""
    a = abs(np.degrees(az))
    have = [v for v in ASSET_VIEWS if v in views]
    return min(have, key=lambda v: abs(ASSET_VIEWS[v] - a)) if have else None


def _place_asset(rgba: np.ndarray, view: str, box, T: np.ndarray, K: np.ndarray):
    """Đặt ảnh RGBA lên khung ảnh đích theo box: cao bằng chiều cao box chiếu tại tâm box, đáy chạm điểm chiếu của
    tâm đáy box, giữ tỉ lệ ảnh; side.png lật ngang khi đầu object hướng sang trái ảnh.
    Returns (rgb float (H, W, 3), mask uint8 (H, W), (u, v đáy, rộng px)) hoặc None."""
    from utils.camera_proj import project
    l, h = box.wlh[1], box.wlh[2]
    local = np.array([[0, 0, -h / 2], [0, 0, h / 2], [l / 2, 0, 0], [-l / 2, 0, 0]])
    uv, _, front = project(T, K, local @ box.rotation_matrix.T + box.center, W, H)
    if not front.all():
        return None
    (ub, vb), vt = uv[0], uv[1, 1]
    ph = vb - vt
    if ph < 8:
        return None
    if view in ("side", "oblique_front", "oblique_rear") and uv[2, 0] < uv[3, 0]:  # đầu object (x+) bên trái ảnh
        rgba = rgba[:, ::-1]
    pw = max(1, int(round(rgba.shape[1] * ph / rgba.shape[0])))
    im = cv2.resize(np.ascontiguousarray(rgba), (pw, max(1, int(round(ph)))),
                    interpolation=cv2.INTER_AREA).astype(np.float32)
    x0, y0 = int(round(ub - pw / 2)), int(round(vb)) - im.shape[0]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + pw, W), min(y0 + im.shape[0], H)
    if xa >= xb or ya >= yb:
        return None
    rgb, mask = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.uint8)
    rgb[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0, :3]
    mask[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0, 3].astype(np.uint8)
    return rgb, mask, (ub, vb, pw)


def _warp_source(src: dict, dst: np.ndarray, cache: Optional[dict]):
    """Ảnh nuScenes nguồn -> khung đích bằng homography 8 góc box. (rgb, mask) hoặc None nếu khớp kém."""
    key = ("img", src["path"])
    if cache is None or key not in cache:
        im = cv2.cvtColor(cv2.imread(src["path"]), cv2.COLOR_BGR2RGB)
        if cache is not None:
            cache[key] = im
    else:
        im = cache[key]
    Hm, _ = cv2.findHomography(src["uv"], dst.astype(np.float32), 0)
    if Hm is None:
        return None
    fit = cv2.perspectiveTransform(src["uv"][None], Hm)[0]
    if np.linalg.norm(fit - dst, axis=1).mean() > MAX_REPROJ * np.linalg.norm(dst.max(0) - dst.min(0)):
        return None
    warped = cv2.warpPerspective(im, Hm, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    valid = cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8), Hm, (W, H))
    return warped, cv2.bitwise_and(_hull_mask(dst), valid)


def paste_objects(nusc, img: np.ndarray, T: np.ndarray, K: np.ndarray, inserted: Sequence[dict],
                  real: Sequence[dict] = (), cache: Optional[dict] = None, target_cond: Optional[frozenset] = None
                  ) -> Tuple[np.ndarray, List[str], List[Optional[dict]]]:
    """Dán ảnh object chèn lên img (RGB 1600x900, ảnh camera đích). T: LIDAR_TOP -> camera, K: intrinsic.

    inserted: annotation sensor frame có source_ann_token. real: annotation object thật (để tính che khuất).
    target_cond: scene_conditions(sample đích); None = bỏ qua cổng điều kiện cảnh.
    Returns (ảnh mới, trạng thái từng object theo thứ tự inserted, ảnh nguồn đã chọn hoặc None).
    Nguồn theo thứ tự: (1) ảnh nuScenes gốc qua mọi cổng -> (2) ảnh tải về ASSET_DIR/<asset_key>/<view>.png ->
    (3) ảnh nuScenes gốc khác điều kiện sáng (RELAX_LIGHTING).
    Trạng thái: pasted / pasted_asset / pasted_relaxed / not_in_view / no_source / source_occluded /
    source_too_small / source_blurry / source_not_in_camera / condition_mismatch / lighting_mismatch /
    view_mismatch / bad_fit.
    Nguồn đã dùng: ảnh nuScenes {camera, offset (lệch khung), score, view_diff (độ), sharp, occ, relaxed},
    ảnh tải về {asset, view}.
    """
    from utils.camera_proj import project
    from utils.visualization import to_box
    out = img.astype(np.float32)
    cam_pos = np.linalg.inv(T)[:3, 3]  # vị trí camera trong LIDAR frame
    target_lum = float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())
    status = ["not_in_view"] * len(inserted)
    sources: List[Optional[dict]] = [None] * len(inserted)

    def cam_depth(b):
        return float(np.linalg.norm(np.asarray(b["translation"]) - cam_pos))

    real_views = []
    for b in real:
        uv, ok, front = project(T, K, to_box(b).corners().T, W, H)
        if front.all() and ok.any():
            real_views.append((cam_depth(b), _hull_mask(uv)))

    for i in sorted(range(len(inserted)), key=lambda i: -cam_depth(inserted[i])):  # xa trước, gần sau
        b = inserted[i]
        box = to_box(b)
        dst, ok, front = project(T, K, box.corners().T, W, H)
        if not front.all() or not ok.any():
            continue
        az = _view_azimuth(box.center, box.rotation_matrix, cam_pos)
        cands, why = source_candidates(nusc, b.get("source_ann_token") or "", cache)
        src, why = pick_source(cands, az, target_lum, target_cond, relax=False) if cands else (None, why)
        placed, kind, shadow = None, None, None
        if src is not None:  # (1) ảnh nuScenes gốc qua mọi cổng
            placed, kind = _warp_source(src, dst, cache), "pasted"
            why = why if placed else "bad_fit"
        if placed is None:  # (2) ảnh tải về
            key = asset_key(b.get("category", ""))
            views = load_asset(key, cache)
            view = _asset_view(az, views) if views else None
            if view:  # biến thể cố định theo object (cùng object -> cùng ảnh ở mọi camera / khung)
                fname, rgba = views[view][zlib.crc32(str(b.get("token", i)).encode()) % len(views[view])]
            res = _place_asset(rgba, view, box, T, K) if view else None
            if res is not None:
                placed, kind, shadow = res[:2], "pasted_asset", res[2]
                src = {"asset": str(ASSET_DIR / key / fname), "view": view}
        if placed is None and cands and RELAX_LIGHTING:  # (3) ảnh nuScenes gốc khác điều kiện sáng
            src, _ = pick_source(cands, az, target_lum, target_cond, relax=True)
            if src is not None:
                placed, kind = _warp_source(src, dst, cache), "pasted_relaxed"
        if placed is None:
            status[i] = why or "no_source"
            continue
        warped, mask = placed
        occl = np.zeros((H, W), bool)
        for d, m in real_views:  # object thật gần camera hơn che object chèn
            if d < cam_depth(b):
                occl |= m > 0
        mask[occl] = 0
        sel = mask > 0
        if sel.sum() < 30:
            status[i] = "not_in_view"
            continue
        if shadow is not None and SHADOW > 0:  # bóng đổ mờ dưới chân ảnh tải về
            u, v, pw = shadow
            sh = np.zeros((H, W), np.float32)
            cv2.ellipse(sh, (int(u), int(v)), (max(2, int(0.45 * pw)), max(2, int(0.06 * pw))), 0, 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 0.05 * pw)) * ~occl
            out *= 1 - SHADOW * sh[..., None]
        ring = cv2.dilate(sel.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        ring &= ~sel
        if ring.sum() > 30:
            warped[sel] = _match_color(warped[sel], out[ring],
                                       COLOR_MATCH_RELAXED if kind == "pasted_relaxed" else COLOR_MATCH)
        alpha = mask.astype(np.float32) / 255
        if kind != "pasted_asset":  # ảnh tải về đã có alpha mềm; ảnh nuScenes: làm mềm mép bao lồi
            blur = cv2.GaussianBlur(alpha, (0, 0), FEATHER)
            alpha = blur * ((cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0) | (blur > 0.5))
        out = out * (1 - alpha[..., None]) + warped * alpha[..., None]
        status[i] = kind
        keys = ("asset", "view") if kind == "pasted_asset" else \
            ("camera", "offset", "score", "view_diff", "sharp", "occ", "relaxed")
        sources[i] = {k: src[k] for k in keys}
    return np.clip(out, 0, 255).astype(np.uint8), status, sources


if __name__ == "__main__":
    import argparse
    import os

    import config
    from task_a.run_insertion import collect_from_bank, run_insertion
    from utils.camera_proj import CAMS, render_camera
    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="sample token đích (mặc định sample[10])")
    ap.add_argument("--num-objects", type=int, default=6)
    ap.add_argument("--categories", nargs="*", default=["vehicle.car", "vehicle.truck", "human.pedestrian.adult"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--vn-scenario", help="chạy kịch bản VN (vd ba_gac_slow, cow_road) thay vì chèn từ bank")
    args = ap.parse_args()

    nusc = load_nuscenes()
    target = args.target or nusc.sample[10]["token"]
    rng = np.random.default_rng(args.seed)
    if args.vn_scenario:
        from task_a.extract_object import TemplateBank
        from task_a.vn_scenarios import VN_SCENARIOS, run_vn_scenario
        res = run_vn_scenario(nusc, target, VN_SCENARIOS[args.vn_scenario], TemplateBank().load_or_build(nusc),
                              rng, 1000)
    else:
        res = run_insertion(nusc, target, collect_from_bank(nusc, target, args.num_objects, args.categories, rng=rng),
                            rng=rng)
    boxes = [dict(a, inserted=True) for a in res["annotations_sensor"]]
    boxes += [dict(a, inserted=False) for a in res["existing_annotations"]]
    rows, counts = [], {}
    for cam in CAMS:
        before, _ = render_camera(nusc, target, cam, boxes, paste=False)
        after, _ = render_camera(nusc, target, cam, boxes, paste=True, paste_status=counts)
        rows.append(np.hstack([np.asarray(before), np.asarray(after)]))
    os.makedirs(config.VIS_DIR, exist_ok=True)
    png = os.path.join(config.VIS_DIR, f"camera_paste_{target}{'_' + args.vn_scenario if args.vn_scenario else ''}.png")
    cv2.imwrite(png, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"chèn {res['n_inserted']}/{res['n_requested']} object; trạng thái dán (theo camera thấy object): {counts}")
    print("saved", png, "(trái: chỉ box chiếu, phải: dán ảnh object)")
