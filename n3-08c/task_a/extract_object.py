"""Object extraction: crop points trong cuboid -> template ở object-local frame.

Template points: (M, 5) [x, y, z, intensity, ring], gốc = tâm cuboid,
trục x = hướng đầu object (yaw = 0), z hướng lên, đáy box ở z = -h/2.

Template bank: <TEMPLATE_DIR>/<category>/<ann_token>.npz + bank.json (metadata, không chứa points).

Chạy:
    python task_a/extract_object.py                    # demo: object nhiều điểm nhất của sample[0]
    python task_a/extract_object.py --ann <token> --show
    python task_a/extract_object.py --build            # build bank cho toàn bộ dataset
    python task_a/extract_object.py --stats            # thống kê bank đã build
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.nuscenes_loader import get_annotations, get_pointcloud, load_nuscenes  # noqa: E402
from utils.transforms import points_in_cuboid, quaternion_to_yaw, to_local_frame, wrap_angle  # noqa: E402

META_KEYS = ("ann_token", "sample_token", "scene_token", "category", "size",
             "distance", "view_angle", "num_points", "attributes")


def _matches(category: str, prefixes: Optional[Sequence[str]]) -> bool:
    return not prefixes or any(category.startswith(p) for p in prefixes)


def attribute_names(nusc, ann_token: str) -> List[str]:
    """Attribute nuScenes của annotation, vd ["cycle.with_rider"], ["vehicle.parked"]."""
    return [nusc.get("attribute", t)["name"] for t in nusc.get("sample_annotation", ann_token)["attribute_tokens"]]


def has_required_attributes(template: dict,
                            required: Optional[Dict[str, str]] = config.TEMPLATE_REQUIRED_ATTRIBUTES) -> bool:
    """required: {prefix category: attribute bắt buộc}. Category không có trong required thì luôn đạt."""
    need = [attr for prefix, attr in (required or {}).items() if template["category"].startswith(prefix)]
    return all(attr in template.get("attributes", ()) for attr in need)


def crop_template(points: np.ndarray, ann: dict, sample_token: str = "", scene_token: str = "",
                  ground_margin: float = config.TEMPLATE_GROUND_MARGIN, attributes: Sequence[str] = ()) -> dict:
    """Crop points (sensor frame) trong annotation ann (sensor frame) -> template.

    ground_margin: bỏ điểm có z_local < -h/2 + ground_margin (mặt đường lẫn trong đáy box).
    attributes: attribute nuScenes của annotation (attribute_names), lưu vào template để lọc.
    """
    mask = points_in_cuboid(points, ann["translation"], ann["size"], ann["rotation"])
    local = to_local_frame(points[mask], ann["translation"], ann["rotation"]).astype(np.float32)
    if ground_margin > 0:
        local = local[local[:, 2] >= -ann["size"][2] / 2 + ground_margin]

    cx, cy = ann["translation"][:2]
    return {
        "points": local,
        "category": ann["category"],
        "size": [float(v) for v in ann["size"]],
        "ann_token": ann["token"],
        "sample_token": sample_token,
        "scene_token": scene_token,
        "distance": float(np.hypot(cx, cy)),
        # Hướng sensor nhìn object, đo trong object frame: giữ góc này khi đặt lại
        # thì mặt object mà LiDAR thấy vẫn khớp với template.
        "view_angle": float(wrap_angle(np.arctan2(cy, cx) - quaternion_to_yaw(ann["rotation"]))),
        "num_points": int(len(local)),
        "attributes": list(attributes),
    }


def extract_object_template(nusc, sample_token: str, annotation_token: str,
                            ground_margin: float = config.TEMPLATE_GROUND_MARGIN) -> dict:
    """Crop points của 1 annotation -> template.

    Returns dict:
        points (M, 5) float32 object-local, category, size [w, l, h], ann_token, sample_token,
        scene_token, distance (m, BEV từ sensor), view_angle (rad), num_points, attributes.
    """
    ann = next((a for a in get_annotations(nusc, sample_token) if a["token"] == annotation_token), None)
    if ann is None:
        raise ValueError(f"Annotation {annotation_token} không thuộc sample {sample_token}")
    scene = nusc.get("sample", sample_token)["scene_token"]
    return crop_template(get_pointcloud(nusc, sample_token), ann, sample_token, scene, ground_margin,
                         attribute_names(nusc, annotation_token))


def extract_all_templates(nusc, sample_token: str, categories: Optional[Sequence[str]] = None,
                          min_points: int = config.MIN_TEMPLATE_POINTS,
                          ground_margin: float = config.TEMPLATE_GROUND_MARGIN) -> List[dict]:
    """Mọi template hợp lệ trong 1 sample (lọc prefix category + số điểm tối thiểu).

    Point cloud và annotations chỉ load 1 lần cho cả sample.
    """
    points = get_pointcloud(nusc, sample_token)
    scene = nusc.get("sample", sample_token)["scene_token"]
    out = []
    for ann in get_annotations(nusc, sample_token):
        if not _matches(ann["category"], categories):
            continue
        t = crop_template(points, ann, sample_token, scene, ground_margin, attribute_names(nusc, ann["token"]))
        if t["num_points"] >= min_points:
            out.append(t)
    return out


def save_template(template: dict, out_dir: str = config.TEMPLATE_DIR) -> str:
    """Lưu template thành <out_dir>/<category>/<ann_token>.npz."""
    path = os.path.join(out_dir, template["category"], f"{template['ann_token']}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, points=template["points"],
             meta=json.dumps({k: template[k] for k in META_KEYS if k in template}))
    return path


def load_template(path: str) -> dict:
    """Đọc template do save_template ghi."""
    with np.load(path) as d:
        t = json.loads(str(d["meta"]))
        t["points"] = d["points"]
    return t


class TemplateBank:
    """Kho template cho toàn dataset: metadata trong bank.json, points load lazy từ .npz."""

    INDEX_NAME = "bank.json"  # không dùng index.json để khỏi ghi đè output của extract_objects.py cũ

    def __init__(self, root: str = config.TEMPLATE_DIR):
        self.root = root
        self.meta: List[dict] = []
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def index_path(self) -> str:
        return os.path.join(self.root, self.INDEX_NAME)

    def __len__(self) -> int:
        return len(self.meta)

    def exists(self) -> bool:
        return os.path.isfile(self.index_path)

    def build(self, nusc, categories: Optional[Sequence[str]] = config.TEMPLATE_CATEGORIES,
              min_points: int = config.MIN_TEMPLATE_POINTS, sample_tokens: Optional[Iterable[str]] = None,
              ground_margin: float = config.TEMPLATE_GROUND_MARGIN, progress: bool = True) -> "TemplateBank":
        """Trích template từ sample_tokens (mặc định toàn bộ dataset), lưu .npz + bank.json."""
        from tqdm import tqdm

        tokens = list(sample_tokens) if sample_tokens is not None else [s["token"] for s in nusc.sample]
        self.meta, self._cache = [], {}
        for tok in tqdm(tokens, desc="Building template bank", disable=not progress):
            for t in extract_all_templates(nusc, tok, categories, min_points, ground_margin):
                path = save_template(t, self.root)
                # "/" để bank build trên Windows đọc được trên Linux (WSL, T6) và ngược lại
                rel = os.path.relpath(path, self.root).replace(os.sep, "/")
                self.meta.append({k: t[k] for k in META_KEYS} | {"file": rel})
        os.makedirs(self.root, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)
        return self

    def load(self) -> "TemplateBank":
        with open(self.index_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self._cache = {}
        return self

    def load_or_build(self, nusc, **build_kwargs) -> "TemplateBank":
        return self.ensure_attributes(nusc) if self.exists() else self.build(nusc, **build_kwargs)

    def ensure_attributes(self, nusc) -> "TemplateBank":
        """Bank build trước khi có trường attributes: bổ sung từ metadata nuScenes (không crop lại),
        ghi lại bank.json. Bank đã đủ thì không làm gì."""
        if not self.meta and self.exists():
            self.load()
        missing = [m for m in self.meta if "attributes" not in m]
        for m in missing:
            m["attributes"] = attribute_names(nusc, m["ann_token"])
        if missing:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, indent=1)
        return self

    def filter(self, categories: Optional[Sequence[str]] = None, min_points: int = 0,
               exclude_samples: Iterable[str] = (), exclude_scenes: Iterable[str] = (),
               distance_range: Optional[Tuple[float, float]] = None,
               required_attributes: Optional[Dict[str, str]] = None) -> List[dict]:
        """Metadata thoả điều kiện. exclude_scenes: tránh chèn lại chính instance đã có trong scene đích.

        required_attributes: {prefix category: attribute}, vd config.TEMPLATE_REQUIRED_ATTRIBUTES để chỉ lấy
        xe 2 bánh có người lái (nuScenes: phần lớn box xe 2 bánh là xe đỗ). Cần bank có attributes.
        """
        ex_samples, ex_scenes = set(exclude_samples), set(exclude_scenes)
        out = []
        for m in self.meta:
            if not _matches(m["category"], categories) or m["num_points"] < min_points:
                continue
            if m["sample_token"] in ex_samples or m["scene_token"] in ex_scenes:
                continue
            if distance_range and not distance_range[0] <= m["distance"] <= distance_range[1]:
                continue
            if required_attributes and not has_required_attributes(m, required_attributes):
                continue
            out.append(m)
        return out

    def get(self, meta: dict) -> dict:
        """Template đầy đủ (có points) từ 1 entry metadata."""
        key = meta["ann_token"]
        if key not in self._cache:
            rel = meta["file"].replace("\\", "/")  # bank cũ build trên Windows lưu "\\"
            self._cache[key] = load_template(os.path.join(self.root, *rel.split("/")))["points"]
        return dict(meta, points=self._cache[key])

    def sample(self, n: int, rng: Optional[np.random.Generator] = None, balance: bool = True,
               **filter_kwargs) -> List[dict]:
        """Chọn n template (có lặp lại). balance=True: chọn category đều trước rồi mới chọn template,
        tránh vehicle.car (chiếm đa số) lấn át các class hiếm."""
        rng = rng or np.random.default_rng()
        pool = self.filter(**filter_kwargs)
        if not pool or n <= 0:
            return []
        if balance:
            by_cat: Dict[str, List[dict]] = {}
            for m in pool:
                by_cat.setdefault(m["category"], []).append(m)
            cats = sorted(by_cat)
            picked = [by_cat[c][rng.integers(len(by_cat[c]))] for c in rng.choice(cats, size=n)]
        else:
            picked = [pool[i] for i in rng.integers(len(pool), size=n)]
        return [self.get(m) for m in picked]

    def stats(self) -> Dict[str, dict]:
        """Số template + median số điểm theo category."""
        by_cat: Dict[str, List[int]] = {}
        for m in self.meta:
            by_cat.setdefault(m["category"], []).append(m["num_points"])
        return {c: {"count": len(v), "median_points": int(np.median(v))}
                for c, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))}


def _print_stats(bank: TemplateBank) -> None:
    print(f"{len(bank)} templates trong {bank.root}")
    for cat, s in bank.stats().items():
        print(f"  {cat:40s} {s['count']:5d}  median {s['median_points']} pts")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", help="annotation token (mặc định: object nhiều điểm nhất trong sample[0])")
    ap.add_argument("--show", action="store_true", help="xem template bằng Open3D")
    ap.add_argument("--build", action="store_true", help="build template bank cho toàn bộ dataset")
    ap.add_argument("--stats", action="store_true", help="thống kê bank đã build")
    ap.add_argument("--categories", nargs="*", default=list(config.TEMPLATE_CATEGORIES))
    ap.add_argument("--min-points", type=int, default=config.MIN_TEMPLATE_POINTS)
    args = ap.parse_args()

    if args.stats:
        _print_stats(TemplateBank().load())
        sys.exit(0)

    nusc = load_nuscenes()
    if args.build:
        _print_stats(TemplateBank().build(nusc, args.categories, args.min_points))
        sys.exit(0)

    if args.ann:
        sample_token = nusc.get("sample_annotation", args.ann)["sample_token"]
        t = extract_object_template(nusc, sample_token, args.ann)
    else:
        sample_token = nusc.sample[0]["token"]
        t = max(extract_all_templates(nusc, sample_token), key=lambda x: x["num_points"])

    print(f"{t['category']}: {t['num_points']} pts, size={np.round(t['size'], 2)}, "
          f"dist={t['distance']:.1f} m, view_angle={np.degrees(t['view_angle']):.0f}°")
    print("local bbox min/max:", t["points"][:, :3].min(0).round(2), t["points"][:, :3].max(0).round(2))
    print("saved:", save_template(t))

    if args.show:
        from utils.visualization import show_open3d
        box = {"translation": [0, 0, 0], "size": t["size"], "rotation": [1, 0, 0, 0]}
        show_open3d(t["points"], new_boxes=[box], window_name=t["category"])
