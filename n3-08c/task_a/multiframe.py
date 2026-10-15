"""T3: kịch bản VN nhiều khung — object chèn có quỹ đạo liên tục qua các keyframe liên tiếp của 1 scene.

Mỗi object là 1 track trong EGO frame (x hướng đầu xe ego, y sang trái) theo thời gian t (s) từ khung đầu:
    swarm   bầy xe 2 bánh chạy cùng ego: x trôi chậm tương đối (vượt / tụt lại), y lượn nhẹ quanh làn
    cut_in  xe 2 bánh từ làn bên chéo vào trước đầu ego trong ~CUT_IN_TIME s rồi chạy thẳng trong làn ego
Track theo ego frame nên object bám đường như ego (ego rẽ thì bầy rẽ theo). Hướng object = hướng vận tốc global
(vận tốc ego + vận tốc tương đối), vận tốc này cũng dùng để chèn vào sweep (T4).

Mỗi khung: vật lý như run_scenario (template chọn lại theo góc nhìn, ưu tiên cùng instance để hình dạng nhất quán;
check_position; insert_object: mật độ, ring, che khuất). Vị trí không hợp lệ ở khung nào thì object vắng mặt
ở khung đó (ghi vào "skipped").

Chạy:
    python task_a/multiframe.py --scene scene-0103 --scenario swarm
Output: <out_dir>/<sample_token>.bin/.json (+ sweeps/) cho từng keyframe, cùng format save_result.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.extract_object import TemplateBank  # noqa: E402
from task_a.ground_plane import GroundModel, load_map_for_sample  # noqa: E402
from task_a.merge import create_annotation, insert_object, points_in_new_box  # noqa: E402
from task_a.placement import check_position  # noqa: E402
from task_a.raycast import build_model, insert_raycast  # noqa: E402
from task_a.run_insertion import save_result  # noqa: E402
from task_a.scenarios import SCENARIOS, EgoFrame, TemplateMatcher, scenario_area  # noqa: E402
from task_a.sweeps import insert_into_sweeps  # noqa: E402
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose, get_lidar_token,  # noqa: E402
                                   get_pointcloud)
from utils.transforms import box_sensor_to_global, quaternion_to_yaw, wrap_angle, yaw_to_quaternion  # noqa: E402

CUT_IN_TIME = 2.0  # (s) thời gian chéo vào làn ego


def scene_samples(nusc, scene_name: str) -> List[str]:
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    toks, tok = [], scene["first_sample_token"]
    while tok:
        toks.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return toks


def ego_state(nusc, tokens: Sequence[str]) -> List[dict]:
    """Mỗi keyframe: t (s từ khung đầu), vận tốc ego global (vx, vy), yaw ego global."""
    poses = [get_ego_pose(nusc, get_lidar_token(nusc, t)) for t in tokens]
    ts = np.array([p["timestamp"] for p in poses]) / 1e6
    xy = np.array([p["translation"][:2] for p in poses])
    v = np.gradient(xy, ts, axis=0) if len(tokens) > 1 else np.zeros((1, 2))
    return [{"t": float(ts[i] - ts[0]), "v": v[i], "yaw": quaternion_to_yaw(p["rotation"]), "xy": xy[i]}
            for i, p in enumerate(poses)]


def make_tracks(kind: str, n: int, rng: np.random.Generator) -> List[dict]:
    """Tham số quỹ đạo theo ego frame."""
    tracks = []
    for _ in range(n):
        if kind == "swarm":
            while True:
                x0, y0 = rng.uniform(-10, 15), rng.uniform(-6, 6)
                if np.hypot(x0 - 1.3, y0) <= 15:
                    break
            tracks.append({"kind": kind, "x0": x0, "y0": y0, "vx": rng.normal(0, 0.4),
                           "amp": rng.uniform(0.05, 0.25), "w": rng.uniform(0.3, 0.8), "ph": rng.uniform(0, 2 * np.pi)})
        elif kind == "cut_in":
            side = 1 if rng.random() < 0.5 else -1
            tracks.append({"kind": kind, "x0": rng.uniform(8, 12), "y0": side * rng.uniform(2.5, 3.5),
                           "x1": rng.uniform(4, 6), "t0": rng.uniform(0.5, 2.0)})
        elif kind == "cargo":  # xe máy chở hàng chạy cùng dòng phía trước, trôi chậm tương đối
            tracks.append({"kind": kind, "x0": rng.uniform(6, 15), "y0": rng.uniform(-2.5, 2.5),
                           "vx": rng.normal(0, 0.3), "amp": rng.uniform(0.05, 0.15), "w": rng.uniform(0.3, 0.8),
                           "ph": rng.uniform(0, 2 * np.pi)})
        elif kind == "dog":  # chó đứng mép đường rồi lao ngang qua trước xe
            side = 1 if rng.random() < 0.5 else -1
            tracks.append({"kind": kind, "x0": rng.uniform(14, 22), "y0": side * rng.uniform(3.0, 4.5), "side": side,
                           "t0": rng.uniform(0.5, 1.5), "v": rng.uniform(3, 5)})
        else:
            raise ValueError(kind)
    return tracks


RC_KINDS = {"cargo": "moto_cargo", "dog": "dog"}  # track dùng vật thể ray-cast (task_a/raycast.py)


def track_pose(tr: dict, t: float):
    """(x, y) và vận tốc tương đối (dx/dt, dy/dt) trong ego frame tại thời điểm t."""
    if tr["kind"] == "dog":
        dt = max(0.0, t - tr["t0"])
        return tr["x0"], tr["y0"] - tr["side"] * tr["v"] * dt, 0.0, (-tr["side"] * tr["v"] if dt > 0 else 0.0)
    if tr["kind"] in ("swarm", "cargo"):
        x = tr["x0"] + tr["vx"] * t
        y = tr["y0"] + tr["amp"] * np.sin(tr["w"] * t + tr["ph"])
        return x, y, tr["vx"], tr["amp"] * tr["w"] * np.cos(tr["w"] * t + tr["ph"])
    s = np.clip((t - tr["t0"]) / CUT_IN_TIME, 0.0, 1.0)  # cut_in: đứng ở làn bên, rồi chéo vào, rồi trong làn
    moving = 0.0 < (t - tr["t0"]) < CUT_IN_TIME
    x = tr["x0"] + (tr["x1"] - tr["x0"]) * s
    y = tr["y0"] * (1 - s)
    return x, y, ((tr["x1"] - tr["x0"]) / CUT_IN_TIME if moving else 0.0), (-tr["y0"] / CUT_IN_TIME if moving else 0.0)


def run_multiframe(nusc, tokens: Sequence[str], kind: str, bank: TemplateBank, out_dir: str, n_objects: int,
                   rng: Optional[np.random.Generator] = None, allow_template=None, sweeps: bool = True) -> dict:
    rng = rng or np.random.default_rng(config.SEED)
    sc = SCENARIOS["moto_swarm"] if kind == "swarm" else SCENARIOS["cut_in"]  # cùng vùng đặt: lòng đường
    states = ego_state(nusc, tokens)
    scene_token = nusc.get("sample", tokens[0])["scene_token"]
    rc = kind in RC_KINDS
    matcher = None if rc else TemplateMatcher(bank, sc.categories, sc.min_points, exclude_scenes=[scene_token],
                                              allow=allow_template)
    nusc_map = load_map_for_sample(nusc, tokens[0])
    tracks = _initial_tracks(nusc, tokens[0], kind, n_objects, sc, nusc_map, rng)
    prev_tpl: Dict[int, dict] = {}
    summary = []
    for tok, st in zip(tokens, states):
        t0 = time.perf_counter()
        points = get_pointcloud(nusc, tok)
        existing = get_annotations(nusc, tok)
        sd = get_lidar_token(nusc, tok)
        calib, pose = get_calibration(nusc, sd), get_ego_pose(nusc, sd)
        ego = EgoFrame(calib)
        ground = GroundModel.from_points(points, rng=rng)
        area = scenario_area(nusc, nusc_map, tok, sc)
        c, s_ = np.cos(st["yaw"]), np.sin(st["yaw"])
        new_anns, used, skipped, tids = [], {}, [], []
        for i, tr in enumerate(tracks):
            if tr["kind"] == "dog":  # chó đứng yên trên mặt đất (hệ global) rồi lao ngang, không trôi theo ego
                ex, ey, v_glob = _dog_pose(tr, states[0], st)
            else:
                ex, ey, dvx, dvy = track_pose(tr, st["t"])
                v_glob = st["v"] + np.array([c * dvx - s_ * dvy, s_ * dvx + c * dvy])
            speed = float(np.linalg.norm(v_glob))
            # hướng object = hướng vận tốc global; xe gần như đứng yên thì theo hướng ego
            yaw_glob = float(np.arctan2(v_glob[1], v_glob[0])) if speed > 0.5 else st["yaw"]
            x, y, yaw_req = ego.to_sensor(ex, ey, float(wrap_angle(yaw_glob - st["yaw"])))
            if rc:  # ray-cast: cùng 1 hình suốt track, bắn tia lại mỗi khung nên mặt nhìn thấy tự đúng
                tpl = tr.setdefault("model", build_model(RC_KINDS[kind], rng))
                yaw, insert = yaw_req, insert_raycast
            else:
                tpl = _pick_template(matcher, prev_tpl.get(i), x, y, yaw_req, nusc, rng)
                yaw, insert = float(wrap_angle(np.arctan2(y, x) - tpl["view_angle"])), insert_object
                if abs(wrap_angle(yaw - yaw_req)) > np.radians(30):  # template giữ lại lệch góc nhìn quá xa
                    yaw = yaw_req
            reason = check_position(area, existing + new_anns + [ego.ego_box()], tpl["size"], x, y, yaw, ground, points)
            if reason:
                skipped.append((i, reason))
                continue
            aug, info = insert(points, tpl, (x, y, yaw), ground, rng)
            if info["n_visible"] < 1:
                skipped.append((i, "occluded"))
                continue
            points = aug
            z, _ = ground.height_at(x, y)
            ann = create_annotation(tpl, (x, y, yaw), float(z[0]), info["n_visible"])
            ann["track_id"] = i
            dyaw = -(st["yaw"] + ego.yaw)  # global -> sensor: bỏ hướng ego (global) và hướng sensor (trên ego)
            ann["velocity_sensor"] = [float(v_glob[0] * np.cos(dyaw) - v_glob[1] * np.sin(dyaw)),
                                      float(v_glob[0] * np.sin(dyaw) + v_glob[1] * np.cos(dyaw))]
            new_anns.append(ann)
            used[tpl["ann_token"]] = tpl
            prev_tpl[i] = tpl
            tids.append(i)
        for ann in new_anns:
            ann["num_lidar_pts"] = int(points_in_new_box(points, ann).sum())
        anns_global = []
        for ann in new_anns:
            g = box_sensor_to_global(ann, calib, pose)
            dyaw = quaternion_to_yaw(g["rotation"]) - quaternion_to_yaw(ann["rotation"])
            vx, vy = ann["velocity_sensor"]
            g["velocity"] = [vx * np.cos(dyaw) - vy * np.sin(dyaw), vx * np.sin(dyaw) + vy * np.cos(dyaw)]
            anns_global.append(g)
        result = {"points": points, "target_sample_token": tok, "physical": True, "scenario": f"multiframe_{kind}",
                  "t": st["t"], "n_requested": len(tracks), "n_inserted": len(new_anns),
                  "elapsed_s": time.perf_counter() - t0, "skipped": skipped,
                  "annotations_sensor": new_anns, "annotations_global": anns_global}
        if sweeps and new_anns:
            result["sweeps"] = insert_into_sweeps(nusc, tok, anns_global, used, out_dir, rng=rng)
        save_result(result, out_dir)
        summary.append((len(new_anns), len(skipped)))
    return {"tokens": list(tokens), "per_frame": summary, "out_dir": out_dir}


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _dog_pose(tr: dict, st0: dict, st: dict):
    """Vị trí ego frame + vận tốc global của chó: chỗ đứng cố định trên mặt đất (lấy theo ego frame khung đầu),
    từ t0 chạy thẳng vuông góc hướng xe lúc đầu, tốc độ v."""
    p0 = st0["xy"] + _rot(st0["yaw"]) @ np.array([tr["x0"], tr["y0"]])
    direction = _rot(st0["yaw"]) @ np.array([0.0, -tr["side"]])
    moving = st["t"] > tr["t0"]
    p = p0 + direction * tr["v"] * max(0.0, st["t"] - tr["t0"])
    ex, ey = _rot(-st["yaw"]) @ (p - st["xy"])
    return float(ex), float(ey), direction * tr["v"] if moving else np.zeros(2)


def _initial_tracks(nusc, tok: str, kind: str, n: int, sc, nusc_map, rng, n_attempts: int = 300) -> List[dict]:
    """Chỉ giữ track có vị trí hợp lệ ở khung đầu, cả ở t = 0 và t = 3 s (trong vùng map, không đè object thật /
    ego / track đã chọn), để object ít bị vắng ở các khung sau."""
    points = get_pointcloud(nusc, tok)
    ego = EgoFrame(get_calibration(nusc, get_lidar_token(nusc, tok)))
    ground = GroundModel.from_points(points, rng=rng)
    area = scenario_area(nusc, nusc_map, tok, sc)
    blockers = get_annotations(nusc, tok) + [ego.ego_box()]
    size = [0.8, 2.1, 1.5]  # xe máy (w, l, h) điển hình để kiểm tra chỗ trống
    tracks = []
    for _ in range(n_attempts):
        if len(tracks) >= n:
            break
        tr = make_tracks(kind, 1, rng)[0]
        boxes = []
        for t in ((0.0,) if kind == "dog" else (0.0, 3.0)):  # chó băng ngang: chỉ cần chỗ đứng ban đầu
            ex, ey, _, _ = track_pose(tr, t)
            x, y, yaw = ego.to_sensor(ex, ey, 0.0)
            if check_position(area, blockers, size, x, y, yaw, ground, points):
                break
            boxes.append({"translation": [x, y, 0.0], "size": size, "rotation": yaw_to_quaternion(yaw)})
        else:
            tracks.append(tr)
            blockers += boxes
    return tracks


def _pick_template(matcher: TemplateMatcher, prev: Optional[dict], x, y, yaw, nusc, rng) -> dict:
    """Template hợp góc nhìn, ưu tiên cùng instance với khung trước; không có thì giữ template cũ / lấy bất kỳ."""
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    ok = ((matcher.dist >= r / config.MAX_DISTANCE_SCALE) & (matcher.dist <= r / 0.95)
          & (np.abs(wrap_angle(matcher.view - (phi - yaw))) < np.radians(15)))
    idx = np.flatnonzero(ok)
    if prev is not None:
        inst = nusc.get("sample_annotation", prev["ann_token"])["instance_token"]
        same = [i for i in idx if nusc.get("sample_annotation", matcher.meta[i]["ann_token"])["instance_token"] == inst]
        if same:
            return matcher.bank.get(matcher.meta[rng.choice(same)])
        return prev
    if len(idx):
        return matcher.bank.get(matcher.meta[rng.choice(idx)])
    # khung đầu chưa có template hợp: lấy template có góc nhìn gần nhất
    j = int(np.argmin(np.abs(wrap_angle(matcher.view - (phi - yaw)))))
    return matcher.bank.get(matcher.meta[j])


if __name__ == "__main__":
    import argparse

    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--scenario", choices=["swarm", "cut_in", "cargo", "dog"], default="swarm")
    ap.add_argument("--n-objects", type=int, default=None, help="mặc định swarm 15, cut_in 1")
    ap.add_argument("--frames", type=int, default=None, help="chỉ lấy N keyframe đầu")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    nusc = load_nuscenes()
    toks = scene_samples(nusc, args.scene)[:args.frames]
    n = args.n_objects or {"swarm": 15, "cargo": 2}.get(args.scenario, 1)
    out = args.out_dir or os.path.join(config.OUTPUT_DIR, "multiframe", f"{args.scene}_{args.scenario}")
    res = run_multiframe(nusc, toks, args.scenario, TemplateBank().load_or_build(nusc), out, n,
                         np.random.default_rng(args.seed))
    print(f"{args.scene} {args.scenario}: {len(toks)} khung, object mỗi khung {[p[0] for p in res['per_frame']]} -> {out}")
