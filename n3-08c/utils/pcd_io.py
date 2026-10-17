"""Ghi point cloud .pcd (binary, PCD v0.7) và .ply (binary little-endian) để mở bằng CVAT / CloudCompare / PCL /
MeshLab — không cần open3d. Giữ đủ 5 trường x, y, z, intensity, ring (không đổi intensity thành màu).
Ý tưởng xuất thêm .ply lấy từ repo nhóm (buildPhasev1.2 nhánh v3, utils/io_converter.py)."""
import numpy as np

FIELDS = ("x", "y", "z", "intensity", "ring")


def _as5(points: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(points[:, :5] if points.shape[1] >= 5 else np.c_[points[:, :4], np.zeros(len(points))],
                                dtype=np.float32)


def save_ply(points: np.ndarray, path: str) -> None:
    """points (N, >=4) float: x, y, z, intensity[, ring] -> PLY binary_little_endian (5 property float)."""
    p = _as5(points)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(p)}\n"
              + "".join(f"property float {f}\n" for f in FIELDS) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(p.astype("<f4").tobytes())


def load_ply(path: str) -> np.ndarray:
    """Đọc lại file do save_ply ghi."""
    with open(path, "rb") as f:
        data = f.read()
    head_end = data.index(b"end_header\n") + len(b"end_header\n")
    n = int([ln for ln in data[:head_end].decode().splitlines() if ln.startswith("element vertex")][0].split()[2])
    return np.frombuffer(data[head_end:], dtype="<f4").reshape(n, len(FIELDS))


def save_pcd(points: np.ndarray, path: str) -> None:
    """points (N, >=4) float: x, y, z, intensity[, ring]."""
    p = _as5(points)
    header = (f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS {' '.join(FIELDS)}\n"
              f"SIZE 4 4 4 4 4\nTYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH {len(p)}\nHEIGHT 1\n"
              f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(p)}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(p.tobytes())


def load_pcd(path: str) -> np.ndarray:
    """Đọc lại file do save_pcd ghi (binary, float32)."""
    with open(path, "rb") as f:
        data = f.read()
    head_end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    n = int([ln for ln in data[:head_end].decode().splitlines() if ln.startswith("POINTS")][0].split()[1])
    return np.frombuffer(data[head_end:], dtype=np.float32).reshape(n, -1)
