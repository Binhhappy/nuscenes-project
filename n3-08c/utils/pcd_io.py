"""Ghi point cloud .pcd (binary, PCD v0.7) để mở bằng CVAT / CloudCompare / PCL — không cần open3d."""
import numpy as np

FIELDS = ("x", "y", "z", "intensity", "ring")


def save_pcd(points: np.ndarray, path: str) -> None:
    """points (N, >=4) float: x, y, z, intensity[, ring]."""
    p = np.ascontiguousarray(points[:, :5] if points.shape[1] >= 5 else np.c_[points[:, :4], np.zeros(len(points))],
                             dtype=np.float32)
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
