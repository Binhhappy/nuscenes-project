# N3-08C — Synthetic Point Cloud Augmentation (nuScenes)

Sinh dữ liệu LiDAR tổng hợp từ nuScenes v1.0-mini: chèn object (Task A) và mô phỏng mưa / sương mù (Task B).

**Hướng dẫn đầy đủ: [HUONG_DAN.md](HUONG_DAN.md)** — cài đặt, tải dữ liệu, cách chạy, kết quả mong đợi, lỗi thường gặp.

Bắt đầu nhanh (Windows PowerShell, Python 3.10):

```powershell
py -3.10 -m venv nuscenes-env
.\nuscenes-env\Scripts\Activate.ps1
pip install -r n3-08c\requirements.txt
# tải nuScenes Mini + Map expansion vào datasets\nuscenes\ (xem HUONG_DAN.md mục 2.3)
cd n3-08c
python verify_setup.py --no-vis
```
