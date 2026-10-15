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

## Kết quả và demo (T2-T7, kịch bản giao thông Việt Nam)

- Báo cáo số liệu: [docs/bao_cao_ket_qua.md](docs/bao_cao_ket_qua.md) · danh mục kịch bản: [docs/danh_muc_kich_ban.md](docs/danh_muc_kich_ban.md) · so sánh repo nhóm, quy mô và chi phí: [docs/so_sanh_repo_nhom.md](docs/so_sanh_repo_nhom.md)
- Demo offline (mở bằng trình duyệt, không cần cài): `docs/demo/index.html`, slide `docs/demo/slides.html`
- Web app (FastAPI + Three.js + 6 camera, Docker): [webapp/README.md](webapp/README.md) — `python run_app.py` hoặc `docker compose up -d`

`output/` (~36 GB dữ liệu sinh) và `datasets/` không có trong repo: sinh lại bằng các script trong `n3-08c/tools/`
(thứ tự trong docs/bao_cao_ket_qua.md) và tải nuScenes theo HUONG_DAN.md. Web app cần `output/templates`, `cases`,
`report`, `videos`, `sequences`.
