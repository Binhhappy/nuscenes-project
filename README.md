# VN-LiDAR Augment (N3-08C)

Sinh dữ liệu LiDAR có sẵn nhãn 3D cho xe tự lái từ nuScenes: chèn vật thể thật vào cảnh khác và mô phỏng mưa / sương /
giông theo đúng vật lý cảm biến, tạo các tình huống giao thông Việt Nam (bầy xe máy, xe ba gác, xe hàng rong, xe chở
hàng cồng kềnh) mà dữ liệu gốc chưa có.

**Demo trực tuyến:** https://binhhappy-n3-08c-demo.static.hf.space ·
[Hugging Face Space](https://huggingface.co/spaces/Binhhappy/n3-08c-demo) (slide, demo 3D tương tác, video giới thiệu)

| Chỉ số (100 cảnh, nuScenes mini) | Kết quả |
|---|---|
| Tỉ lệ cảnh hợp lệ (không chồng vật thể, đúng mặt đường) | **100%** |
| Tốc độ sinh (laptop, 1 luồng CPU) | **~9 400 cảnh / giờ** |
| Mật độ điểm so với vật thể thật cùng khoảng cách | **1.01×** |
| Detector nhận vật thể chèn như vật thật (ô tô) | **93% = 93%** |

## Bắt đầu nhanh

Windows PowerShell, Python 3.10. Hướng dẫn đầy đủ (tải dữ liệu, chạy từng module, lỗi thường gặp):
**[HUONG_DAN.md](HUONG_DAN.md)**.

```powershell
py -3.10 -m venv nuscenes-env
.\nuscenes-env\Scripts\Activate.ps1
pip install -r n3-08c\requirements.txt
# tải nuScenes Mini + Map expansion vào datasets\nuscenes\ (HUONG_DAN.md mục 2.3)
cd n3-08c
python verify_setup.py --no-vis
pytest tests                                   # 73 test
```

## Chạy

```powershell
cd n3-08c
python task_a/run_insertion.py --vis           # chèn vật thể từ cảnh khác vào sample[10]
python task_b/run_weather.py --all-levels      # mưa / sương / giông trên sample[0]
python evaluate.py --run-insertion 100 --augmented-dir ..\output\augmented_eval   # 2 chỉ số đề bài
cd ..
python run_app.py                              # web app: http://localhost:8050 (PORT để đổi cổng)
```

Web app (FastAPI + Three.js): sinh 23 kịch bản / tự chọn vật thể, 4 kiểu thời tiết, xem 3D + 6 camera (có đồng bộ ảnh
vật thể), tải `.bin` / `.pcd` / `.ply` / nhãn JSON. Docker: `docker compose up -d --build` ([webapp/README.md](webapp/README.md)).

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `n3-08c/task_a/` | Chèn vật thể: cắt template, đặt vị trí trên bản đồ, mặt đất đồng thuận RANSAC, che khuất theo tia, ray-cast vật thể VN, kịch bản |
| `n3-08c/task_b/` | Thời tiết theo Beer–Lambert: mưa, sương, giông (v2 mặc định; `N308C_WEATHER_MODEL=v1` để tái lập số cũ) |
| `n3-08c/evaluate.py` | Kiểm tra cảnh hợp lệ độc lập, scene / giờ, chất lượng vật lý |
| `n3-08c/utils/camera_paste.py` | Dán ảnh 2D vật thể lên 6 camera |
| `n3-08c/tools/` | Đánh giá detector, biểu đồ, video, kiểm chứng mặt đường ướt, tách nền ảnh |
| `webapp/` | Web app |
| `assets/vn_objects/` | Ảnh vật thể VN (Wikimedia Commons, nguồn trong `SOURCES.md`) |
| `docs/` | Báo cáo, kịch bản demo, slide (`demo/trinh_bay.html`, `demo/trinh_bay.pdf`), demo offline (`demo/index.html`) |
| `deploy/` | Đóng gói Hugging Face Space tĩnh, xuất slide PDF |

## Tài liệu

- Báo cáo số liệu: [docs/bao_cao_ket_qua.md](docs/bao_cao_ket_qua.md) (mục 0: chỉ số theo đề bài)
- Danh mục kịch bản: [docs/danh_muc_kich_ban.md](docs/danh_muc_kich_ban.md)
- Kịch bản trình bày + demo: [docs/kich_ban_demo.md](docs/kich_ban_demo.md)
- So sánh repo nhóm: [docs/so_sanh_repo_nhom.md](docs/so_sanh_repo_nhom.md)

`output/` (dữ liệu sinh, ~32 GB) và `datasets/` không có trong repo: sinh lại bằng các script trong `n3-08c/tools/`
và tải nuScenes theo HUONG_DAN.md. Web app cần `output/templates`, `cases`, `report`, `videos`, `sequences`.

Dữ liệu nuScenes © Motional, giấy phép CC BY-NC-SA 4.0. Dự án học tập phi thương mại, không liên kết với Motional.
