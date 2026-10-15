# Web demo N3-08C

Giao diện + kiến trúc (FastAPI + Three.js + 6 camera + Docker) lấy ý tưởng từ repo nhóm
`tranducquan2k5-afk/buildPhasev1.2`, nối với pipeline của repo này. Mọi số liệu hiển thị đọc từ `output/` (đo thật).

| Tab | Nội dung |
|---|---|
| Sinh kịch bản | chọn cảnh nuScenes + 1 trong 23 kịch bản có sẵn **hoặc tự chọn vật thể** (15 loại: ô tô / xe tải / xe buýt / xe máy / xe đạp / người cắt mẫu thật; xe máy chở 2-3 người / chở hàng, ba gác, xe kéo, xe hàng rong, chó, bò, xe bò, ô tô ray-cast — mỗi loại số lượng + kiểu bố trí riêng, tối đa 150 object; mẫu nhanh: giờ cao điểm, chợ, tan trường, đường quê; bảng "đặt được / yêu cầu" theo từng loại) + thời tiết (mưa / sương / giông + vũng nước) -> xem 3D (màu Z / cường độ / làm nổi object chèn, gốc / sau chèn, 3D / BEV / góc lái xe) + 6 camera (có thời tiết) -> tải `.pcd` / `.bin` / `labels.json` / `.zip`; nút "Chạy detector" (chỉ khi có GPU + OpenPCDet) |
| Trường hợp đã đo | **5 chuỗi khung có video** (bầy xe máy, cắt đầu, xe chở hàng, chó, bầy xe trong giông — 40 / 12 khung, `tools/sequence_export.py`): Play/Pause, từng khung, tốc độ 1/2/4 khung/s, mỗi khung = 3D + box màu theo detector chạy thật trên khung đó + 6 camera đúng thời điểm + số đúng / nhầm / sót, nút xem video MP4; + 20 kịch bản + 4 điều kiện thời tiết (1 khung) + 6 camera |
| Kết quả đo | số liệu chính, bảng recall có CI 95%, biểu đồ, video |

## Chạy

Chuẩn bị (một lần): đã có `datasets/nuscenes` (v1.0-mini) và `output/` (template bank + kết quả đo, sinh bởi các
script trong `n3-08c/tools/`).

**Windows (không detector trực tiếp):**
```
nuscenes-env\Scripts\pip install -r webapp\requirements.txt
nuscenes-env\Scripts\python run_app.py          # mở http://localhost:8050
```

**WSL env pcdet (có GPU, bật nút "Chạy detector"):**
```
~/miniconda3/envs/pcdet/bin/pip install fastapi uvicorn python-multipart
cd /mnt/c/nuscenes-project && ~/miniconda3/envs/pcdet/bin/python run_app.py   # mở http://localhost:8050 trên Windows
```
Muốn chạy detector: tick "Sinh đủ 10 sweep" trước khi bấm "Sinh kịch bản".

**Docker (CPU, không detector trực tiếp)** — đã build + chạy thử với Docker Engine 29.1.3 trong WSL Ubuntu 22.04
(image 1.27 GB, build lần đầu ~5-10 phút tuỳ mạng). Chạy trong terminal Ubuntu:
```
sudo service docker start            # mỗi lần bật lại máy (WSL không tự chạy docker)
cd /mnt/c/nuscenes-project
docker compose up -d --build         # mở http://localhost:8050 trên trình duyệt Windows
docker compose logs -f               # xem log
docker compose down                  # dừng
```
Cài Docker Engine trong WSL: `sudo apt install docker.io docker-compose-v2 && sudo usermod -aG docker $USER`
(rồi `wsl --shutdown` để nhận quyền nhóm). Docker Desktop / Podman Desktop cũng chạy được cùng Dockerfile.

Địa chỉ mở thẳng: `/#auto` (sinh ngay với mặc định), `/#cases`, `/#results`; slide: `/slides`; API: `/docs`.

## Lưu ý trình bày
- Ảnh camera là ảnh thật của nuScenes: object chèn KHÔNG có trong ảnh (chỉ chèn vào LiDAR), chỉ hiện qua box + điểm
  LiDAR chiếu lên; thời tiết trên ảnh là minh hoạ (`n3-08c/task_b/camera_weather.py`), số đo là trên LiDAR.
- Server mặc định chỉ nghe `127.0.0.1`; Docker đặt `HOST=0.0.0.0` để truy cập qua cổng map.
