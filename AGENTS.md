# AGENTS.md

Hướng dẫn chung cho mọi agent (Claude Code, Codex, Gemini CLI / Antigravity). Đọc hết trước khi sửa repo.
Chi tiết cài đặt, chạy: `HUONG_DAN.md`. Tổng quan, số liệu: `README.md`.

## Dự án

VN-LiDAR Augment (N3-08C): sinh dữ liệu LiDAR có nhãn 3D từ nuScenes v1.0-mini.
- Task A (`n3-08c/task_a/`): chèn vật thể (template thật hoặc ray-cast vật thể VN) vào cảnh khác, kịch bản VN.
- Task B (`n3-08c/task_b/`): mưa / sương / giông theo Beer–Lambert (v2 mặc định, `N308C_WEATHER_MODEL=v1` cho số cũ).
- `n3-08c/utils/camera_paste.py`: dán ảnh 2D vật thể (`assets/vn_objects/`) lên 6 camera. Chỉ để minh hoạ,
  không ảnh hưởng điểm LiDAR hay số đo.
- `webapp/`: FastAPI (`backend/main.py`, `backend/engine.py`) + Three.js (`frontend/`). Chạy: `python run_app.py`.

## Môi trường

- Windows, PowerShell, **Python 3.10** (venv `nuscenes-env\`). Không dùng 3.12+.
- Thư viện cố định phiên bản trong `n3-08c/requirements.txt`. Không nâng phiên bản, không thêm thư viện
  nếu task không yêu cầu.
- Đường dẫn lấy từ `n3-08c/config.py` (`PROJECT_ROOT`, `DATAROOT`, `OUTPUT_DIR`). Không hard-code đường dẫn.
- Biến môi trường: `NUSCENES_DATAROOT`, `N308C_OUTPUT_DIR`, `N308C_WEATHER_MODEL`.

## Lệnh

```powershell
.\nuscenes-env\Scripts\Activate.ps1
cd n3-08c
python verify_setup.py --no-vis
pytest tests -q                       # toàn bộ test
pytest tests/test_camera_paste.py -q  # một file
```

Test dùng fixture `nusc` (`tests/conftest.py`) đọc dữ liệu nuScenes thật trong `datasets/nuscenes/`.
Test mới nên dùng mảng numpy giả khi có thể, để chạy được khi không có dataset.

## Không được đụng

- `datasets/`, `nuscenes-env/`, `output/`, `webapp/temp_outputs/`, `deploy/hf_static/`: dữ liệu / sinh tự động,
  không có trong git. Không xoá, không commit.
- `n3-08c/config.py`: ngưỡng, seed, tham số vật lý. Đổi giá trị làm lệch số liệu trong `docs/bao_cao_ket_qua.md`.
  Chỉ sửa khi task nói rõ.
- Seed và kết quả tái lập: không đổi thứ tự sinh số ngẫu nhiên trong code đã có số liệu báo cáo.

## Quy ước code

- Comment, docstring, tài liệu viết **tiếng Việt**, ngắn, theo văn phong file xung quanh.
- Tên biến / hàm tiếng Anh, snake_case.
- Toạ độ: theo `utils/transforms.py` (sensor <-> ego <-> global <-> object-local). Không tự viết lại phép biến đổi.
- Sửa ít file nhất có thể. Không refactor ngoài phạm vi task.
- Có thay đổi hành vi thì thêm hoặc sửa test trong `n3-08c/tests/`.

## Ảnh vật thể (`assets/vn_objects/`)

Quy ước đầy đủ: `assets/vn_objects/README.md`. Tóm tắt:
- Đích: `assets/vn_objects/<loại>/<mặt>.png`, `<mặt>` ∈ `front`, `oblique_front`, `side`, `oblique_rear`, `rear`.
  PNG có kênh alpha, đầu vật thể quay sang **phải**.
- Cảnh sát: `police_officer/<hiệu lệnh>/<mặt>.png`.
- Ảnh gốc đặt trong `assets/vn_objects/_raw/`, tách nền bằng `python n3-08c/tools/make_cutouts.py`
  (thêm khai báo vào `CUTS` hoặc `POLICE` trong file đó). Ảnh kiểm tra: `output/vis/cutouts.png`.
- Mỗi ảnh phải có một dòng trong `assets/vn_objects/SOURCES.md` (link gốc, tác giả, giấy phép).
  Ảnh AI tạo ghi rõ "Ảnh minh hoạ do AI tạo (<model>)".
- Chỉ dùng ảnh có giấy phép cho phép dùng lại hoặc ảnh AI tạo.

### Yêu cầu khi tạo ảnh bằng AI

- Ảnh chụp thật (photorealistic), ban ngày, bối cảnh đường phố Việt Nam.
- Nền trắng trơn hoặc trong suốt, không đổ bóng xuống đất, không có vật thể khác.
- Toàn thân, không bị cắt mép. Camera cao ~1.5 m, ngang tầm, không méo phối cảnh.
- Cạnh dài ~1024 px. Lưu: `assets/vn_objects/_raw/<loại>/<tư thế>_<mặt>.png`.

## Phân vai agent

| Agent | Phạm vi |
|---|---|
| Claude Code | Lập plan, code lõi (`task_a/`, `task_b/`, `utils/`, `evaluate.py`), review diff của agent khác |
| Codex | Task code có phạm vi rõ: test, script `n3-08c/tools/`, `webapp/` |
| Gemini / Antigravity | Tạo ảnh asset, soát ảnh camera sau khi chèn (`output/vis/`, `output/cases/`). Không sửa `.py` |

- Mỗi task chạy trên branch riêng (`agent/<tên-agent>/<task>`), không làm trực tiếp trên `main`.
- Không sửa file ngoài phạm vi được giao. Thấy bug ngoài phạm vi: ghi vào báo cáo, không tự sửa.

## Báo cáo khi xong task

1. Danh sách file đã sửa / tạo.
2. Lệnh test đã chạy và kết quả (pass / fail, kèm output lỗi nếu có).
3. Việc chưa làm, giả định đã đặt, bug phát hiện ngoài phạm vi.
