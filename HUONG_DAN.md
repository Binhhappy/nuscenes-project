# Hướng dẫn dự án N3-08C — Synthetic Point Cloud Augmentation (nuScenes)

Dự án sinh dữ liệu LiDAR tổng hợp từ nuScenes v1.0-mini:

- **Task A — Object insertion:** cắt point cloud của vật thể thật (template), đặt vào scene khác ở vị trí hợp lệ (trên đường, không va chạm, chạm mặt đất), sinh annotation mới.
- **Task B — Weather simulation:** mô phỏng mưa và sương mù trên point cloud (dropout, noise, suy giảm intensity).
- **Evaluate:** đo tỉ lệ chèn thành công, throughput, thay đổi số điểm và intensity.

---

## 1. Cấu trúc thư mục

```
C:\nuscenes-project\
├── datasets\nuscenes\          # dữ liệu nuScenes (v1.0-mini, samples, sweeps, maps)
├── nuscenes-env\               # virtualenv Python 3.10
├── output\                     # mọi kết quả sinh ra
│   ├── templates\              # template bank (<category>\<ann_token>.npz + bank.json)
│   ├── augmented\              # Task A: <sample_token>.bin + .json
│   ├── weather\                # Task B: original\, rain_<level>\, fog_<level>\
│   ├── vis\                    # ảnh BEV (.png)
│   └── eval_report.json        # kết quả evaluate.py
├── extract_objects.py          # script cũ (bước 2, ghi index.json + .npy) — đã thay bằng task_a/extract_object.py
├── verify_setup.py, verify_map.py   # script kiểm tra cũ
└── n3-08c\                     # CODE CHÍNH
    ├── config.py               # đường dẫn, seed, ngưỡng, tham số ground/weather
    ├── verify_setup.py         # kiểm tra dữ liệu + môi trường
    ├── evaluate.py             # đánh giá Task A + Task B
    ├── requirements.txt
    ├── utils\
    │   ├── nuscenes_loader.py  # load nuScenes, point cloud, annotations, pose, calib
    │   ├── transforms.py       # sensor <-> ego <-> global <-> object-local
    │   └── visualization.py    # Open3D 3D + matplotlib BEV
    ├── task_a\
    │   ├── extract_object.py   # crop template, TemplateBank
    │   ├── ground_plane.py     # RANSAC + GroundModel, vùng đặt từ map expansion
    │   ├── placement.py        # sample (x, y, yaw) hợp lệ + collision BEV
    │   ├── merge.py            # đặt template vào scene, tạo annotation
    │   └── run_insertion.py    # pipeline Task A end-to-end
    ├── task_b\
    │   ├── rain.py             # mưa: dropout + noise gần sensor + suy giảm intensity
    │   ├── fog.py              # sương mù: exp-decay dropout + backscatter
    │   └── run_weather.py      # pipeline Task B
    └── tests\                  # test trên data thật (conftest.py tạo fixture nusc)
```

---

## 2. Cài đặt

### 2.1. Dữ liệu

Giải nén vào `C:\nuscenes-project\datasets\nuscenes\`:

| Gói | Cần thư mục |
|---|---|
| nuScenes **Mini** (v1.0-mini) | `v1.0-mini\`, `samples\`, `sweeps\`, `maps\*.png` |
| **Map expansion** (v1.3) | `maps\expansion\*.json` (bắt buộc cho Task A: drivable area / walkway) |

Đặt dữ liệu ở chỗ khác thì đặt biến môi trường (PowerShell):

```powershell
$env:NUSCENES_DATAROOT = "D:\data\nuscenes"
```

Lưu ý: `OUTPUT_DIR` trong `config.py` đang hard-code `C:\nuscenes-project\output`. Đổi máy thì sửa dòng này.

### 2.2. Môi trường Python

Virtualenv có sẵn tại `nuscenes-env` (Python 3.10). Kích hoạt:

```powershell
cd C:\nuscenes-project
.\nuscenes-env\Scripts\Activate.ps1
```

PowerShell chặn script thì chạy một lần: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Tạo môi trường mới (máy khác):

```powershell
python -m venv nuscenes-env
.\nuscenes-env\Scripts\Activate.ps1
pip install -r n3-08c\requirements.txt
pip install pytest        # tuỳ chọn, để chạy test bằng pytest
```

### 2.3. Kiểm tra setup

```powershell
cd C:\nuscenes-project\n3-08c
python verify_setup.py            # in thống kê + mở cửa sổ Open3D
python verify_setup.py --no-vis   # không mở cửa sổ
```

Kết quả đúng: các dòng `[OK]`, `Scenes: 10  Samples: 404`, cuối cùng `verify_setup: PASS`.

> **Mọi lệnh bên dưới chạy trong thư mục `n3-08c\`** với venv đã kích hoạt.

---

## 3. Quy trình làm việc

Thứ tự khuyến nghị: setup → build template bank → Task A → Task B → evaluate.

### Bước 1 — Build template bank

```powershell
python task_a/extract_object.py --build    # trích template toàn dataset -> output\templates\
python task_a/extract_object.py --stats    # thống kê bank đã build
python task_a/extract_object.py            # demo: object nhiều điểm nhất của sample[0]
python task_a/extract_object.py --ann <ann_token> --show   # xem 1 template bằng Open3D
```

Tuỳ chọn: `--categories vehicle human.pedestrian` (prefix), `--min-points 20`.

Template = mảng `(M, 5)` `[x, y, z, intensity, ring]` ở object-local frame: gốc = tâm box, trục x = hướng đầu object, z hướng lên, đáy box ở `z = -h/2`.

### Bước 2 — Task A: chèn object

```powershell
python task_a/run_insertion.py                                     # demo: sample[0] -> sample[10]
python task_a/run_insertion.py --source-sample <tok> --target <tok> --num-objects 3 --vis
python task_a/run_insertion.py --source-ann <ann_token> --target <sample_token>
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--source-sample` | sample lấy object | `sample[0]` |
| `--source-ann` | danh sách annotation token cụ thể | — |
| `--target` | sample token đích | `sample[10]` |
| `--num-objects` | số object chèn | 3 |
| `--n-attempts` | số lần thử vị trí mỗi object | 100 |
| `--seed` | seed RNG | 42 |
| `--vis` | mở Open3D sau khi chèn | tắt |

Output:
- `output\augmented\<target>.bin` — point cloud `(N+M, 5)` float32, sensor frame, cùng format nuScenes.
- `output\augmented\<target>.json` — annotation mới (sensor + global frame), `n_requested`, `n_inserted`, `elapsed_s`.
- `output\vis\insertion_<target>.png` — ảnh BEV.

Pipeline bên trong (`run_insertion`):
1. `GroundModel.from_points` — ước lượng mặt đất (RANSAC + median z cục bộ).
2. `get_map_area` — vùng hợp lệ theo category (`PLACEMENT_LAYERS` trong config: xe → `drivable_area`, người → `walkway`, `ped_crossing`).
3. `sample_valid_position` — random (x, y, yaw) trong vùng, khoảng cách 3–70 m, không chồng box BEV.
4. `insert_object` — transform template tới vị trí, đáy chạm đất, concat vào scene.
5. `create_annotation` — sinh annotation mới.

### Bước 3 — Task B: mô phỏng thời tiết

```powershell
python task_b/run_weather.py                                    # sample[0], rain moderate
python task_b/run_weather.py --weather fog --level dense
python task_b/run_weather.py --all-levels                       # mọi weather x level + ảnh so sánh
python task_b/run_weather.py --input ..\output\augmented\<tok>.bin --weather rain --level heavy
```

Level:
- `rain`: `light` / `moderate` / `heavy` (dropout 0.1/0.2/0.3, noise 3/7/12 %).
- `fog`: `light` / `moderate` / `dense` (visibility 200/50/20 m).

`--input` nhận output của Task A → kết hợp chèn object + thời tiết.

Output (`output\weather\`):
- `original\<name>.bin` — bản gốc.
- `<weather>_<level>\<name>.bin` — point cloud sau mô phỏng.
- `<weather>_<level>\<name>_labels.npy` — `0` = điểm gốc, `1` = noise/backscatter.
- `output\vis\weather_<name>.png` — ảnh BEV so sánh.

### Bước 4 — Đánh giá

```powershell
python evaluate.py                       # đánh giá kết quả có sẵn trong output\
python evaluate.py --run-insertion 20    # chạy Task A trên 20 sample rồi đánh giá
```

Kết quả in ra màn hình và lưu `output\eval_report.json`:
- `insertion`: `valid_rate` (inserted / requested), `total_objects`, `scenes_per_hour`, `per_category`.
- `weather/<weather>_<level>`: `point_count_change` (mean_ratio, mean_diff), `intensity_stats` trước/sau.

---

## 4. Chạy từng module (demo)

Mỗi module có demo riêng, dùng để kiểm tra nhanh khi sửa code:

```powershell
python utils/nuscenes_loader.py
python utils/transforms.py
python utils/visualization.py     # lưu ảnh BEV vào output\vis
python task_a/ground_plane.py     # ảnh BEV vùng đặt object
python task_a/placement.py
python task_a/merge.py
python task_b/rain.py
python task_b/fog.py
```

---

## 5. Test

Test chạy trên data thật (cần dataset + map expansion).

```powershell
pytest tests/                       # cần pip install pytest
python tests/test_utils.py          # chạy không cần pytest
python tests/test_ground_plane.py
python tests/test_extract_object.py
```

| File | Kiểm tra |
|---|---|
| `test_utils.py` | loader, transforms khớp nuScenes devkit; vẽ BEV |
| `test_ground_plane.py` | RANSAC trên dữ liệu tổng hợp, GroundModel theo dốc, sai số với đáy box thật, map area |
| `test_extract_object.py` | template nằm trong box, roundtrip local frame, ground margin, save/load, build bank |

---

## 6. Cấu hình (`n3-08c\config.py`)

| Nhóm | Biến | Ghi chú |
|---|---|---|
| Dataset | `DATAROOT`, `VERSION`, `LIDAR_CHANNEL` | `DATAROOT` override bằng `NUSCENES_DATAROOT` |
| Output | `OUTPUT_DIR`, `TEMPLATE_DIR`, `AUGMENTED_DIR`, `WEATHER_DIR`, `VIS_DIR` | |
| Chung | `SEED = 42` | |
| Task A | `MIN_DISTANCE`, `MAX_DISTANCE` | khoảng cách đặt object (m) |
| | `COLLISION_IOU_THRESH` | BEV IoU > ngưỡng = va chạm |
| | `MIN_TEMPLATE_POINTS`, `TEMPLATE_GROUND_MARGIN`, `TEMPLATE_CATEGORIES` | lọc template |
| | `GROUND` | tham số RANSAC + GroundModel |
| | `PLACEMENT_LAYERS`, `DEFAULT_LAYERS` | map layer hợp lệ theo category |
| Task B | `RAIN_LEVELS`, `FOG_VISIBILITY` | thêm level mới ở đây |

---

## 7. Quy ước dữ liệu

- Point cloud: `(N, 5)` float32 `[x, y, z, intensity, ring]`, file `.bin` đọc bằng `np.fromfile(path, np.float32).reshape(-1, 5)` (`utils.nuscenes_loader.load_bin`).
- Mọi xử lý Task A/B ở **LIDAR_TOP sensor frame** (sensor cao khoảng 1.84 m, mặt đất z ≈ -1.8).
- Annotation dict: `translation` (3,), `size` = `[w, l, h]`, `rotation` = quaternion `[w, x, y, z]` (theo nuScenes).
- Chỉ 3 cột xyz bị biến đổi khi transform, intensity và ring giữ nguyên.

---

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không thấy v1.0-mini/sample.json trong ...` | Sai `DATAROOT`. Sửa `config.py` hoặc đặt `NUSCENES_DATAROOT`. |
| `[MISSING] ...\maps\expansion` | Tải và giải nén Map expansion vào `datasets\nuscenes\maps\`. |
| `ModuleNotFoundError: No module named 'pytest'` | `pip install pytest`, hoặc chạy test bằng `python tests/test_*.py`. |
| `ModuleNotFoundError: config` / `utils` | Chạy lệnh từ thư mục `n3-08c\`. |
| Cửa sổ Open3D không mở / treo | Dùng `--no-vis` hoặc bỏ `--vis`; ảnh BEV vẫn lưu trong `output\vis`. |
| `n_inserted < n_requested` | Không tìm được vị trí hợp lệ. Tăng `--n-attempts` hoặc chọn target có vùng đường rộng. |

---

## 9. Việc còn dở (TODO trong code)

- `merge.py`: xoá điểm target nằm trong box mới; xử lý occlusion (range image).
- `run_insertion.py`: chọn template theo category / khoảng cách phù hợp target.
- `rain.py`: mô hình suy giảm vật lý `alpha = 0.01 * R^0.6`.
- `fog.py`: hiệu chỉnh `BACKSCATTER_REF` theo dữ liệu thật.
- `run_weather.py`: thêm tuyết, kết hợp mưa + sương.
- `evaluate.py`: lý do reject (off_road / collision), so sánh phân bố points/box, thống kê theo dải khoảng cách.
