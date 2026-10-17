# Hướng dẫn dự án N3-08C — Synthetic Point Cloud Augmentation (nuScenes)

Dự án sinh dữ liệu LiDAR tổng hợp từ nuScenes v1.0-mini:

- **Task A — Object insertion:** cắt point cloud của vật thể thật (template), đặt vào scene khác ở vị trí hợp lệ (trên đường, không va chạm, chạm mặt đất), sinh annotation mới.
- **Task B — Weather simulation:** mô phỏng mưa và sương mù trên point cloud (dropout, noise, suy giảm intensity).
- **Evaluate:** đo tỉ lệ chèn thành công, throughput, thay đổi số điểm và intensity.

> **Thành viên mới:** làm lần lượt mục 2 (cài đặt) → mục 3 (chạy) → so kết quả với mục 4 (kết quả mong đợi). Gặp lỗi thì xem mục 8.

---

## 1. Cấu trúc thư mục

`<repo>` là thư mục bạn clone về (ví dụ `C:\nuscenes-project`). Code tự tính đường dẫn theo vị trí repo, clone vào đâu cũng được.

```
<repo>\
├── datasets\nuscenes\          # dữ liệu nuScenes — KHÔNG có trong git, tự tải (mục 2.2)
├── nuscenes-env\               # virtualenv Python 3.10 — KHÔNG có trong git, tự tạo (mục 2.3)
├── output\                     # kết quả sinh ra khi chạy — KHÔNG có trong git
│   ├── templates\              # template bank (<category>\<ann_token>.npz + bank.json)
│   ├── augmented_eval*\        # Task A (+ thời tiết): <sample_token>.bin + .json + batch_timing.json
│   ├── weather\                # Task B: original\, rain_<level>\, fog_<level>\, storm_storm\
│   ├── vis\                    # ảnh BEV / camera (.png)
│   └── eval_report_insertion*.json  # kết quả evaluate.py (chỉ số đề bài)
├── assets\vn_objects\          # ảnh PNG nền trong suốt để dán vật thể lên camera (README.md, SOURCES.md)
└── n3-08c\                     # CODE CHÍNH
    ├── config.py               # đường dẫn, seed, ngưỡng, tham số ground/weather
    ├── verify_setup.py         # kiểm tra dữ liệu + môi trường
    ├── evaluate.py             # đánh giá Task A + Task B
    ├── tools\dataset_gap.py    # thống kê thiếu hụt của nuScenes (pain point)
    ├── requirements.txt        # thư viện, đã ghi cố định phiên bản
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

## 2. Cài đặt (làm một lần)

### 2.1. Yêu cầu

| Thứ             | Phiên bản                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Hệ điều hành | Windows 10/11 (lệnh trong tài liệu viết cho PowerShell)                                                                         |
| Python           | **3.10** (đã test với 3.10.11). Không dùng 3.12+: `open3d==0.20.0` và `numpy==1.26.4` có thể không cài được. |
| Git              | bản bất kỳ                                                                                                                       |
| Ổ đĩa trống  | khoảng 8 GB (dataset ~5.5 GB, venv ~0.8 GB, output)                                                                                |

Kiểm tra: `py -3.10 --version` phải in ra `Python 3.10.x`.

### 2.2. Clone repo

```powershell
git clone <URL repo>
cd <thư mục repo>
```

### 2.3. Tải dữ liệu nuScenes

Dữ liệu không có trong git (giấy phép nuScenes không cho phân phối lại). Mỗi người tự tải:

1. Vào https://www.nuscenes.org/nuscenes#download, đăng ký tài khoản và đăng nhập.
2. Ở mục **Full dataset (v1.0)**, tải **Mini** (`v1.0-mini.tgz`, khoảng 4 GB).
3. Ở mục **Map expansion**, tải **Map expansion pack (v1.3)** (`nuScenes-map-expansion-v1.3.zip`).
4. Giải nén để có cấu trúc sau:

```
<repo>\datasets\nuscenes\
├── v1.0-mini\          # các file .json (sample.json, scene.json, ...)
├── samples\
├── sweeps\
└── maps\
    ├── *.png
    └── expansion\      # 4 file .json từ map expansion (boston-seaport.json, ...)
```

Lưu ý:

- Giải nén `.tgz` bằng 7-Zip hoặc lệnh `tar -xzf v1.0-mini.tgz -C datasets\nuscenes`.
- File zip map expansion có thể chứa sẵn thư mục `expansion\`, `basemap\`, `prediction\`. Chép cả thư mục `expansion\` vào `datasets\nuscenes\maps\`. Không để thành `maps\expansion\expansion\`.

Muốn để dữ liệu ở ổ khác thì đặt biến môi trường trước khi chạy:

```powershell
$env:NUSCENES_DATAROOT = "D:\data\nuscenes"
```

### 2.4. Tạo môi trường Python

```powershell
cd <thư mục repo>
py -3.10 -m venv nuscenes-env
.\nuscenes-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r n3-08c\requirements.txt
```

PowerShell báo lỗi "running scripts is disabled" thì chạy một lần, rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Mỗi lần mở terminal mới đều phải kích hoạt lại venv (`.\nuscenes-env\Scripts\Activate.ps1`). Dòng lệnh có tiền tố `(nuscenes-env)` là đã kích hoạt.

### 2.5. Kiểm tra setup

```powershell
cd n3-08c
python verify_setup.py --no-vis
```

Kết quả đúng: các dòng `[OK]`, `Scenes: 10  Samples: 404`, dòng cuối `verify_setup: PASS`.
Bỏ `--no-vis` để mở thêm cửa sổ Open3D xem point cloud.

> **Từ đây trở đi, mọi lệnh chạy trong thư mục `n3-08c\`, với venv đã kích hoạt.**

---

## 3. Quy trình chạy

Thứ tự: setup → build template bank → Task A → Task B → evaluate.

Chạy nhanh toàn bộ (khoảng vài phút, chủ yếu ở bước build):

```powershell
python task_a/extract_object.py --build
python task_a/run_insertion.py
python task_b/run_weather.py --all-levels
python evaluate.py --run-insertion 5
```

Sau đó so kết quả với mục 4. Chi tiết từng bước ở dưới.

### Bước 1 — Build template bank

```powershell
python task_a/extract_object.py --build    # trích template toàn dataset -> output\templates\
python task_a/extract_object.py --stats    # thống kê bank đã build
python task_a/extract_object.py            # demo: object nhiều điểm nhất của sample[0]
python task_a/extract_object.py --ann <ann_token> --show   # xem 1 template bằng Open3D
```

Tuỳ chọn: `--categories vehicle human.pedestrian` (lọc theo tiền tố category), `--min-points 20`.

Template là mảng `(M, 5)` `[x, y, z, intensity, ring]` ở object-local frame: gốc = tâm box, trục x = hướng đầu object, z hướng lên, đáy box ở `z = -h/2`.

### Bước 2 — Task A: chèn object

```powershell
python task_a/run_insertion.py                                     # demo: sample[0] -> sample[10]
python task_a/run_insertion.py --source-sample <tok> --target <tok> --num-objects 3 --vis
python task_a/run_insertion.py --source-ann <ann_token> --target <sample_token>
python task_a/run_insertion.py --from-bank --categories vehicle.motorcycle human.pedestrian --num-objects 6
python task_a/run_insertion.py --naive --out-dir ..\output\augmented_naive    # pipeline cũ để so sánh
```

| Tham số            | Ý nghĩa                                                                                    | Mặc định          |
| ------------------- | -------------------------------------------------------------------------------------------- | -------------------- |
| `--source-sample` | sample lấy object                                                                           | `sample[0]`        |
| `--source-ann`    | danh sách annotation token cụ thể                                                         | —                   |
| `--target`        | sample token đích                                                                          | `sample[10]`       |
| `--num-objects`   | số object chèn                                                                             | 3                    |
| `--n-attempts`    | số lần thử vị trí mỗi object                                                           | 100                  |
| `--seed`          | seed RNG                                                                                     | 42                   |
| `--categories`    | prefix category của template                                                                | `vehicle`          |
| `--from-bank`     | lấy template từ template bank (cân bằng category, bỏ template của chính scene đích) | tắt                 |
| `--naive`         | pipeline cũ: r, yaw ngẫu nhiên, không che khuất (chỉ để so sánh)                    | tắt                 |
| `--out-dir`       | thư mục ghi`.bin` + `.json`                                                            | `output\augmented` |
| `--vis`           | mở Open3D sau khi chèn                                                                     | tắt                 |

Output:

- `output\augmented\<target>.bin` — point cloud `(N+M, 5)` float32, sensor frame, cùng format nuScenes.
- `output\augmented\<target>.json` — annotation mới (sensor + global frame), `n_requested`, `n_inserted`, `elapsed_s`, `reject_stats` (lý do loại vị trí), `quality` (chỉ số vật lý của từng object chèn).
- `output\vis\insertion_<target>.png` — ảnh BEV.

Pipeline bên trong (`run_insertion`):

1. `GroundModel.from_points` — ước lượng mặt đất (RANSAC + median z cục bộ).
2. `get_map_area` — vùng hợp lệ theo category (`PLACEMENT_LAYERS` trong config: xe → `drivable_area`, người → `walkway`, `ped_crossing`).
3. `sample_position_for_template` — **xoay template quanh sensor**: giữ khoảng cách `r` của template (được dời ra xa tối đa ×`MAX_DISTANCE_SCALE`), chọn azimuth `φ` ngẫu nhiên, `yaw = φ − view_angle`. Nhờ vậy mặt có điểm vẫn quay về LiDAR và mật độ điểm đúng khoảng cách. Vị trí bị loại khi: footprint ra ngoài vùng (`off_area`), LiDAR không thấy mặt đất ở đó (`no_ground`), chồng box (`collision`), có vật cản không nhãn trong footprint (`obstacle`).
4. `insert_object` — đáy chạm đất, rồi áp luật LiDAR: subsample nếu dời ra xa (`(r_cũ/r_mới)²`), gán lại `ring` theo góc ngước, xoá điểm target trong box mới, **che khuất** bằng range image 32 ring × 1084 azimuth (mỗi tia chỉ giữ điểm gần nhất). Object còn < `MIN_VISIBLE_POINTS` điểm thì thử vị trí khác (`occluded`).
5. `create_annotation` — sinh annotation mới; `num_lidar_pts` = số điểm còn thấy sau che khuất (đếm lại sau khi chèn hết, vì object sau có thể che object trước).

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

`--input` nhận output của Task A, dùng để kết hợp chèn object + thời tiết.

Output (`output\weather\`):

- `original\<name>.bin` — bản gốc.
- `<weather>_<level>\<name>.bin` — point cloud sau mô phỏng.
- `<weather>_<level>\<name>_labels.npy` — `0` = điểm gốc, `1` = noise/backscatter.
- `output\vis\weather_<name>.png` — ảnh BEV so sánh.

### Bước 4 — Đánh giá

```powershell
python evaluate.py                       # đánh giá kết quả có sẵn trong output\
python evaluate.py --run-insertion 100 --augmented-dir ..\output\augmented_eval   # 100 sample rải đều, template từ scene khác
python evaluate.py --run-insertion 100 --weather rain heavy --augmented-dir ..\output\augmented_eval_rain   # chèn + mưa
python evaluate.py --run-insertion 20 --naive --augmented-dir ..\output\augmented_naive   # bản cũ để so
```

`--run-insertion` cần thư mục `--augmented-dir` trống, để số đo không bị trộn với kết quả cũ. Template lấy từ template
bank và loại scene đích. Thêm `--num-objects`, `--categories`, `--out` nếu cần.

Kết quả in ra màn hình và lưu `output\eval_report.json` (hoặc file ở `--out`):

- `validity` (**chỉ số đề bài**): kiểm tra lại từng scene trên file đầu ra, độc lập với bộ đặt vị trí.
  - `valid_scene_rate`: tỉ lệ scene có >= 1 object chèn, không chồng box BEV, đúng layer bản đồ và đáy chạm mặt đất.
  - `complete_scene_rate`: như trên và chèn đủ số object yêu cầu.
  - `cross_scene_rate`: tỉ lệ scene mà mọi template đến từ scene khác.
  - `fail_reasons` và `invalid_scenes`: lý do một scene không hợp lệ.
- `insertion`: `valid_rate` (object chèn được / yêu cầu, tính theo object), `total_objects`, `per_category`, `reject_stats`.
  - `scenes_per_hour` (**chỉ số đề bài**): wall-clock cả batch từ `batch_timing.json`, gồm lấy template, chèn, thời tiết
    và ghi file. `scenes_per_hour_core` chỉ tính thời gian trong `run_insertion`.
  - `timing`: thời gian từng bước và thông tin máy.
- `insertion.quality` (chỉ số vật lý): `facing_ok_rate` (mặt có điểm quay về sensor), `ground_supported_rate` (đặt ở chỗ LiDAR thấy mặt đất), `density_ratio_median` (số điểm / số điểm object thật cùng lớp, cùng khoảng cách, lấy từ template bank; ~1 là đúng), `visible_ratio_mean`.

Đo trên 30 sample đầu (seed 42), pipeline cũ (`--naive`) so với pipeline mới:

| Chỉ số                  | Cũ                           | Mới                                                     |
| ------------------------- | ----------------------------- | -------------------------------------------------------- |
| `valid_rate`            | 1.00                          | 0.85                                                     |
| `facing_ok_rate`        | 0.52                          | 0.94 (các object còn lại nhìn ngang hông, cos ≈ 0) |
| `ground_supported_rate` | 0.35                          | 1.00                                                     |
| `density_ratio_median`  | 4.39 (dày gấp 4 lần thật) | 1.32                                                     |

`valid_rate` giảm vì luật chặt hơn: vị trí sai vật lý bị loại thay vì được chấp nhận.

### Bước 5 — Thống kê thiếu hụt của dataset (pain point)

```powershell
python tools/dataset_gap.py                            # v1.0-mini -> output\dataset_gap_v1.0-mini.json
python tools/dataset_gap.py --version v1.0-trainval    # chỉ cần gói metadata của trainval
```

Đếm tỉ lệ lớp, xe 2 bánh (đang chạy / đỗ), bầy xe 2 bánh gần ego, khoảng cách tới xe đang chạy gần nhất, các proxy tình huống nguy hiểm (cắt đầu, ngược chiều, đi bộ giữa đường) và số scene mưa / đêm. Proxy chỉ dựa trên vị trí + hướng box ở 1 khung, xem docstring để biết giới hạn.

- `weather/<weather>_<level>`: `point_count_change` (mean_ratio, mean_diff), `intensity_stats` trước/sau.

---

## 4. Kết quả mong đợi

Số liệu dưới đây chạy trên nuScenes v1.0-mini, `SEED = 42`. Chạy cùng dữ liệu, cùng seed thì các con số (trừ thời gian
chạy) phải giống hoặc rất gần. File kết quả: `output/eval_report_insertion*.json`, `output/weather/`.

### 4.1. Template bank (`extract_object.py --build`)

Tổng **4111 template**, 14 category:

| Category                           | Số template | Category                             | Số template |
| ---------------------------------- | ------------ | ------------------------------------ | ------------ |
| vehicle.car                        | 2117         | vehicle.construction                 | 110          |
| human.pedestrian.adult             | 582          | vehicle.motorcycle                   | 90           |
| movable_object.barrier             | 579          | vehicle.trailer                      | 47           |
| vehicle.truck                      | 302          | movable_object.trafficcone           | 37           |
| vehicle.bus.rigid                  | 167          | vehicle.bus.bendy                    | 31           |
| human.pedestrian.personal_mobility | 20           | human.pedestrian.construction_worker | 17           |
| vehicle.bicycle                    | 7            | movable_object.pushable_pullable     | 5            |

### 4.2. Task A — chèn object

Chỉ số đề bài, 100 scene (`python evaluate.py --run-insertion 100 --augmented-dir ..\output\augmented_eval`):
`valid_scene_rate` 1.00, `complete_scene_rate` 0.59, `cross_scene_rate` 1.00, 253 object (0 chồng box, 0 ngoài đường,
0 lệch mặt đất), `scenes_per_hour` khoảng 9 400 (phụ thuộc máy). Kèm mưa to / sương dày: 0.99 / 1.00 scene hợp lệ,
khoảng 9 400 scene/giờ.

Thời tiết mặc định là mô hình v2. Muốn tái lập số detector thời tiết cũ (đo bằng v1) thì đặt `$env:N308C_WEATHER_MODEL="v1"`.
`evaluate.py` báo thêm `by_range` cho mỗi thư mục thời tiết: tỉ lệ điểm gốc còn lại và tỉ lệ nhiễu theo dải 0-10 / 10-30 / 30+ m.
Chi tiết: [docs/bao_cao_ket_qua.md](docs/bao_cao_ket_qua.md) mục 0.

| Chỉ số chất lượng (100 scene, chỉ chèn) | Giá trị |
| --- | --- |
| `facing_ok_rate` | 0.85 (còn lại nhìn ngang hông, cos ≈ 0) |
| `ground_supported_rate` | 1.00 |
| `density_ratio_median` | 1.01 (số điểm / object thật cùng lớp, cùng khoảng cách) |
| `visible_ratio_mean` | 0.67 |

Xem tận mắt: `python task_a/run_insertion.py --vis` (template từ scene khác -> sample[10]) lưu ảnh BEV vào `output\vis\`.
Object chèn phải nằm trên vùng đường, không chồng box có sẵn, chân chạm mặt đất, điểm nằm ở mặt quay về sensor.

### 4.3. Task B — thời tiết

Sample `ca9a282c...` (sample[0]), point cloud gốc 34 688 điểm, intensity trung bình 19.85. Mô hình thời tiết v2
(`python task_b/run_weather.py --all-levels`, rồi `python evaluate.py`).

| Weather / level | Tỉ lệ điểm còn lại | Số điểm thay đổi | Intensity TB sau | Điểm gốc còn lại 0-10 / 10-30 / 30+ m |
| --- | --- | --- | --- | --- |
| rain light | 0.892 | -3 743 | 17.65 | 0.93 / 0.80 / 0.59 |
| rain moderate | 0.798 | -6 996 | 16.22 | 0.84 / 0.60 / 0.29 |
| rain heavy | 0.714 | -9 905 | 13.94 | 0.74 / 0.42 / 0.12 |
| fog light | 0.947 | -1 827 | 18.79 | 0.98 / 0.92 / 0.79 |
| fog moderate | 0.824 | -6 104 | 16.89 | 0.91 / 0.70 / 0.40 |
| fog dense | 0.674 | -11 323 | 14.76 | 0.78 / 0.41 / 0.11 |
| storm | 0.674 | -11 311 | 14.29 | 0.71 / 0.34 / 0.08 |

Cách đọc: level càng nặng thì càng mất nhiều điểm, intensity càng giảm, và càng xa càng mất nhiều. Kết quả sai nếu
không theo xu hướng này. Ảnh so sánh: `output\vis\weather_<token>.png`.

---

## 5. Chạy từng module (demo)

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

## 6. Test

Test chạy trên data thật (cần dataset + map expansion). `pytest` đã có trong `requirements.txt`.

```powershell
pytest tests/
python tests/test_utils.py          # chạy riêng từng file, không cần pytest
python tests/test_ground_plane.py
python tests/test_extract_object.py
python tests/test_merge_placement.py
```

| File                        | Kiểm tra                                                                                                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_utils.py`           | loader, transforms khớp nuScenes devkit; vẽ BEV                                                                                                                                                                               |
| `test_ground_plane.py`    | RANSAC trên dữ liệu tổng hợp, GroundModel theo dốc, sai số với đáy box thật, map area                                                                                                                                |
| `test_extract_object.py`  | template nằm trong box, roundtrip local frame, ground margin, save/load, build bank                                                                                                                                            |
| `test_merge_placement.py` | gán ring khớp LiDAR thật, che khuất 2 chiều, subsample theo khoảng cách, xoá điểm trong box, placement giữ r + view_angle, loại vị trí không thấy đất / có vật cản, chèn trên data thật đúng vật lý |

Trước khi push code, chạy `pytest tests/` và đảm bảo không có test fail.

---

## 7. Cấu hình (`n3-08c\config.py`)

| Nhóm         | Biến                                                                             | Ghi chú                                                                                    |
| ------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Đường dẫn | `PROJECT_ROOT`                                                                  | tự tính = thư mục cha của`n3-08c\`                                                   |
| Dataset       | `DATAROOT`, `VERSION`, `LIDAR_CHANNEL`                                      | `DATAROOT` mặc định `<repo>\datasets\nuscenes`, override bằng `NUSCENES_DATAROOT` |
| Output        | `OUTPUT_DIR`, `TEMPLATE_DIR`, `AUGMENTED_DIR`, `WEATHER_DIR`, `VIS_DIR` | `OUTPUT_DIR` mặc định `<repo>\output`, override bằng `N308C_OUTPUT_DIR`           |
| Chung         | `SEED = 42`                                                                     | đổi seed thì kết quả mục 4 sẽ khác                                                  |
| Task A        | `MIN_DISTANCE`, `MAX_DISTANCE`                                                | khoảng cách đặt object (m)                                                              |
|               | `COLLISION_IOU_THRESH`                                                          | BEV IoU > ngưỡng = va chạm                                                               |
|               | `COLLISION_MARGIN`                                                              | lề an toàn (m) giữa box khi chèn (Task A); kịch bản bầy xe dùng 0                        |
|               | `MIN_TEMPLATE_POINTS`, `TEMPLATE_GROUND_MARGIN`, `TEMPLATE_CATEGORIES`      | lọc template                                                                               |
|               | `GROUND`                                                                        | tham số RANSAC + GroundModel                                                               |
|               | `PLACEMENT_LAYERS`, `DEFAULT_LAYERS`                                          | map layer hợp lệ theo category                                                            |
| Task B        | `RAIN_LEVELS`, `FOG_VISIBILITY`                                               | thêm level mới ở đây                                                                   |
|               | `WEATHER_MODEL` (env `N308C_WEATHER_MODEL`)                                   | `v2` mặc định; `v1` để tái lập số đo thời tiết cũ                                      |

### Quy ước dữ liệu

- Point cloud: `(N, 5)` float32 `[x, y, z, intensity, ring]`, file `.bin` đọc bằng `np.fromfile(path, np.float32).reshape(-1, 5)` (`utils.nuscenes_loader.load_bin`).
- Mọi xử lý Task A/B ở **LIDAR_TOP sensor frame** (sensor cao khoảng 1.84 m, mặt đất z ≈ -1.8).
- Annotation dict: `translation` (3,), `size` = `[w, l, h]`, `rotation` = quaternion `[w, x, y, z]` (theo nuScenes).
- Khi transform chỉ 3 cột xyz thay đổi, intensity và ring giữ nguyên.

---

## 8. Lỗi thường gặp

| Lỗi                                                                         | Cách xử lý                                                                                             |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `Không thấy v1.0-mini/sample.json trong ...`                             | Dữ liệu chưa đúng chỗ. Kiểm tra cấu trúc mục 2.3, hoặc đặt`NUSCENES_DATAROOT`.             |
| `[MISSING] ...\maps\expansion`                                             | Chưa có map expansion, hoặc bị lồng`maps\expansion\expansion\`. Xem mục 2.3.                      |
| `running scripts is disabled on this system`                               | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.                                                  |
| `pip install` lỗi ở `open3d` hoặc `numpy`                           | Sai phiên bản Python. Xoá`nuscenes-env\`, tạo lại bằng `py -3.10 -m venv nuscenes-env`.         |
| `ModuleNotFoundError: No module named 'nuscenes'` (hoặc thư viện khác) | Chưa kích hoạt venv. Chạy`.\nuscenes-env\Scripts\Activate.ps1`.                                     |
| `ModuleNotFoundError: config` / `utils`                                  | Chạy lệnh từ thư mục`n3-08c\`.                                                                     |
| Cửa sổ Open3D không mở / treo                                            | Dùng`--no-vis` hoặc bỏ `--vis`; ảnh BEV vẫn lưu trong `output\vis`.                           |
| `n_inserted < n_requested`                                                 | Không tìm được vị trí hợp lệ. Tăng`--n-attempts` hoặc chọn target có vùng đường rộng. |

---

## 9. Làm việc nhóm với Git

- Không commit `datasets\`, `nuscenes-env\`, `output\` (đã có trong `.gitignore`). GitHub chặn file trên 100 MB.
- Cập nhật code mới nhất trước khi làm: `git pull`.
- Làm trên nhánh riêng, ví dụ `git checkout -b task-b-snow`, rồi tạo Pull Request.
- Thêm thư viện mới thì ghi kèm phiên bản vào `n3-08c\requirements.txt`.
- Thay đổi làm kết quả mục 4 khác đi thì chạy lại lệnh tương ứng và cập nhật bảng ở mục 4.

---

## 10. Việc còn dở (TODO trong code)

- `merge.py`: chỉnh intensity theo khoảng cách mới (hiện giữ nguyên vì r lệch ít).
- `placement.py`: yaw theo hướng lane (kịch bản xe ngược chiều, cắt đầu — task T2).
- `rain.py`: mô hình suy giảm vật lý `alpha = 0.01 * R^0.6`.
- `fog.py`: hiệu chỉnh `BACKSCATTER_REF` theo dữ liệu thật.
- `run_weather.py`: thêm tuyết, kết hợp mưa + sương.
- `evaluate.py`: thống kê Task B theo dải khoảng cách.
