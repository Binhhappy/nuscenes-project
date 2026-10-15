# N3-08C — Tổng hợp kết quả (số liệu cho báo cáo)

Mọi số đo trên nuScenes **v1.0-mini**, detector **CenterPoint-PointPillar pretrained trên nuScenes** (OpenPCDet),
đánh giá trên **mini_val** (2 scene model chưa thấy khi train). Khớp prediction - GT kiểu nuScenes: cùng lớp, tâm
BEV < 2 m, score >= 0.3. CI 95% = bootstrap theo khung hình. Biểu đồ: `output/report/`, số liệu gốc: `output/report/summary.json`.

## 1. Vấn đề — nuScenes thiếu tình huống giao thông Việt Nam

| Chỉ số (nuScenes mini, 404 khung) | Giá trị | Nguồn |
|---|---|---|
| Khung có bầy >= 5 xe máy đang chạy trong 15 m | **0%** (tối đa 2 xe) | `output/dataset_gap_v1.0-mini.json` |
| Khung có xe cắt đầu | 0.5% | như trên |
| Xe đang chạy cách xe < 3 m | 0% | như trên |
| Tỉ lệ xe 2 bánh trong mọi object | 3.9% | như trên |
| Xe ba gác, xe máy chở hàng, xe hàng rong, xe bò | không có lớp / không có mẫu | taxonomy nuScenes |

Hệ quả đo được: ngay trên dữ liệu thật, detector chỉ tìm ra **55% xe máy** (CI 45-64%), so với ô tô 93%, người 95%.

## 2. Giải pháp — sinh edge case LiDAR trên dữ liệu thật

Không tạo cảnh mới: lấy khung LiDAR thật của nuScenes rồi chèn vật thể theo kịch bản VN.

1. **Nguồn điểm vật thể**: cắt mẫu object thật nuScenes (xe máy 1 người, ô tô, người) hoặc **ray-cast** hình dựng từ
   kích thước công bố (xe máy chở 2-3 người / chở hàng, ba gác, xe kéo, xe hàng rong, chó, bò, xe bò), bắn đúng 32 tia
   của Velodyne HDL-32E. Danh mục + quy ước: `docs/danh_muc_kich_ban.md`.
2. **Kịch bản**: luật vị trí / hướng / tốc độ trong hệ toạ độ xe (`task_a/scenarios.py`, `task_a/vn_scenarios.py`);
   nhiều khung liên tục (`task_a/multiframe.py`).
3. **Ghép theo vật lý LiDAR** (`task_a/merge.py`): mật độ theo khoảng cách, chỉ mặt nhìn thấy, che khuất 2 chiều
   (mỗi tia 1 điểm gần nhất), gán ring, mặt đất cục bộ, đủ 10 sweep như model đọc (`task_a/sweeps.py`).
4. **Thời tiết** (`task_b/`): mưa, sương, **giông / bão** (mưa rất to + mặt đường ướt + bụi nước sau bánh xe).
5. **Đánh giá có đối chứng** (`tools/t6_eval.py`): cùng vật thể, cùng pipeline chèn, chỉ khác tình huống
   (vị trí bình thường vs kịch bản VN) -> khác biệt là do tình huống, không do dữ liệu giả.

### Kiểm chứng dữ liệu sinh ra
| Kiểm chứng | Kết quả |
|---|---|
| Chèn cắt mẫu: mật độ so với object thật cùng khoảng cách | 0.98 lần; 100% đúng mặt nhìn thấy, 100% có mặt đất (`output/eval_report.json`) |
| Ray-cast: số điểm ô tô so với ô tô thật (7-12 m) | 308 vs median 305 / p75 432 (`python task_a/raycast.py`) |
| Ray-cast: số điểm xe máy so với xe máy thật (7-12 m) | 101 vs median 52 / p75 95 |
| Detector thấy ô tô ray-cast như ô tô thật (vị trí bình thường) | **93% (CI 90-96%) vs 93%** -> đạt |
| Hình ảnh điểm LiDAR từng vật thể chèn, cạnh xe máy / ô tô thật | `output/report/vn_closeup.png` (ray-cast đều và liền hơn object thật — khác biệt còn lại) |
| Detector thấy xe máy ray-cast như xe máy cắt mẫu | bản 1 (thân 0.34 m): 34% vs 94%, 61% bị nhận là xe đạp. Bản 2 (thân theo kích thước hãng ~0.6 m): **64%** (CI 59-69%) vs 94% -> **chưa đạt**: hình dựng từ khối vẫn hay bị nhận là xe đạp. Không chỉnh tiếp (chỉnh nữa là chỉnh cho vừa detector); kịch bản xe máy ray-cast chỉ so với chính đối chứng ray-cast |

## 3. Kết quả

### 3.1 Bầy xe máy là điểm yếu thật của detector (`output/report/t6_motorcycle_recall.png`)
| Nhóm (xe máy) | Recall | CI 95% |
|---|---|---|
| Đối chứng: chèn, đi cùng chiều | 93% | 91-96% |
| Bầy 10-20 xe máy | **81%** | 79-83% |
| Xe máy cắt đầu 3-8 m | 100% | 100% |
| Hiệu bầy - đối chứng | **-12.1 điểm** | -14.8 .. -9.1 (không chứa 0 -> có ý nghĩa) |

- Cùng số điểm LiDAR vẫn kém hơn (`t6_swarm_by_points.png`): 1-19 điểm 42% vs 86%, 20-49 điểm 79% vs 94%.
- Bầy càng đông càng kém: 5-9 xe 87%, 10-14 xe 83%, 15-19 xe 78%.
- Lỗi chính: xe máy trong bầy bị nhận thành **ô tô** (nhiều xe sát nhau -> 1 box ô tô) hoặc xe đạp.
- Kịch bản 1 xe (cắt đầu, ngược chiều 98% không lật hướng, người băng đường 100%, xe lấn làn 95%): detector **vẫn thấy**;
  rủi ro của các tình huống này nằm ở dự đoán quỹ đạo / lập kế hoạch, ngoài phạm vi đề tài.

### 3.2 Thời tiết (`output/report/weather.png`, `weather_swarm.png`; 41 khung mini_val)
| Recall | Trời trong | Mưa to | Sương dày | Giông / bão |
|---|---|---|---|---|
| Ô tô | 93% | 80% | 73% | 71% |
| Người đi bộ | 95% | 64% | 51% | 55% |
| Xe máy (thật) | 54% | 38% | 40% | 35% |
| Xe máy trong bầy | 80% | 64% | 76% | **57%** |

Bầy xe máy trong giông là tổ hợp tệ nhất cho xe máy: hai yếu tố cộng dồn.

### 3.3 Vật thể VN không có lớp trong nuScenes (`output/report/vn_objects_recall.png`)
Đo "có phát hiện vật cản không" (box bất kỳ tại chỗ) — điều kiện cần cho an toàn, dễ hơn đo đúng tên lớp.

| Vật thể | Recall vật cản |
|---|---|
| Vật thể | Recall vật cản (CI 95%) | Median điểm LiDAR |
|---|---|---|
| Xe ba gác | 96% (93-99) | 60 |
| Trâu / bò trên đường | 98% (95-99) | 34 |
| Xe hàng rong lấn lòng đường | 95% (93-98) | 46 |
| Xe máy kéo thùng | 93% (88-96) | 58 |
| Chó lao ra (median 10.6 m) | 94% (90-98) | **11** |
| Xe bò: xe gỗ / con bò | 91% / **53%** (bò bị chính xe gỗ che, median 9 điểm) | |

Kết hợp (mix): phố chợ (xe hàng rong + người dưới lòng đường) 98%; giờ cao điểm 76%; chó lao ra giữa bầy xe máy:
chó 85% (giảm từ 94% khi đứng một mình), xe máy 84%.

**Xe máy ray-cast** (đo đúng lớp motorcycle, so với đối chứng xe máy ray-cast 64%):

| Kịch bản | Recall | Hiệu so với đối chứng ray-cast (CI 95%) | Thấy vật cản (không xét tên) |
|---|---|---|---|
| Xe máy chở 2-3 người | 60% | -3.9 điểm (-11.5 .. +3.7) -> **không khác biệt** | 98% |
| Xe máy chở hàng cồng kềnh | **29%** | **-34.7 điểm (-42.4 .. -26.3)** -> có ý nghĩa | 79% |

Xe chở hàng cồng kềnh — rất phổ biến ở VN — bị nhận thành **ô tô** (59 lần) hoặc xe đạp (46), và **21% bị bỏ sót hẳn**:
khối hàng rộng 1-1.8 m làm xe máy không còn giống xe máy nào trong dữ liệu train.

### 3.3b Người qua đường: ít hay nhiều người (`output/report/pedestrians_recall.png`)
| Số người | Recall (CI 95%) | n |
|---|---|---|
| 1 người băng ngang ngoài vạch | 100% (99-100) | 243 |
| Nhóm 3-8 người băng ngang cùng lúc | 99% (98-100) | 834 |
| Đám đông 10-20 người dưới lòng đường | 99% (99-100) | 2362 |

Theo số người chèn trong khung: 0-3 người 96%, 4-7 người 100%, 8-11 người 99%, 12-15 người 99%, >= 16 người 99%.
Detector **không** yếu đi khi đông người — trái với bầy xe máy (-12 điểm). Lý giải: nuScenes có rất nhiều người đi bộ
theo đám (4765 người lớn trong mini) nhưng 0% bầy xe máy -> model yếu đúng ở chỗ dữ liệu train thiếu (ủng hộ pain point).
Người đi bộ chỉ tụt mạnh khi thời tiết xấu (95% -> 55% giông bão, mục 3.2).

### 3.4 Giới hạn cảm biến (`output/report/lidar_beams.png`)
Số điểm LiDAR rơi trên **con chó** (ray-cast, cùng FOV dọc): 32 tia (nuScenes) 15 điểm ở 10 m, 4 ở 20 m, **0 từ 40 m**;
128 tia: 63 / 14 / 2. Xe máy: 32 tia 80 / 19 / 4 điểm ở 10 / 20 / 40 m. Vật thể nhỏ, thấp gần như vô hình với LiDAR 32 tia ở xa, dù model tốt đến đâu.

### 3.5 Minh hoạ từng trường hợp
Màu box: xanh lá = phát hiện đúng, đỏ = bỏ sót, cam = nhận nhầm lớp (ghi lớp bị nhận thành), nét đứt xanh dương =
prediction, điểm tím = điểm LiDAR của object chèn. Khung minh hoạ chọn cố định (nhiều object chèn nhất trong 20 m),
không chọn theo kết quả; mỗi ảnh ghi cả recall toàn kịch bản.

| File | Nội dung |
|---|---|
| `output/cases/grid_t6.png` | 6 kịch bản T6 (đối chứng, bầy xe máy, cắt đầu, ngược chiều, băng đường, lấn làn) |
| `output/cases/grid_vn.png` | 14 kịch bản vật thể VN (chở 2-3 người, chở hàng, ba gác, kéo thùng, hàng rong, chó, bò, xe bò, mix, nhóm người, đám đông) |
| `output/cases/t6_*.png`, `vn_*.png` | 20 ảnh lẻ, phóng vào object chèn |
| `docs/demo/index.html` | trang demo offline: xem 3D từng trường hợp, thời tiết, video, biểu đồ |
| `output/cases/weather_swarm.png` | cùng 1 khung bầy xe máy: trời trong / mưa to / sương dày / giông |
| `output/report/vn_closeup.png` | cận cảnh điểm LiDAR từng vật thể VN cạnh xe máy / ô tô thật |

### 3.6 Video demo (`output/videos/`, 2 khung / giây; kèm .png khung giữa)
| Video | Trái / phải | Kết quả object chèn |
|---|---|---|
| `scene-0103_swarm_pretrained.mp4` | nuScenes gốc / có bầy xe máy | 416/520 đúng (80%), 52 nhầm lớp, 52 sót |
| `scene-0103_cut_in_pretrained.mp4` | nuScenes gốc / xe máy cắt đầu | 40/40 |
| `scene-0103_cargo_pretrained.mp4` | nuScenes gốc / 2 xe máy chở hàng cồng kềnh | **1/53 đúng (2%)**, 29 nhầm lớp, 23 sót |
| `scene-0103_dog_pretrained.mp4` | nuScenes gốc / chó lao ngang (12 khung) | 6/7 thấy vật cản |
| `scene-0103_swarm_storm.mp4` | bầy xe máy trời trong / cùng cảnh giông bão | trời trong 80% -> **giông 42%** (221/520), 232 sót |

Video là 1 scene minh hoạ (số trong video khác số thống kê ở 3.1-3.3, là trên toàn mini_val).

## 4. Hạn chế (nên ghi trung thực)
- Dữ liệu nhỏ: nuScenes mini, chỉ 6 xe máy có người lái để cắt mẫu; mini_val 2 scene.
- Kích thước / hình vật thể ray-cast và tham số thời tiết là giả định có cơ sở, chưa hiệu chỉnh theo đo thật ở VN.
- Ray-cast dùng góc tia theo thông số HDL-32E (cột ring của nuScenes hỏng ở 7/81 scan mini_val; bản đầu dùng cột này
  làm bắn dồn tia — đã sửa, mọi số ray-cast trong tài liệu là sau khi sửa).
- Bản đồ nền là Singapore / Boston (đô thị): xe bò, trâu bò là "vật cản bất thường", không phải bối cảnh nông thôn.
- Đo vật cản không phân lớp là thước đo dễ dãi (model có thể gọi sai tên).
- Xe máy dựng bằng khối + ray-cast chưa được detector nhận như xe máy thật (64% vs 94%): số tuyệt đối của kịch bản
  xe máy ray-cast không dùng được, chỉ dùng phép so với đối chứng ray-cast cùng hình. Cần mesh 3D thật để cải thiện.
- Fine-tune (T7) chưa thành công: cả bản có và không có dữ liệu VN đều làm model kém đi trên dữ liệu thật
  (mAP 0.437 -> 0.351); trên bầy xe máy chưa thấy, bản có dữ liệu VN đạt 99% so với 89% của bản không có — dấu hiệu
  tích cực nhưng quy trình fine-tune cần sửa (nghi BatchNorm với batch 2). Không trình bày như giải pháp đã chứng minh.
