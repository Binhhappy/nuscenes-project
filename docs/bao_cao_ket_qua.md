# N3-08C — Tổng hợp kết quả (số liệu cho báo cáo)

Mọi số đo trên nuScenes **v1.0-mini**, detector **CenterPoint-PointPillar pretrained trên nuScenes** (OpenPCDet),
đánh giá trên **mini_val** (2 scene model chưa thấy khi train). Khớp prediction - GT kiểu nuScenes: cùng lớp, tâm
BEV < 2 m, score >= 0.3. CI 95% = bootstrap theo khung hình. Biểu đồ: `output/report/`, số liệu gốc: `output/report/summary.json`.

## 0. Chỉ số theo đề bài

Đề bài: lấy điểm LiDAR của vật thể đã gán nhãn, chèn vào scene khác kèm cuboid; mô phỏng mưa / sương bằng cách
bỏ bớt và làm nhiễu điểm. Chỉ số: **tỉ lệ scene hợp lệ** (không chồng vật thể, đúng mặt đường) và **số scene / giờ**.

Cách đo (`n3-08c/evaluate.py`): 100 sample rải đều toàn bộ nuScenes mini, mỗi sample yêu cầu chèn 3 xe. Template
lấy từ template bank (4111 object thật) và **loại scene đích**, nên mọi vật thể đều đến từ scene khác. Tỉ lệ hợp lệ
được kiểm tra lại **độc lập** trên file đầu ra (`validate_scene`), không dựa vào bộ đặt vị trí. Bộ kiểm tra dựng lại
box có sẵn, bản đồ và mặt đất từ nuScenes gốc. Một scene hợp lệ khi có >= 1 vật thể chèn và mọi vật thể:
- không giao box nào trong BEV (box có sẵn và vật thể chèn trước);
- có footprint nằm trọn trong layer bản đồ đúng loại (xe: `drivable_area`, người: `walkway` / `ped_crossing`);
- có đáy box chạm mặt đất mà LiDAR thấy (sai số <= 0.3 m).

| Chỉ số (100 scene, seed 42) | Chỉ chèn | Chèn + mưa to | Chèn + sương dày |
|---|---|---|---|
| **Tỉ lệ scene hợp lệ** | **100%** | **99%** | **100%** |
| Scene chèn đủ 3/3 vật thể và hợp lệ | 59% | 59% | 60% |
| Scene có vật thể lấy từ scene khác | 100% | 100% | 100% |
| Vật thể chèn / chồng nhau / ra ngoài đường / lệch mặt đất | 253 / 0 / 0 / 0 | 254 / 0 / 0 / 0 | 254 / 0 / 0 / 0 |
| Sai số đáy box - mặt đất lớn nhất | 0.002 m | 0.006 m | 0.019 m |
| **Số scene / giờ** (wall-clock cả batch, gồm ghi file) | **~9 400** | **~9 400** | **~9 400** |
| Thời gian trung bình / scene | 0.38 s | 0.38 s | 0.38 s |

Máy đo: CPU AMD Zen 3 (Family 25 Model 80, 16 luồng), Windows 11, Python 3.10, chạy đơn luồng. Chưa tính load nuScenes + template bank
(một lần, ~0.5 s). Phần lớn thời gian nằm ở bước chèn (mặt đất đồng thuận 3 lần RANSAC, che khuất); bước thời tiết tốn ~3 ms / scene.

Pipeline đo ở trên có 2 luật chặt hơn bản đầu:
- **Mặt đất đồng thuận** (`ConsensusGround`): chỉ đặt vật thể ở chỗ cả 3 lần RANSAC (seed khác nhau) đều thấy mặt đất,
  và độ cao các lần lệch nhau <= 0.15 m.
- **Lề an toàn** 0.3 m giữa các box (`config.COLLISION_MARGIN`).

So với bản đầu (99% / 98% / 99% hợp lệ, ~12 000 scene/giờ):
- **Lỗi lệch mặt đất giảm từ 3 xuống 0.** Ở bản đầu, 3 vật thể được đặt ở chỗ mặt đất ước lượng không ổn định: đổi seed
  RANSAC thì chỗ đó mất điểm mặt đất, hoặc lệch 0.6-0.9 m.
- Đánh đổi: tỉ lệ đủ 3/3 giảm từ 65-68% xuống 59-60%, và mỗi cảnh chậm thêm ~0.09 s.

Scene không hợp lệ còn lại: 1 scene (lượt mưa) không đặt được vật thể nào, do vùng đường quanh xe quá hẹp.

Scene chèn thiếu vật thể (~40%) vẫn hợp lệ: bộ đặt vị trí từ chối vị trí sai thay vì chèn ẩu. Lý do từ chối nhiều
nhất là `off_area`: template giữ khoảng cách gốc để mật độ điểm đúng, nên vị trí mới hay rơi ra ngoài đường.

So sánh: kết quả cũ trong `output/augmented` lấy template từ sample kế tiếp nên **0%** khác scene; 3/6 scene do
pipeline cũ (không kiểm tra mặt đất) sinh ra bị đánh `off_ground`.

Chạy lại:
```powershell
cd n3-08c
python evaluate.py --run-insertion 100 --augmented-dir ..\output\augmented_eval --out ..\output\eval_report_insertion.json
python evaluate.py --run-insertion 100 --weather rain heavy --augmented-dir ..\output\augmented_eval_rain --out ..\output\eval_report_insertion_rain.json
python evaluate.py --run-insertion 100 --weather fog dense --augmented-dir ..\output\augmented_eval_fog --out ..\output\eval_report_insertion_fog.json
```

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
| Chèn cắt mẫu: mật độ so với object thật cùng khoảng cách | 1.01 lần (median, 198 object); 85% mặt có điểm quay về sensor (còn lại nhìn ngang hông), 100% có mặt đất (`output/eval_report_insertion.json`, 100 scene) |
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

Số trong bảng đo bằng **mô hình thời tiết v1**, gồm 3 điểm khác v2:
- v1 bỏ điểm đều ở mọi khoảng cách; v2 bỏ điểm tăng theo khoảng cách.
- v1 rải nhiễu mưa trong khối cầu; v2 đặt nhiễu trên đúng tia thật.
- v1 cộng thêm điểm backscatter sương; v2 cho điểm backscatter thay điểm gốc cùng tia.

Để tái lập đúng bảng trên, đặt `N308C_WEATHER_MODEL=v1` rồi chạy `tools/weather_eval.py`. Với v2 (mặc định hiện nay),
mưa to giữ lại 74% / 42% / 12% điểm gốc ở 0-10 / 10-30 / 30+ m, còn v1 giữ ~70% ở mọi dải
(`evaluate.py`, mục `by_range`). Vì vậy recall với v2 dự kiến giảm mạnh hơn ở xa. Chưa đo lại bằng detector.

### 3.2b Kiểm chứng mặt đường ướt bằng dữ liệu thật (`tools/wet_road_check.py`, `output/report/wet_road_check.png`)
nuScenes mini **không có cảnh đang mưa**. Cảnh duy nhất liên quan là scene-1094, "Night, after rain" (đường ướt).
Cảnh này được so với scene-1077 và scene-1100: cùng ban đêm, cùng khu Holland Village, cùng cảm biến, nhưng đường khô.
- Điểm mặt đường: điểm cách mặt phẳng RANSAC dưới 0.2 m.
- Mỗi cảnh khoảng 40 khung; CI 95% tính bằng bootstrap theo khung.

| Dải | Điểm mặt đường còn lại (so với 5-10 m): ướt thật / khô thật / khô + mô hình | Cường độ: ướt thật / khô thật / khô + mô hình |
|---|---|---|
| 10-20 m | 0.33 / 0.35 / 0.30 | 5.4 / 6.3 / 6.5 |
| 20-30 m | 0.04 / 0.05 / 0.04 | 5.2 / 4.9 / 4.9 |
| 5-10 m | (tham chiếu) | **7.7 [7.2-8.3] / 10.1 [9.6-10.4] / 10.0** |

- **Dữ liệu thật xác nhận:** đường ướt làm điểm mặt đường **tối hơn ~23%** ở 5-10 m. Hai khoảng tin cậy không chồng nhau.
- **Mất điểm ở xa** do đường ướt chỉ nhỏ, khoảng 6-20%, và sát ngưỡng ý nghĩa. Mô hình `wet_road_mask` mất điểm **hơi nhiều
  hơn thực tế** ở 10-20 m (0.30 so với 0.33) và khớp ở 20-30 m.
- **Mô hình còn thiếu:** chưa làm giảm cường độ mặt đường ướt. Có thể bổ sung hệ số ~0.77, lấy từ chính đo đạc này.
- Hạn chế:
  - chỉ có 1 cảnh ướt;
  - mỗi cảnh là một con đường khác, nên phải chuẩn hoá theo dải gần;
  - "after rain" có thể ướt ít hơn lúc giông;
  - **mưa đang rơi chưa kiểm chứng được**, vì cần các cảnh mưa của nuScenes bản đầy đủ.

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

### 3.5b Dán ảnh 2D object lên camera (minh hoạ, `n3-08c/utils/camera_paste.py`)
Cùng ý tưởng với chèn trên LiDAR: ảnh object được **cắt từ chính ảnh camera gốc** của annotation đã lấy điểm LiDAR.
Ảnh được biến đổi phối cảnh (homography từ 8 góc box nguồn sang 8 góc box đích) rồi dán vào vị trí chèn. Không vẽ
tay, không sinh bằng AI.

Ảnh nguồn rõ nhất được chọn qua 2 bước.

**Bước 1, cổng loại.** Ảnh nguồn bị loại nếu:
- box bị cắt mép ảnh;
- visibility nuScenes < 60%;
- bị che > 10%, đo bằng box có nhãn phía trước **và** điểm LiDAR phía trước (để bắt cả vật không nhãn như cây, cột);
- box cao < 24 px, hoặc ảnh nhoè;
- lệch góc nhìn > 30°;
- lệch độ sáng > 1.8 lần, hoặc khác điều kiện đêm / mưa. Đây là **cổng mềm**: không có ảnh cùng điều kiện thì vẫn dán
  ảnh tốt nhất còn lại, trạng thái `pasted_relaxed`, màu kéo mạnh hơn về nền đích.

**Bước 2, xếp hạng.** Ưu tiên đúng khung của point cloud, rồi điểm `sqrt(diện tích) x (1 - bị che) x độ nét x cos(lệch góc)`.

Khi dán:
- object thật gần camera hơn sẽ che ảnh dán;
- object xa được dán trước, object gần dán sau.

**Vật thể không có trong nuScenes** (xe ba gác, bò, chó, xe hàng rong...): dán ảnh PNG nền trong suốt tải về, đặt ở
`assets/vn_objects/<loại>/front.png | rear.png | side.png` (xem `assets/vn_objects/README.md`).
- Chọn mặt theo góc camera nhìn vật thể: |góc| <= 45° dùng front, >= 135° dùng rear, còn lại dùng side, có lật ngang
  theo hướng đầu.
- Chiều cao ảnh bằng chiều cao box chiếu lên camera, đáy chạm điểm chiếu tâm đáy box, giữ tỉ lệ ảnh.
- Thêm bóng đổ mờ; che khuất và chỉnh màu như trên.

Thứ tự nguồn ảnh: ảnh nuScenes gốc đạt mọi tiêu chí → ảnh tải về → ảnh nuScenes gốc khác điều kiện sáng.

Giới hạn:
- Ảnh nuScenes mang theo nền quanh object; bóng đổ không khớp hẳn.
- Ảnh tải về chỉ có 3 mặt (góc chéo dùng mặt gần nhất) và không khớp từng chi tiết với hình dựng ray-cast.
- Xe máy có người lái trong nuScenes mini chủ yếu ở cảnh đêm / mưa. Dán vào cảnh ngày (`pasted_relaxed`) chỉ ra mảng
  tối, nhiễu, khó nhận ra xe máy. Muốn đẹp hơn cần nguồn ảnh xe máy ban ngày, ví dụ nuScenes trainval.

Chỉ dùng để minh hoạ. Mọi số đo vẫn trên LiDAR.

Demo: `python utils/camera_paste.py --target <sample_token>` lưu `output/vis/camera_paste_<token>.png`
(trái: chỉ box, phải: có dán ảnh). Web app: ô "Dán ảnh object" ở khung 6 camera.

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
