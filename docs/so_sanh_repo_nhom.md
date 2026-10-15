# So sánh với repo nhóm `tranducquan2k5-afk/buildPhasev1.2` và câu hỏi về quy mô sinh dữ liệu

Repo nhóm (commit `459d296`, 2026-10-14) tách từ phiên bản cũ của repo này (trước T2-T7), viết lại phần lõi và thêm
web app. Đã đọc toàn bộ code (không chạy).

## 1. Khác biệt chính

| Hạng mục | Repo nhóm | Repo này | Đánh giá |
|---|---|---|---|
| Mục tiêu | "SaaS" tăng cường dữ liệu, nhấn mạnh thời tiết | Edge case giao thông VN + chứng minh model yếu | khác trọng tâm |
| Đặt object | cách nhau **≥ 15 m** và lệch **≥ 40°** phương vị, buffer 1.5 m, hướng theo làn ước lượng từ xe xung quanh | luật theo kịch bản (bầy xe sát nhau, cắt đầu, ngược chiều...) | ràng buộc 15 m / 40° **loại bỏ đúng edge case VN** (bầy xe máy, chen sát) — không lấy; ý tưởng hướng theo làn đáng tham khảo |
| Lọc template | chỉ template đặc (xe lớn ≥ 400 điểm, xe con ≥ 180) | mọi template ≥ 20 điểm, chỉnh mật độ theo khoảng cách | lọc đặc làm object chèn dày điểm hơn thực tế ở cùng khoảng cách -> dễ phát hiện giả tạo; không lấy |
| Vật thể VN | không có (chỉ lớp nuScenes) | xe máy chở người / hàng, ba gác, xe hàng rong, chó, bò, xe bò (ray-cast HDL-32E) | repo này đầy đủ hơn |
| Nhiều khung / sweep | không | 10 sweep + quỹ đạo nhiều khung | repo này |
| Thời tiết LiDAR | mưa Beer-Lambert, sương, **vũng nước** | mưa, sương, giông bão (mất theo khoảng cách, mặt đường ướt, spray) | **lấy vũng nước** (tuỳ chọn trong `task_b/storm.py`) |
| Thời tiết trên ảnh camera | **vệt mưa + giọt nước ống kính, sương Koschmieder** | không | **lấy** -> `task_b/camera_weather.py` (sửa độ sâu trên chân trời 220 m -> 60 m để không trắng xoá) |
| 6 camera + box chiếu | có (web app) | có (`tools/case_gallery.py`, trang demo kiểu CVAT) | tương đương |
| Định dạng vào / ra | .pcd .ply .las .laz .xyz .npy (`utils/io_converter.py`) | .bin nuScenes | chưa lấy (cần open3d / laspy; môi trường Windows hiện lỗi DLL open3d) — nên lấy nếu cần xuất sang CVAT |
| Giao diện | FastAPI + Three.js, upload point cloud, Docker | trang HTML tĩnh offline + slide | web app có thể ghép sau (chỉ cần đọc output của repo này) |
| Detector / train | **không có code** | OpenPCDet CenterPoint-PointPillar thật, đo recall / CI 95% | xem mục 2 |

## 2. Cảnh báo: số "AI benchmark" trong repo nhóm là số gõ tay

`scripts/run_ai_benchmark.py` **ghi cứng** các giá trị (`baseline_map: 35.2`, `ours_map: 58.6`, `gain_map: 23.4`,
"-41.2% báo động giả"...) rồi in ra và ghi vào `output/eval_report.json`; script chỉ đếm số file đã sinh, không chạy
detector, không train. Repo không có code / checkpoint PointPillars hay CenterPoint. README trình bày các số này như
kết quả thực nghiệm. **Không đưa các số này vào báo cáo**; nếu bị hỏi nguồn sẽ không chứng minh được.
Số đo thật của đề tài (detector chạy thật trên mini_val) ở `docs/bao_cao_ket_qua.md`.

## 3. Đã tích hợp vào repo này

- `task_b/camera_weather.py`: mưa / sương / giông trên ảnh camera (gốc từ repo nhóm, có ghi nguồn); dùng cho 6 ảnh
  camera trong tab Thời tiết của `docs/demo/index.html`.
- `task_b/storm.py`: `puddle_mask` + tham số `puddles` (mặc định 0 để các số giông bão đã đo vẫn tái tạo đúng).
- Test: `tests/test_raycast_storm.py` (53/53 test đạt).

## 4. Sinh 100 object / khung có được không?

Thử trên mini_val (5 khung, xe máy cắt mẫu + ray-cast trong bán kính 40 m, tối đa 3000 lần thử):

| Yêu cầu | Đặt được | Điểm LiDAR / xe (median) | Xe < 5 điểm | Giây / khung (keyframe) |
|---|---|---|---|---|
| 25 | 25 (100%) | 25 | 7% | 0.8 |
| 50 | 50 (100%) | 16 | 14% | 1.3 |
| 100 | 91 (91%) | 13 | 20% | 2.5 |
| 150 | 118 (79%) | 10 | 31% | 3.6 |

- **Được về kỹ thuật**, nhưng bão hoà ~100-120 xe máy / khung (hết chỗ trống trên lòng đường, hết template hợp góc nhìn).
- Càng nhiều object, chúng **che lẫn nhau**: 20% xe gần như vô hình (< 5 điểm) khi yêu cầu 100. Nhãn của object
  không còn thấy (0 điểm) nên bỏ; 1-4 điểm nên đánh dấu "khó" như nuScenes (num_lidar_pts) thay vì tính như object rõ.
- Nhiều loại object khác nhau: `task_a/vn_scenarios.py` cho mỗi kịch bản 1 danh sách nhóm (loại, số lượng min-max,
  luật vị trí, tốc độ); trộn được (vd giờ cao điểm: xe máy 1 người + 2-3 người + chở hàng + ba gác). Giới hạn: loại cắt
  mẫu bị chặn bởi số template (mini chỉ 6 xe máy có người lái), loại ray-cast không giới hạn số lượng nhưng độ giống
  thật phụ thuộc hình dựng (ô tô đạt, xe máy chưa).

## 5. Không phải cảnh nào cũng sinh được mọi edge case

Mỗi kịch bản cần điều kiện bản đồ / hình học: lòng đường đủ rộng, có vỉa hè (`walkway`), vạch sang đường, mặt đất
thấy được, không vướng xe thật. Đo trên mini_val (81 khung, 2 scene đô thị):

- đa số kịch bản đặt được ở 100% khung; `control_ped` (người trên vỉa hè) chỉ 75% (khung không có walkway gần);
  `ped_group_cross` 98%; bầy xe máy đạt 88% số xe yêu cầu.
- Ở dữ liệu lớn hơn (bãi đỗ xe, đường hẹp, đường cao tốc) tỉ lệ sẽ thấp hơn cho kịch bản cần vỉa hè / chợ / làn ngược.
- Cách xử lý khi sinh hàng loạt: **chọn kịch bản theo cảnh** (kiểm tra trước lớp bản đồ, độ rộng lòng đường, mật độ xe
  thật) thay vì ép mọi kịch bản vào mọi cảnh; log `reject_stats` (đã có) cho biết vì sao không đặt được.

## 6. Chi phí và chất lượng khi sinh 100k khung

Số đo trên máy hiện tại (laptop, giới hạn 3 luồng CPU):

| Hạng mục | Số đo | Ước 100k khung |
|---|---|---|
| Dung lượng | 6.8 MB / khung (keyframe 0.7 MB + 9 sweep) | **~680 GB** (chỉ keyframe: ~70 GB) |
| Thời gian sinh | 0.2-0.7 s / keyframe; 1-4.5 s / khung đủ 10 sweep | ~2.5 s x 100k ≈ **70 giờ** ở 3 luồng; ~15 giờ nếu dùng hết 16 lõi; vài giờ trên máy chủ nhiều lõi |
| Chạy detector để kiểm tra | ~0.3 s / khung (RTX 3060) | ~8 giờ GPU |

Chất lượng có đảm bảo không — **số lượng được, chất lượng có điều kiện**:

1. **Đa dạng nền cảnh là nút thắt**: mini chỉ 404 khung -> 100k khung nghĩa là mỗi cảnh nền dùng lại ~250 lần, model sẽ
   học thuộc nền. Cần nuScenes trainval (34k keyframe, 850 scene, ~300 GB LiDAR) hoặc dữ liệu nền khác (lý tưởng: LiDAR
   thu ở VN).
2. **Đa dạng vật thể**: 6 xe máy có người lái trong mini là quá ít (bầy 15 xe lặp hình dạng). Trainval có nhiều template
   hơn; ray-cast cần mesh 3D thật để giống thật (xe máy dựng khối hiện chỉ 64% so với 94%).
3. **Kiểm soát chất lượng tự động** (đã có, chạy được ở quy mô lớn): mỗi object ghi mặt nhìn thấy, mật độ so với
   khoảng cách, mặt đất, số điểm còn thấy, va chạm -> lọc tự động; mỗi lô chạy **nhóm đối chứng** (object chèn ở vị
   trí bình thường phải được detector thấy như object thật) -> phát hiện lô hỏng (như lỗi bảng góc tia đã gặp).
4. **Kiểm tra bằng mắt theo mẫu**: không thể xem 100k khung; lấy ngẫu nhiên ~0.5-1% (500-1000 khung) xem trên trang demo
   kiểu CVAT.
5. Tham số thời tiết và kích thước vật thể giả định cần hiệu chỉnh bằng số đo thật trước khi dùng để train hàng loạt.
