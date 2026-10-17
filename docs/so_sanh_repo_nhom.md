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

---

# Nhánh `v3` của repo nhóm (commit `d44a137`, 2026-10-16)

Nhánh `v3` tách từ commit `f81df17` của repo này, thêm 1 commit: 32 file code / tài liệu và 857 ảnh PNG
(`assets/vn_objects/`, 16 lớp). Đã đọc toàn bộ diff, không chạy code.

## 1. Cảnh báo — không đưa vào báo cáo

| Hạng mục | Vấn đề |
|---|---|
| **Bảng kết quả T7** (mAP 0.448, NDS 0.551, recall bầy xe 92.4%, xe chở hàng 74.2%) trong `bao_cao_ket_qua.md`, `slides.html`, `index.html`, `ket_qua_mau/eval_report.json` | **Không có bằng chứng**: `tools/t7_train.py` chỉ chạy kiểm tra trên mô hình giả (`MockCenterPoint`), không có vòng huấn luyện, không có cấu hình `cfgs/t7_ft_frozen_replay.yaml`, không log, không checkpoint. Số NDS / mAP từng lớp chưa từng đo trong repo nào. Giống trường hợp số "AI benchmark" gõ tay ở mục 2 phía trên |
| **Ảnh vật thể VN** (ba gác, xe máy chở hàng, chó, bò, xe bò...) | Lấy từ thư mục `C:\Users\QSKGM\.gemini\antigravity\brain\...` (tên file kiểu `vietnamese_golden_dog_front_view_v2_<số>.jpg`, "ảnh studio mới sinh") -> ảnh **do AI sinh**; không ghi nguồn / giấy phép. Ảnh xe / người lấy từ nuScenes bằng **Mask R-CNN** (mạng AI tách nền) |
| **ControlNet / diffusion** (`camera_synthesis.generate_controlnet_conditions`, `run_controlnet_inpainting_hook`) | Hướng sinh ảnh bằng AI; trái với lập luận "sinh dữ liệu bằng luật vật lý" của đề tài |
| **Khoảng cách tối thiểu 9 m** cho mọi vị trí chèn (`config.MIN_DISTANCE` 3 -> 9, mọi luật vị trí kịch bản VN >= 9 m) | Lý do ghi trong code: "bảo vệ nắp capo xe ego" (cho đẹp ảnh camera). Hệ quả: **mất đúng edge case** (xe cắt đầu 3-8 m, chó lao ra 5 m, bầy xe sát xe), mọi số đo đã báo cáo không còn tái lập được |
| `_swarm_lane`: làn trái (y > 0) luôn **ngược chiều** | Đúng với giao thông bên phải (VN, Boston) nhưng **sai với Singapore** (đi bên trái) — một nửa nuScenes mini |
| Xe đạp: nới **bỏ loại scene đích** khi ít mẫu, bỏ khớp góc nhìn, khoảng cách lệch tới 25 m | Vi phạm yêu cầu "chèn vào scene khác" và luật mật độ / mặt nhìn thấy |
| `raycast.py`: thêm lốc máy / ống xả kim loại vào xe máy, đổi độ phản xạ vật liệu | Đổi dữ liệu LiDAR nhưng không đo lại (số 64% xe máy ray-cast không còn đúng); docstring ghi "micro-roughness, Lambert" nhưng chưa kiểm chứng |
| `config._find_dataroot` | Đường dẫn cá nhân cứng (`D:\VinAI\Build phase\...`) |
| `utils/io_converter.py` đọc `.bin` | Đoán số cột theo `len % 5`: file KITTI 4 cột có N chia hết cho 5 sẽ bị đọc sai thành 5 cột |

## 2. Đã lấy vào repo này (ghi nguồn trong code)

| Ý tưởng từ v3 | Cách lấy | File |
|---|---|---|
| 5 mặt ảnh (`front`, `oblique_front`, `side`, `oblique_rear`, `rear`) + nhiều biến thể mỗi mặt | Chọn mặt có góc tâm gần góc camera nhất; biến thể `<mặt>_*.png` cố định theo object (cùng object -> cùng ảnh ở mọi camera). Dùng ảnh Wikimedia có giấy phép, **không** dùng ảnh AI của v3 | `utils/camera_paste.py` |
| Xuất nhiều định dạng (`io_converter.py`) | Chỉ lấy `.ply` (CloudCompare / MeshLab), tự viết, giữ đủ 5 trường (v3 qua open3d đổi intensity thành màu, mất cột ring); web app có nút tải `.ply` | `utils/pcd_io.py`, `webapp/` |

Test: `tests/test_camera_paste.py::test_asset_view_nearest_and_variants`, `tests/test_raycast_storm.py::test_ply_roundtrip`.

## 3. Đáng làm tiếp nhưng cần chạy thật (GPU)

- **FrozenBatchNorm + đóng băng backbone khi fine-tune** (`t7_train.freeze_backbone_and_bn`): hướng đúng với nguyên nhân
  đã nghi ở mục Hạn chế (BatchNorm với batch 2). Chỉ được ghi vào báo cáo **sau khi** huấn luyện + đo thật trong WSL.
- Stage GPU trong `Dockerfile` (CUDA 11.8 + OpenPCDet): tiện để chạy detector trong container, nhưng cài OpenPCDet từ
  nhánh mới nhất không ghim phiên bản; cần ghim commit và thử build.

## 4. Repo này có, v3 chưa có (thêm sau khi v3 tách nhánh)

Kiểm tra scene hợp lệ độc lập + chèn từ scene khác + đo scene/giờ wall-clock (`evaluate.py`), mặt đất đồng thuận
3 RANSAC, lề an toàn 0.3 m, thời tiết v2 (bỏ điểm theo khoảng cách, nhiễu trên tia thật, backscatter thay điểm gốc),
kiểm chứng mặt đường ướt bằng dữ liệu thật (`tools/wet_road_check.py`), dán ảnh nuScenes gốc theo tiêu chí rõ nhất +
che khuất bằng LiDAR, ảnh Wikimedia có giấy phép + GrabCut, web: đổi seed khi sinh lại, ô "Dán ảnh object".
