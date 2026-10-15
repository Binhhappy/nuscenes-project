# Danh mục vật thể và kịch bản giao thông Việt Nam (bản nháp để nhóm góp ý)

Nguyên tắc: mỗi vật thể / kịch bản phải có thật trong giao thông Việt Nam và thiếu (hoặc gần như không có) trong
nuScenes (Boston / Singapore). Dữ liệu giả lập dùng để sinh edge case, không thay dữ liệu thật.

## 1. Vật thể

Nguồn điểm LiDAR:
- **cắt mẫu**: lấy điểm của object thật trong nuScenes (ô tô, xe tải, xe máy có người lái, người đi bộ)
- **ray-cast**: dựng hình từ khối cơ bản theo kích thước công bố, bắn tia theo đúng LiDAR nuScenes (Velodyne HDL-32E:
  32 tia dọc -30.67°..+10.67°, ~1084 điểm / vòng / tia). Mật độ, mặt nhìn thấy, che khuất tự đúng theo cảm biến.

| Mã | Vật thể | Kích thước dài x rộng x cao (m) | Nguồn kích thước | Nguồn điểm | Lớp khi đo |
|---|---|---|---|---|---|
| car | Ô tô 4-5 chỗ | ~4.4 x 1.7 x 1.5 (vd Toyota Vios 4.425 x 1.730 x 1.475) | hãng công bố | cắt mẫu nuScenes | car |
| moto_1 | Xe máy 1 người | xe 1.9 x 0.7 x 1.1 (vd Honda Wave Alpha 1.914 x 0.688 x 1.075); có người ~1.6 | hãng công bố | cắt mẫu nuScenes | motorcycle |
| moto_multi | Xe máy chở 2-3 người | 1.9 x 0.7 x 1.6-1.7, người ngồi nối tiếp | hãng + giả định | ray-cast | motorcycle |
| moto_cargo | Xe máy chở hàng cồng kềnh | xe 1.9 x 0.7 + khối hàng rộng 1.0-1.8, cao tới 2.0 | giả định | ray-cast | motorcycle (+ vật cản) |
| ba_gac | Xe ba gác | ~ ô tô 4 chỗ: 4.0 x 1.7 x 1.5 (thùng hở, người lái phía trước) | giả định của nhóm | ray-cast | vật cản (không có lớp) |
| moto_tow | Xe máy kéo hàng | xe máy + thùng kéo 1.8 x 1.2 x 1.0 | giả định | ray-cast | vật cản |
| vendor_cart | Xe đẩy bán hàng rong | 1.5 x 0.8 x 1.2 + người đẩy | giả định | ray-cast | vật cản |
| dog | Chó | 0.9 x 0.3 x 0.6 | giả định | ray-cast | vật cản |
| cow | Bò / trâu | 2.2 x 0.7 x 1.4 | giả định | ray-cast | vật cản |
| ox_cart | Xe bò | bò 2.2 x 0.7 x 1.4 + xe gỗ 2.5 x 1.5 x 1.2 (2 box riêng như truck + trailer) | giả định | ray-cast | vật cản |

Quy ước nhãn (theo nuScenes): box bao toàn bộ vật thể kể cả người ngồi / hàng chở; hướng = hướng đầu xe / đầu con vật;
vật kéo (thùng, xe gỗ) là box riêng. Detector nuScenes chỉ có 10 lớp nên vật thể không có lớp được đo bằng
**recall vật cản không phân lớp** (có box bất kỳ đè lên tâm trong 2 m).

## 2. Kịch bản

Đã có (đo ở T6):

| Mã | Mô tả | Kết quả detector pretrained |
|---|---|---|
| moto_swarm | 10-20 xe máy quanh xe, cùng chiều | **yếu**: 80% (đối chứng 94%), nhầm thành ô tô |
| cut_in | xe máy cắt đầu 3-8 m | tốt (100%) |
| wrong_way | xe ngược chiều trong làn | tốt (98%) |
| jaywalk | người băng ngang ngoài vạch | tốt (100%) |
| lane_block | ô tô / xe tải đỗ lấn làn | tốt (95%) |

Mới, theo từng vật thể:

| Mã | Mô tả |
|---|---|
| multi_rider | xe máy chở 2-3 người trong dòng xe |
| cargo_moto | xe máy chở hàng cồng kềnh, chạy chậm cạnh / trước xe |
| ba_gac_slow | xe ba gác chạy chậm 5-15 km/h sát lề phải / trong làn trước xe |
| tow_moto | xe máy kéo thùng hàng |
| vendor_curb | xe đẩy hàng rong đứng / đi sát lề, lấn xuống lòng đường |
| dog_dart | chó lao ngang trước xe 5-20 m, 3-6 m/s |
| cow_road | bò / trâu đứng hoặc đi chậm trên lòng đường |
| ox_cart_road | xe bò đi chậm trong làn (vật cản bất thường; bản đồ nuScenes là đô thị) |
| ped_group_cross | nhóm 3-8 người băng ngang cùng lúc, đứng sát nhau (trong 3 x 5 m) |
| ped_crowd | đám đông 10-20 người đi lại dưới lòng đường quanh xe (chợ, giờ tan trường) |

Mới, kết hợp (mix):

| Mã | Mô tả |
|---|---|
| rush_hour | bầy xe máy (1 người, 2-3 người, chở hàng) + 1 xe ba gác |
| market_street | xe hàng rong + xe máy đỗ lấn + người đi bộ |
| animal_in_traffic | chó lao ra giữa bầy xe máy |

Thời tiết (Task B): mưa, sương (đã có); **giông / bão** = mưa rất to + bụi nước bắn sau bánh xe + mặt đường ướt
(mất điểm mặt đường ở xa). Kết hợp được với mọi kịch bản trên.

## 3. Cần nhóm xác nhận

- Kích thước giả định ở bảng 1 (ba gác, hàng cồng kềnh, xe đẩy, động vật).
- Thêm / bớt kịch bản; tham số (khoảng cách, tốc độ, số lượng) ghi trong code `task_a/scenarios.py`.
