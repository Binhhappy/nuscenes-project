# Ảnh 2D cho vật thể chèn (dán lên ảnh camera)

`n3-08c/utils/camera_paste.py` dùng ảnh ở đây cho các vật thể:
- không có trong nuScenes (vật thể VN ray-cast);
- hoặc có trong nuScenes nhưng ảnh gốc không đạt tiêu chí (ví dụ xe máy chỉ có ảnh ban đêm).

Ảnh chỉ để **minh hoạ trên camera**. Điểm LiDAR và mọi số đo không dùng ảnh này.

## Cấu trúc

```
assets/vn_objects/<loại>/front.png          camera nhìn thấy ĐẦU vật thể (góc 0°)
assets/vn_objects/<loại>/oblique_front.png  chéo phía đầu (45°), đầu quay sang PHẢI
assets/vn_objects/<loại>/side.png           nhìn NGANG (90°), đầu vật thể quay sang PHẢI (code tự lật khi cần)
assets/vn_objects/<loại>/oblique_rear.png   chéo phía đuôi (135°), đầu quay sang PHẢI
assets/vn_objects/<loại>/rear.png           camera nhìn thấy ĐUÔI vật thể (180°)
assets/vn_objects/<loại>/side_2.png ...     biến thể thêm của cùng mặt (<mặt>_<tên>.png), mỗi object cố định 1 biến thể
```

Không bắt buộc đủ 5 mặt: code chọn mặt có góc gần góc camera nhìn vật thể nhất trong số mặt đang có.

Tên `<loại>` lấy từ category, là phần sau dấu chấm đầu tiên:

| Thư mục | Vật thể |
|---|---|
| `ba_gac` | xe ba gác |
| `moto_cargo` | xe máy chở hàng cồng kềnh |
| `moto_multi` | xe máy chở 2-3 người |
| `moto_tow` | xe máy kéo thùng |
| `vendor_cart` | xe hàng rong |
| `dog` | chó |
| `cow` | trâu / bò |
| `ox_wagon` | xe bò (xe gỗ, phần con bò dùng `cow`) |
| `motorcycle` | xe máy có người lái (thay ảnh đêm của nuScenes mini) |
| `car`, `truck`, `pedestrian`, ... | dự phòng khi ảnh nuScenes gốc không đạt |

## Yêu cầu ảnh

- PNG **có kênh alpha** (nền trong suốt). Ảnh nền trắng sẽ bị bỏ qua. Tách nền bằng remove.bg hoặc `rembg`.
- Vật thể chụp gần thẳng mặt (trước / sau / ngang), cả thân nằm trong ảnh, ít bị che.
- Phần trong suốt thừa quanh vật thể không sao: code tự cắt sát.
- Ánh sáng ban ngày. Code có chỉnh màu theo nền, nhưng ảnh đêm vẫn trông lạc.

## Nguồn ảnh

Chỉ dùng ảnh tự chụp hoặc có giấy phép cho phép dùng lại (Wikimedia Commons, Pexels, Unsplash...).
Ghi mỗi file vào `SOURCES.md` theo mẫu: `ba_gac/side.png | <link gốc> | <tác giả> | <giấy phép>`.

## Xem kết quả

```powershell
cd n3-08c
python utils/camera_paste.py --target b6b0d9f2f2e14a3aaa2c8aedeb1edb69 --vn-scenario ba_gac_slow
# -> output/vis/camera_paste_<token>_ba_gac_slow.png (trái: chỉ box, phải: có dán ảnh)
```
Các kịch bản khác: `cargo_moto`, `multi_rider`, `tow_moto`, `vendor_curb`, `dog_dart`, `cow_road`, `ox_cart_road`,
`rush_hour`, `market_street`, `animal_in_traffic`. Web app: ô "Dán ảnh object" ở khung 6 camera.
