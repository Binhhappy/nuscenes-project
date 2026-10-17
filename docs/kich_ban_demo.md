# Kịch bản trình bày + demo web

Mạch: **Vấn đề → Giải pháp → Kết quả → Demo trực tiếp**. Tổng khoảng 12 phút: slide 6 phút, demo 6 phút.

- Slide: `docs/demo/trinh_bay.html`. Mở bằng Chrome, nhấn `F` để toàn màn hình.
  Điều khiển: mũi tên ← →, `Home`, `End`. Ở slide 5 và 6, 3D xoay được bằng chuột.
- Web: `python run_app.py` (cổng mặc định 8050), hoặc đặt `$env:PORT=8060` rồi chạy.

Chia vai: **Người nói (A)** trình bày slide và demo. **Người bấm (B)** chỉ thao tác, không nói.
**Người C** trả lời câu hỏi.

---

## Chuẩn bị (trước giờ trình bày 15 phút)

1. Bật web app. Bấm **Sinh kịch bản** 1 lần để nạp sẵn dữ liệu (lần đầu chậm vài giây).
2. Mở 2 tab Chrome: tab 1 là slide, tab 2 là web app. Phóng to web khoảng 110% cho dễ nhìn.
3. Kiểm tra ô **Giữ seed** đang **tắt**, ô **Dán ảnh object** đang **bật**.
4. Dự phòng khi web lỗi: video `docs/demo/assets/videos/scene-0103_swarm_pretrained.mp4` và trang offline
   `docs/demo/index.html`.
5. **Không bấm "Chạy detector"**: nút này cần GPU. Kết quả detector đã có ở tab **Kết quả đo**.

---

## Phần 1: Slide (6 phút)

| Slide | Nói (1–2 câu) | Điểm nhấn |
|---|---|---|
| 1. Bìa | "Nhóm em làm công cụ sinh dữ liệu LiDAR giao thông Việt Nam, từ chính dữ liệu thật." | Để 3D bầy xe máy tự xoay 3 giây rồi mới nói |
| 2. Vấn đề | "Bộ dữ liệu nổi tiếng nhất có **0%** cảnh có bầy xe máy. Detector chỉ tìm ra **55%** xe máy, trong khi ô tô là 93%." | Chỉ vào số 0% |
| 3. Giải pháp | "Không cần đi thu thập thêm. Hệ thống sinh dữ liệu mới qua 5 bước, có sẵn nhãn." | Đọc tên 5 bước theo mũi tên |
| 4. Công nghệ | "Mỗi điểm chèn vào đều theo 4 luật vật lý của cảm biến: che khuất, mật độ, mặt nhìn thấy, mặt đất." | "Theo luật vật lý", không phải vẽ tay |
| 5. 3D | Bấm **Ẩn vật chèn** → **Có vật chèn**: "Thấy các 'bóng' trống phía sau mỗi xe không? Đó là vùng bị che, đúng như LiDAR thật." | Xoay 3D một vòng |
| 6. Thời tiết | Bấm lần lượt 4 nút: "Cùng một cảnh, bốn kiểu trời. Tia đi càng xa càng dễ mất, theo định luật Beer–Lambert." | Dừng ở **Giông bão** |
| 7. Vật thể VN | "Ba gác, xe chở 3, hàng rong, bò, chó: những thứ không bộ dữ liệu nào có." | Lướt 6 ảnh |
| 8. Camera | Kéo thanh trượt từ trái sang phải: "LiDAR và ảnh khớp nhau từng điểm." | Kéo chậm |
| 9. Chất lượng | "**100%** cảnh hợp lệ, **9 400** cảnh mỗi giờ trên một laptop." | 2 số lớn |
| 10. So sánh | "So với cách dán thẳng thông thường: mật độ đúng như thật, 100% chạm đất." | Chờ thanh chạy xong |
| 11. Hiệu quả | "Dữ liệu sinh ra chỉ ra điểm mù: xe máy trong bầy giảm 12 điểm, xe chở hàng cồng kềnh chỉ còn 29%. Detector nhìn ô tô chèn giống hệt ô tô thật: 93% bằng 93%." | Câu chốt: "tìm điểm mù **trước khi** xe lăn bánh" |
| 12. Sản phẩm | "Tất cả gói trong một ứng dụng web. Mời thầy cô xem trực tiếp." | Bấm **Demo trực tiếp →** hoặc chuyển sang tab 2 |

Slide 13 (kết) để dành sau phần demo.

---

## Phần 2: Demo web (6 phút)

### Màn A. "Một cú bấm, một bộ dữ liệu" (1.5 phút)
| Thao tác (B) | Lời nói (A) |
|---|---|
| Tab **Sinh kịch bản**. Mục 1 giữ `scene-0103 · khung 10`. Mục 2 chọn **Kịch bản có sẵn** → `moto_swarm`. Bấm **Sinh kịch bản** | "Đây là một khung LiDAR thật ở Boston. Em chọn kịch bản bầy xe máy và bấm Sinh." |
| Chỉ khung thống kê bên trái: thời gian (ms), **Object chèn**, **Điểm LiDAR** | "Chưa tới một giây: khoảng 18 xe máy được chèn, mỗi xe có sẵn nhãn 3D." |
| Ở 3D chọn **Làm nổi object chèn** | "Các điểm tím là xe được chèn. Xe nào cũng nằm trên lòng đường, không xe nào chồng lên xe nào." |
| Chỉ dòng **Vị trí bị loại** | "Hệ thống đã tự loại hàng trăm vị trí sai: ngoài đường, va chạm. Chỉ giữ vị trí hợp lệ." |

### Màn B. "Đúng như cảm biến thật" (1 phút)
| Thao tác | Lời nói |
|---|---|
| Hiển thị **Gốc** → **Sau khi chèn** → **Cả hai** | "Trước và sau khi chèn. Ở chế độ 'Cả hai', phần nền phía sau mỗi xe biến mất: xe đã che tia laser." |
| Góc nhìn **Góc lái xe**, xoay nhẹ | "Đây là góc nhìn từ xe tự lái. Xe ở xa thì thưa điểm, xe ở gần thì dày, đúng tỉ lệ khoảng cách." |

### Màn C. "Thời tiết" (1 phút)
| Thao tác | Lời nói |
|---|---|
| Mục 3 chọn **Mưa** / **to**, bấm **Sinh kịch bản** | "Cùng cảnh đó, thêm mưa to." |
| Chỉ dòng **Điểm LiDAR … → …** | "Mất khoảng 30% số điểm, chủ yếu ở xa. Gần xe xuất hiện nhiễu từ giọt mưa." |
| Đổi sang **Giông / bão**, bấm Sinh. Mở khung **6 camera** | "Giông bão: mặt đường ướt, nước bắn sau bánh xe. Ảnh camera cũng phủ mưa tương ứng." |

### Màn D. "Đường phố Việt Nam" (1.5 phút)
| Thao tác | Lời nói |
|---|---|
| Mục 3 về **Trời trong**. Mục 2 chọn **Tự chọn vật thể** → preset **Giờ cao điểm VN**, bấm Sinh | "Kịch bản tự chọn: bầy xe máy, xe chở 3 người, xe chở hàng, ba gác. Trộn bao nhiêu cũng được." |
| Đổi preset **Đường quê** (bò, xe bò, chó), bấm Sinh. Cuộn xuống **6 camera** | "Những vật thể này không có trong bộ dữ liệu gốc. Hệ thống dựng hình theo kích thước thật rồi mô phỏng bắn đúng 32 tia laser." |
| Ở khung camera bật / tắt **Dán ảnh object** | "Ảnh camera được đồng bộ theo đúng vị trí và góc nhìn của từng vật thể." |
| Bấm **Sinh kịch bản** thêm 1 lần | "Mỗi lần bấm là một cảnh mới: hệ thống tự đổi seed." |

### Màn E. "Dùng được ngay" (0.5 phút)
| Thao tác | Lời nói |
|---|---|
| Chỉ các nút **.pcd**, **.ply**, **.bin nuScenes**, **labels.json**, **bundle .zip** | "Dữ liệu tải về ngay, đúng định dạng nuScenes, mở được bằng CVAT hay CloudCompare để huấn luyện." |

### Màn F. Chốt bằng số đo (0.5 phút)
| Thao tác | Lời nói |
|---|---|
| Tab **Kết quả đo** | "Toàn bộ số đo nhóm trình bày đều đo thật trên detector CenterPoint, có khoảng tin cậy 95%." |
| Quay lại slide, nhấn `End` (slide 13) | "Dữ liệu cho xe tự lái hiểu đường phố Việt Nam. Em cảm ơn thầy cô." |

---

## Câu trả lời ngắn cho câu hỏi hay gặp

| Câu hỏi | Trả lời (1–2 câu) |
|---|---|
| Có dùng AI để sinh không? | "Không. Sinh dữ liệu bằng thuật toán theo luật vật lý của cảm biến, chạy lại cùng seed ra đúng kết quả. Detector chỉ là thước đo." |
| Làm sao biết cảnh hợp lệ? | "Mỗi cảnh được kiểm tra lại độc lập sau khi sinh: bản đồ, va chạm, mặt đất. Đạt 100% trên 100 cảnh." |
| Dữ liệu sinh có giống thật không? | "Mật độ điểm bằng 1.01 lần vật thật cùng khoảng cách. Detector nhận ô tô chèn 93%, bằng ô tô thật." |
| Thời tiết dựa trên gì? | "Định luật suy giảm Beer–Lambert. Đã đối chiếu với cảnh đường ướt thật của nuScenes." |
| Mở rộng được không? | "Pipeline không phụ thuộc kích thước dữ liệu. Chạy trên nuScenes đầy đủ chỉ cần đổi đường dẫn." |
| Ảnh camera lấy ở đâu? | "Cắt từ chính ảnh gốc của vật thể trong nuScenes. Vật thể Việt Nam dùng ảnh có giấy phép tự do (Wikimedia), ghi nguồn đầy đủ." |
