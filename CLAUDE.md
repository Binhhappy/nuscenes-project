@AGENTS.md

# Riêng cho Claude Code

Vai trò: lập plan, chia task cho Codex / Gemini, code lõi, review.

## Giao task cho agent khác

Viết prompt theo mẫu, mỗi task một branch:

```
Mục tiêu: <một câu>
Bối cảnh: đọc AGENTS.md; code liên quan: <file:hàm>
Phạm vi: CHỈ sửa/tạo <file>. Không đụng <file/thư mục>.
Yêu cầu: <các case cụ thể>
Xong khi: <lệnh kiểm tra, ví dụ `pytest tests/test_x.py -q` pass>
Đầu ra: theo mục "Báo cáo khi xong task" trong AGENTS.md
```

## Review kết quả agent khác

- Kiểm tra diff chỉ nằm trong phạm vi đã giao.
- Chạy lại lệnh "Xong khi" để xác nhận, không tin báo cáo.
- Ảnh mới: có dòng trong `assets/vn_objects/SOURCES.md`, đúng hướng (đầu quay sang phải), có alpha.
- Đổi `config.py` hoặc thứ tự random: đối chiếu số liệu trong `docs/bao_cao_ket_qua.md`.
