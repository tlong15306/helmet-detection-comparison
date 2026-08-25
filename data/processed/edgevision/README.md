# Annotation EdgeVision đã xử lý

Thư mục này chỉ chứa bản sao dữ liệu được làm sạch để tạo split và huấn luyện; không thay thế hoặc chỉnh sửa `data/raw/edgevision/`.

Các tệp sinh cục bộ dự kiến:

- `annotations.json`: COCO JSON đã qua kiểm tra.
- `annotation_changes.json`: mọi annotation/ảnh bị clip hoặc loại, kèm lý do.
- `image_hashes.json`: manifest SHA-256 và group ID.

Ảnh có EXIF Orientation được loader xoay về đúng chiều bằng `ImageOps.exif_transpose`; kích thước trong `annotations.json` phải theo cùng chiều hiển thị này.

Chỉ dùng phiên bản có hash đã được Long duyệt để tạo `data/splits/`.
