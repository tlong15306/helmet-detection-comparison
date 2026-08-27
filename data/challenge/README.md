# Challenge Set v1 — Ảnh giao thông khó

Thư mục này dùng để kiểm tra khả năng tổng quát của Faster R-CNN và RetinaNet trên ảnh giao thông khó, đặc biệt các tình huống đông xe máy, nhiều loại mũ, thiếu sáng, che khuất và vật thể nhỏ.

## Phạm vi

- Mục tiêu: khoảng 50 ảnh chính thức sau khi lọc từ 80–100 ảnh ứng viên.
- Không thay thế hoặc sửa các split EdgeVision tại `data/splits/`.
- Không dùng Challenge Set v1 để chọn epoch, confidence threshold hay bất cứ hyperparameter nào.
- Mọi ảnh được đánh giá bằng cùng checkpoint/evaluator cho cả Faster R-CNN và RetinaNet.

## Quy tắc lưu trữ

- `raw/`: tệp tải nguyên gốc; không commit.
- `candidates/`: ảnh/frame ứng viên; không commit.
- `selected/`: ảnh đã được Long duyệt để gán nhãn; không commit.
- `annotations/`: COCO JSON, policy nhãn và kết quả review; được phép commit sau khi duyệt.
- `metadata/`: manifest nguồn, nhóm nguồn, thẻ độ khó và hash; được phép commit.
- `previews/`: contact sheet/ảnh render cục bộ; không commit.

Không đặt ảnh/video có nguồn hoặc giấy phép không rõ ràng vào bất cứ thư mục nào tại đây. Mọi tệp phải có một dòng trong `metadata/sources_manifest.csv` trước khi được chọn.

## Trạng thái hiện tại

- Pha A: thu thập và đánh giá challenge set độc lập.
- Chưa có ảnh chính thức, chưa có annotation và chưa train lại.
- Kế hoạch đầy đủ: [`KE_HOACH_BO_SUNG_TAP_ANH_KHO.md`](../../KE_HOACH_BO_SUNG_TAP_ANH_KHO.md).
