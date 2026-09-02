# Thư mục `app/`

## Mục đích

Ứng dụng demo cho phép chọn Faster R-CNN hoặc RetinaNet phiên bản fine-tune
Việt Nam v6 và chạy suy luận trên ảnh, video hoặc camera.

## Thành phần

- `model_loader.py`: đọc config, label mapping và checkpoint; tạo đúng kiến trúc trước khi nạp weights.
- `api.py`: nhận ảnh/video, trả detection thô và detection hậu xử lý cho giao diện.
- `video_jobs.py`: áp dụng cùng hậu xử lý cho từng frame video.
- `assets/`: tài sản giao diện như icon, ảnh hướng dẫn hoặc CSS nếu cần.

## Checkpoint demo đang triển khai

Giao diện đang dùng hai checkpoint trong
`outputs/vietnam_pilot_v6_wikimedia/`. Checkpoint baseline EdgeVision và
candidate train gộp vẫn được giữ cục bộ để đối chiếu/rollback, nhưng không xuất
hiện trên giao diện.

Hai checkpoint v6 được đưa vào giao diện để kiểm tra trực quan theo yêu cầu;
chúng không thay thế bảng baseline học thuật nếu chưa qua deployment gate.

## Điều kiện trước khi triển khai

- Có checkpoint tốt nhất đã được đánh giá của cả hai mô hình.
- `src/infer.py` đã hoạt động ổn định.
- Label mapping và cách tiền xử lý giống lúc đánh giá.

Không sao chép weights vào thư mục này; ứng dụng phải đọc từ `outputs/<model>/checkpoints/`.

## Giới hạn của demo

- Hậu xử lý chỉ thay đổi cách diễn giải/hiển thị suy luận, không làm thay đổi checkpoint, `test_metrics.json` hoặc mAP trong báo cáo.
- Cảnh báo `driver_no_helmet` chỉ được tạo khi quy tắc ghép đầu–xe đủ điều kiện.
  Trường hợp xung đột/mơ hồ bị ẩn khỏi giao diện và không tạo cảnh báo.
- Ngưỡng theo lớp chỉ điều khiển suy luận/demo. Chúng không được dùng để thay đổi
  `test_metrics.json` hay để suy ra lại mAP đã công bố.
