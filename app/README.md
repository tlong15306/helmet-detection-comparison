# Thư mục `app/`

## Mục đích

Xây dựng ứng dụng demo cho phép chọn Faster R-CNN hoặc RetinaNet và chạy suy luận trên ảnh, video hoặc camera.

## Thành phần

- `app.py`: điểm vào giao diện và luồng tương tác người dùng.
- `model_loader.py`: đọc config, label mapping và checkpoint; tạo đúng kiến trúc trước khi nạp weights.
- `assets/`: tài sản giao diện như icon, ảnh hướng dẫn hoặc CSS nếu cần.

## Việc cần làm

1. Chọn công nghệ giao diện sau khi pipeline inference ổn định.
2. Cho phép chọn mô hình và nguồn đầu vào.
3. Hiển thị bounding box, tên lớp, confidence, latency và FPS nếu có.
4. Có điều khiển confidence threshold nhưng ghi rõ giá trị đang dùng.
5. Xử lý lỗi tệp, camera không mở được hoặc checkpoint không tương thích.
6. Chụp màn hình các trường hợp đúng, bỏ sót và phát hiện nhầm cho báo cáo.

## Điều kiện trước khi triển khai

- Có checkpoint tốt nhất đã được đánh giá của cả hai mô hình.
- `src/infer.py` đã hoạt động ổn định.
- Label mapping và cách tiền xử lý giống lúc đánh giá.

Không sao chép weights vào thư mục này; ứng dụng phải đọc từ `outputs/<model>/checkpoints/`.
