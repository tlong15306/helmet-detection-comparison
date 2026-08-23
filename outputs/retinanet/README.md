# Thư mục `outputs/retinanet/`

## Mục đích

Tập hợp toàn bộ đầu ra của các thí nghiệm RetinaNet theo cấu trúc tương ứng Faster R-CNN.

## Luồng tạo dữ liệu

1. `src/train.py` ghi log và checkpoint.
2. `src/evaluate.py` đánh giá checkpoint tốt nhất.
3. `src/infer.py` lưu prediction minh họa.
4. `tools/benchmark_speed.py` đo latency/FPS.

## Điều phải kiểm tra

- Checkpoint khớp với `configs/retinanet.yaml`.
- Classification head có số lớp và label mapping chính xác.
- Focal Loss được dùng bởi kiến trúc RetinaNet; không gọi training loss là metric.
- Test chỉ chạy sau khi chốt cấu hình.

Không trộn tệp Faster R-CNN hoặc kết quả từ split khác vào thư mục này.
