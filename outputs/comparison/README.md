# Thư mục `outputs/comparison/`

## Mục đích

Chứa kết quả so sánh cuối cùng giữa Faster R-CNN và RetinaNet trên cùng test split và phần cứng.

## Tệp dự kiến

- `comparison.csv` và `comparison.json`: bảng tổng hợp metric.
- `precision_recall_curves.png`: đường cong Precision–Recall.
- `map_comparison.png`: biểu đồ mAP/AP.
- `speed_comparison.png`: latency/FPS.

## Việc cần làm

1. Đọc tự động `test_metrics.json` và `speed_metrics.json` của hai mô hình.
2. Kiểm tra checkpoint, split hash và giao thức đánh giá khớp nhau.
3. Ghi đơn vị, số chữ số thập phân và chiều tốt hơn của từng metric.
4. Phân tích cả độ chính xác, Recall `NoHelmet`, tốc độ và tài nguyên.

Không tạo bảng so sánh bằng cách nhập số liệu thủ công hoặc trộn validation với test.
