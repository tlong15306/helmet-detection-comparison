# Thư mục `outputs/faster_rcnn/`

## Mục đích

Tập hợp toàn bộ đầu ra của các thí nghiệm Faster R-CNN.

## Luồng tạo dữ liệu

1. `src/train.py` ghi log và checkpoint.
2. `src/evaluate.py` nạp checkpoint tốt nhất và ghi metric.
3. `src/infer.py` lưu prediction minh họa.
4. `tools/benchmark_speed.py` bổ sung latency/FPS theo giao thức chung.

## Điều phải kiểm tra

- Checkpoint khớp với `configs/faster_rcnn.yaml`.
- Label mapping và số lớp đúng với dataset.
- `best_map.pth` được chọn bằng validation mAP@0.5:0.95, không phải test.
- Metric test được sinh sau khi cấu hình đã chốt.

Không trộn tệp RetinaNet hoặc checkpoint từ cấu hình khác vào thư mục này.
