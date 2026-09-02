# Metric Faster R-CNN v6

- `test_metrics.json`: kết quả evaluator trên EdgeVision test đã chốt.
- `latency_validation.json`: benchmark 20 warm-up và 100 ảnh validation,
  batch size 1 trên RTX 2050.

Không sửa JSON bằng tay. Khi cần chạy lại, phải ghi checkpoint SHA-256 và split
hash mới vào artifact rồi tạo comparison lại cùng RetinaNet.
