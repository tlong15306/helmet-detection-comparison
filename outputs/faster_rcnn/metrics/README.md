# Metric Faster R-CNN

## Tệp dự kiến

- `val_metrics.json`: metric của checkpoint tốt nhất trên validation.
- `test_metrics.json`: metric cuối cùng trên test.
- `speed_metrics.json`: latency, FPS và giao thức benchmark.

## Trường tối thiểu

- mAP@0.5:0.95 và mAP@0.5.
- AP theo lớp.
- Precision/Recall của `NoHelmet` tại threshold đã chọn trên validation.
- Số ảnh, checkpoint, split hash và thời gian đánh giá.

Không chỉnh các giá trị trong JSON bằng tay; nếu phát hiện lỗi phải chạy lại evaluator.
