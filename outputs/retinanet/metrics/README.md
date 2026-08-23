# Metric RetinaNet

## Tệp dự kiến

- `val_metrics.json`.
- `test_metrics.json`.
- `speed_metrics.json`.

## Trường tối thiểu

- mAP@0.5:0.95 và mAP@0.5.
- AP theo lớp.
- Precision/Recall của `NoHelmet` tại threshold đã chốt.
- Checkpoint, split hash, số ảnh và thời gian đánh giá.

Schema phải giống metric của Faster R-CNN để `src/compare_models.py` đọc được mà không cần chuyển đổi thủ công.
