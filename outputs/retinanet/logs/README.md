# Log RetinaNet

## Tệp dự kiến

- `history.csv`: classification loss, box regression loss và validation metric theo epoch.
- `train.log`: cấu hình, thời gian chạy, cảnh báo và lỗi.
- `config_snapshot.yaml`: cấu hình của thí nghiệm nếu triển khai snapshot.

## Cần theo dõi

- Loss có NaN/Inf hay không.
- Learning rate và thời gian mỗi epoch.
- Validation mAP@0.5 và mAP@0.5:0.95.
- Peak VRAM nếu có thể đo.

Không so sánh trực tiếp độ lớn loss của RetinaNet với Faster R-CNN như một metric chất lượng.
