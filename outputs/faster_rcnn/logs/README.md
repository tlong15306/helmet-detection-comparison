# Log Faster R-CNN

## Tệp dự kiến

- `history.csv`: loss và validation metric theo epoch.
- `train.log`: thời gian bắt đầu/kết thúc, cấu hình, cảnh báo và lỗi.
- `config_snapshot.yaml`: cấu hình thực tế của lần chạy nếu triển khai snapshot.

## Cần ghi

- Tổng loss và các thành phần loss do mô hình trả về.
- Learning rate theo epoch.
- Validation mAP@0.5 và mAP@0.5:0.95.
- Thời gian mỗi epoch, GPU và peak VRAM nếu đo được.

Không dùng training loss làm kết luận cuối cùng về chất lượng phát hiện.
