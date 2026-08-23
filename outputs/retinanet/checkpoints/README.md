# Checkpoint RetinaNet

## Tệp dự kiến

- `best_map.pth`: checkpoint tốt nhất theo validation mAP@0.5:0.95.
- `last.pth`: trạng thái epoch cuối.
- Checkpoint phục hồi nếu cần tiếp tục một lần train bị gián đoạn.

## Nội dung checkpoint nên lưu

- Model, optimizer và scheduler state.
- Epoch, best metric, experiment ID và config snapshot.
- Label mapping và phiên bản thư viện.

Không chọn checkpoint bằng metric test. Checkpoint dùng cho demo phải đúng checkpoint đã ghi trong manifest.
