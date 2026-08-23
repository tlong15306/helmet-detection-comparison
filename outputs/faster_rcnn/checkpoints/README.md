# Checkpoint Faster R-CNN

## Tệp dự kiến

- `best_map.pth`: checkpoint có validation mAP@0.5:0.95 tốt nhất.
- `last.pth`: checkpoint tại epoch cuối.
- Có thể bổ sung checkpoint phục hồi khi quá trình train bị gián đoạn.

## Nội dung checkpoint nên lưu

- `model_state_dict`.
- `optimizer_state_dict` và scheduler state nếu cần tiếp tục train.
- Epoch, best metric, experiment ID và config snapshot.
- Label mapping và phiên bản thư viện quan trọng.

Trước khi dùng cho demo, checkpoint phải được đánh giá trên test và ghi vào `report/experiment_manifest.md`.
