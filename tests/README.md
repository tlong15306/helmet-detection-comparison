# Thư mục `tests/`

## Mục đích

Kiểm tra pipeline trước khi tốn thời gian huấn luyện. Một lỗi nhỏ về label hoặc bounding box có thể làm hỏng toàn bộ thí nghiệm.

## Các nhóm kiểm thử

- `test_dataset.py`: ảnh tồn tại, box có dạng `[x1, y1, x2, y2]`, box nằm trong ảnh, label hợp lệ và batch collate đúng.
- `test_models.py`: tạo được hai mô hình, số lớp đầu ra đúng và forward pass hoạt động.
- `test_metrics.py`: kiểm tra IoU, Precision, Recall và schema đầu ra mAP.
- `test_inference.py`: checkpoint được nạp đúng, threshold hoạt động và đầu ra có boxes/labels/scores.

## Việc cần làm

1. Tạo một COCO fixture rất nhỏ, không dùng toàn bộ dataset.
2. Bỏ trạng thái `skip` khi fixture/checkpoint thử nghiệm đã sẵn sàng.
3. Chạy test sau mỗi thay đổi liên quan đến dữ liệu, model head hoặc metric.
4. Thêm regression test cho mọi lỗi đã từng gặp.

## Lệnh dự kiến

```powershell
pytest
pytest tests/test_dataset.py -v
```

Test thành công không thay thế cho việc kiểm tra trực quan annotation và prediction.
