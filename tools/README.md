# Thư mục `tools/`

## Mục đích

Chứa các công cụ hỗ trợ chuẩn bị và kiểm tra dự án. Các tệp ở đây không phải lõi mô hình nhưng tạo bằng chứng cần thiết cho báo cáo.

## Công cụ và thứ tự sử dụng

1. `check_environment.py`
   - Chạy trước tiên.
   - Ghi phiên bản Python/PyTorch/Torchvision, CUDA, GPU và VRAM.
2. `inspect_dataset.py`
   - Chạy sau khi có ảnh và annotation COCO.
   - Kiểm tra cấu trúc COCO, ID, category, bounding box, ảnh thiếu/hỏng,
     kích thước ảnh và các tệp ảnh không được annotation tham chiếu.
   - Lưu báo cáo bằng tùy chọn `--output outputs/dataset_report.json`.
3. `visualize_annotations.py`
   - Vẽ bounding box lên ảnh mẫu.
   - Dùng để phát hiện category mapping hoặc tọa độ sai.
4. `prepare_annotations.py`
   - Tạo bản annotation processed và nhật ký thay đổi, không sửa `data/raw/`.
   - Chỉ clip bbox vượt biên nhỏ; bbox lỗi nghiêm trọng được loại và ghi rõ để xem lại.
5. `build_groups.py`
   - Tính SHA-256, tìm ảnh trùng chính xác và xuất ứng viên gần trùng lặp.
   - Manifest chỉ tự nhóm ảnh trùng chính xác; nhóm cảnh/video cần được người phụ trách duyệt.
6. `create_splits.py`
   - Tạo split cố định theo `group_id`, seed và tỷ lệ đã xác nhận.
7. `freeze_splits.py`
   - Kiểm tra giao nhau, category/bbox và lưu SHA-256 của processed annotation cùng ba split.
   - Chạy ngay trước smoke test hoặc train chính thức.
8. `benchmark_speed.py`
   - Chạy sau khi có checkpoint tốt nhất.
   - Đo latency/FPS của hai mô hình theo cùng giao thức.
9. `src.threshold_selection`
   - Chạy suy luận một lần trên validation, quét threshold từ 0,05 đến 0,95.
   - Chọn threshold có F1 cao nhất cho lớp `NoHelmet`; khi bằng nhau ưu tiên Recall,
     rồi Precision và threshold cao hơn.
   - Lưu artifact vào `outputs/<model>/metrics/validation_threshold_selection.json`
     và cập nhật `configs/demo_thresholds.yaml` cho demo dùng lại.

## Quy tắc

- Công cụ không được âm thầm sửa `data/raw/`.
- Mọi tệp tạo ra phải có đường dẫn đầu ra rõ ràng.
- Nếu phát hiện lỗi dữ liệu, lưu minh chứng vào `data/samples/invalid_examples/`.
- Benchmark phải ghi phần cứng, kích thước ảnh, warm-up, số lượt đo và phạm vi thời gian được tính.
- Tập test không được đọc bởi công cụ chọn threshold cho demo.
# Benchmark tốc độ suy luận

`benchmark_inference.py` đo latency/FPS công bằng cho hai checkpoint tốt nhất.
Nó luôn dùng ảnh validation, 20 ảnh warm-up và 100 ảnh đo mặc định; test split
không được dùng để điều chỉnh tốc độ hoặc ngưỡng confidence.

```powershell
python -m tools.benchmark_inference --config configs/faster_rcnn.yaml --checkpoint outputs/faster_rcnn/checkpoints/best_map.pth --output outputs/faster_rcnn/metrics/latency_validation.json --device cuda
```
