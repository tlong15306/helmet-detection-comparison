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
7. `benchmark_speed.py`
   - Chạy sau khi có checkpoint tốt nhất.
   - Đo latency/FPS của hai mô hình theo cùng giao thức.

## Quy tắc

- Công cụ không được âm thầm sửa `data/raw/`.
- Mọi tệp tạo ra phải có đường dẫn đầu ra rõ ràng.
- Nếu phát hiện lỗi dữ liệu, lưu minh chứng vào `data/samples/invalid_examples/`.
- Benchmark phải ghi phần cứng, kích thước ảnh, warm-up, số lượt đo và phạm vi thời gian được tính.
