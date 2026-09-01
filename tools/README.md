# Thư mục `tools/`

Chỉ giữ các công cụ tái lập pipeline, kiểm tra dữ liệu, đánh giá và chạy demo.
Các script duyệt nhãn/pseudo-label dùng một lần được giữ local và không đưa lên
GitHub.

## Chuẩn bị và kiểm tra dữ liệu

1. `check_environment.py`: ghi nhận Python, PyTorch, CUDA và GPU.
2. `inspect_dataset.py`: kiểm tra COCO, ảnh hỏng, bbox và category.
3. `prepare_annotations.py`: tạo annotation đã làm sạch mà không sửa dữ liệu raw.
4. `build_groups.py`: tính hash và hỗ trợ phát hiện ảnh trùng.
5. `create_splits.py`: tạo train/validation/test theo group và seed.
6. `freeze_splits.py`: khóa fingerprint và kiểm tra rò rỉ giữa các split.
7. `visualize_annotations.py`: vẽ mẫu annotation để rà trực quan.

## Đánh giá và demo

- `benchmark_inference.py`, `benchmark_speed.py`: đo latency/FPS theo giao thức
  cố định.
- `check_deployment_gate.py`: so candidate với baseline trên validation.
- `evaluate_fusion.py`: đánh giá hợp nhất Faster R-CNN và RetinaNet.
- `profile_validation_thresholds.py`: phân tích threshold trên validation và từ
  chối tệp có tên chứa `test`.
- `run_local_demo_image_batch.py`: gửi một thư mục ảnh tới API demo cục bộ.
- `build_challenge_contact_sheet.py`, `select_edgevision_hard_subset.py` và
  `dedupe_challenge_images.py`: chuẩn bị tập ảnh khó phục vụ phân tích lỗi.

## Nguyên tắc

- Không dùng tập test để chọn threshold hoặc quyết định candidate.
- Không dùng pseudo-label chưa duyệt làm ground truth báo cáo.
- Artifact sinh ra được ghi vào `outputs/` hoặc `data/processed/` và không commit.
