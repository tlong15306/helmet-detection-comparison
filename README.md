# Helmet Detection Comparison

Dự án fine-tune và so sánh Faster R-CNN với RetinaNet cho bài toán phát hiện người điều khiển xe máy không đội mũ bảo hiểm từ ảnh giao thông.

## Trạng thái

- Đã khởi tạo cấu trúc dự án.
- Đã khóa cấu hình môi trường theo vai trò trong `HUONG_DAN_CAI_DAT.md`.
- Đã tiếp nhận EdgeVision v1 cục bộ và tạo bản annotation processed có nhật ký thay đổi.
- Đã tạo split tạm thời theo seed 42; cần duyệt thêm nguy cơ ảnh cùng cảnh trước khi đóng băng kết quả thực nghiệm.
- Môi trường trên từng máy thành viên cần được cài và xác minh riêng.
- Chưa huấn luyện hoặc có kết quả thực nghiệm.

## Cài đặt

Nhóm dùng Python 3.11 và cài theo một trong ba chế độ `gpu`, `cpu` hoặc `data`:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_environment.ps1 -InstallMode gpu
```

Xem hướng dẫn đầy đủ tại `HUONG_DAN_CAI_DAT.md`. Không chia sẻ `.venv` và không tự nâng phiên bản PyTorch/Torchvision trong thời gian thực nghiệm.

## Giả định tạm thời

- Dataset: EdgeVision.
- Annotation: COCO JSON.
- Các lớp đối tượng: `BikeWithRider`, `NoHelmet`, `Helmet`.
- Mô hình: `fasterrcnn_resnet50_fpn_v2` và `retinanet_resnet50_fpn_v2`.

Các giả định phải được xác nhận trước khi triển khai pipeline chính thức.

## Thứ tự thực hiện

1. Kiểm tra môi trường và dung lượng VRAM.
2. Tải dataset vào `data/raw/edgevision/`.
3. Kiểm tra annotation và tạo train/validation/test split.
4. Chạy smoke test một epoch cho từng mô hình.
5. Fine-tune và lưu checkpoint tốt nhất theo validation mAP@0.5:0.95.
6. Đánh giá hai mô hình trên cùng tập test.
7. Tạo bảng, biểu đồ, demo và hoàn thiện báo cáo.

## Lệnh dự kiến

Các lệnh dưới đây là giao diện dự kiến của dự án. Chỉ sử dụng sau khi các mô-đun tương ứng đã được hoàn thiện.

```powershell
python tools/check_environment.py
python tools/inspect_dataset.py --annotations data/raw/edgevision/annotations.json --images data/raw/edgevision/images --output outputs/dataset_report.json
python tools/prepare_annotations.py --annotations data/raw/edgevision/annotations.json --images data/raw/edgevision/images --apply-exif-orientation --processed-output data/processed/edgevision/annotations.json --changes-output data/processed/edgevision/annotation_changes.json --problems-output outputs/dataset_quality/problem_annotations.json
python tools/build_groups.py --annotations data/processed/edgevision/annotations.json --images data/raw/edgevision/images --output data/processed/edgevision/image_hashes.json --near-duplicate-output outputs/dataset_quality/near_duplicate_candidates.json
python tools/create_splits.py --annotations data/processed/edgevision/annotations.json --groups data/processed/edgevision/image_hashes.json --output-dir data/splits --seed 42
python tools/freeze_splits.py --seed 42
python -m src.train --config configs/faster_rcnn.yaml --smoke-test
python -m src.train --config configs/retinanet.yaml --smoke-test
python -m src.train --config configs/faster_rcnn.yaml
python -m src.train --config configs/retinanet.yaml
python -m src.evaluate --config configs/faster_rcnn.yaml --split test
python -m src.evaluate --config configs/retinanet.yaml --split test
python -m src.compare_models
```

## Quy tắc quan trọng

- Không sửa trực tiếp dữ liệu trong `data/raw/`.
- Không dùng tập test để chọn hyperparameter.
- Hai mô hình phải dùng cùng các tệp split và evaluator.
- Chạy `tools/freeze_splits.py` trước smoke test hoặc train chính thức; thay đổi split sau đó làm mất hiệu lực artifact cũ.
- Không đưa số liệu chưa được xuất từ log/JSON/CSV vào báo cáo.
- Không kết luận mô hình nào tốt hơn trước khi có kết quả kiểm thử.
