# Helmet Detection Comparison

Dự án fine-tune và so sánh Faster R-CNN với RetinaNet cho bài toán phát hiện người điều khiển xe máy không đội mũ bảo hiểm từ ảnh giao thông.

## Trạng thái

- Đã khởi tạo cấu trúc dự án.
- Đã khóa cấu hình môi trường theo vai trò trong `HUONG_DAN_CAI_DAT.md`.
- Đã tiếp nhận EdgeVision v1 cục bộ và tạo bản annotation processed có nhật ký thay đổi.
- Đã tạo và đóng băng split theo seed 42 sau khi kiểm tra nguy cơ ảnh cùng cảnh.
- Đã hoàn thành smoke test CUDA cho Faster R-CNN và RetinaNet; chưa có kết quả thực nghiệm chính thức.
- Đã dựng frontend React và kết nối FastAPI cho suy luận ảnh bằng hai checkpoint tốt nhất.
- Môi trường trên từng máy thành viên cần được cài và xác minh riêng.
- Chưa huấn luyện baseline chính thức hoặc có kết quả thực nghiệm cuối cùng.

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
4. Chạy smoke test một batch cho từng mô hình.
5. Chạy pilot 3 epoch trên validation để xác nhận pipeline và ước lượng thời gian.
6. Fine-tune baseline và lưu checkpoint tốt nhất theo validation mAP@0.5:0.95.
7. Đánh giá hai mô hình trên cùng tập test đúng một lần sau khi chốt cấu hình.
8. Tạo bảng, biểu đồ, demo và hoàn thiện báo cáo.

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
$env:PYTHONUTF8 = "1"
python -m src.train --config configs/pilot_faster_rcnn.yaml --device cuda
python -m src.train --config configs/pilot_retinanet.yaml --device cuda
python -m src.train --config configs/faster_rcnn.yaml
python -m src.train --config configs/retinanet.yaml
python -m src.evaluate --config configs/faster_rcnn.yaml --split test
python -m src.evaluate --config configs/retinanet.yaml --split test
python -m src.threshold_selection --config configs/faster_rcnn.yaml --output outputs/faster_rcnn/metrics/validation_threshold_selection.json
python -m src.threshold_selection --config configs/retinanet.yaml --output outputs/retinanet/metrics/validation_threshold_selection.json
python -m tools.benchmark_inference --config configs/faster_rcnn.yaml --checkpoint outputs/faster_rcnn/checkpoints/best_map.pth --output outputs/faster_rcnn/metrics/latency_validation.json --device cuda
python -m tools.benchmark_inference --config configs/retinanet.yaml --checkpoint outputs/retinanet/checkpoints/best_map.pth --output outputs/retinanet/metrics/latency_validation.json --device cuda
python -m src.compare_models
```

## Chạy demo ảnh React + FastAPI

Mở hai cửa sổ PowerShell tại thư mục dự án.

Backend:

```powershell
.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

Mở `http://127.0.0.1:5173/`. Chế độ ảnh đã hoạt động; video và camera là giai đoạn tiếp theo.

## Quy tắc quan trọng

- Không sửa trực tiếp dữ liệu trong `data/raw/`.
- Không dùng tập test để chọn hyperparameter.
- Hai mô hình phải dùng cùng các tệp split và evaluator.
- Confidence threshold cho demo phải được chọn trên validation; không dùng test.
- Chạy `tools/freeze_splits.py` trước smoke test hoặc train chính thức; thay đổi split sau đó làm mất hiệu lực artifact cũ.
- Pilot dùng `configs/pilot_*.yaml`, lưu độc lập tại `outputs/pilot/` và không được resume từ checkpoint smoke.
- Benchmark latency/FPS dùng 20 ảnh warm-up và 100 ảnh validation, batch size 1; thời gian bao gồm chuyển tensor lên GPU, inference và NMS, không gồm đọc ảnh/render giao diện.
- Không đưa số liệu chưa được xuất từ log/JSON/CSV vào báo cáo.
- Không kết luận mô hình nào tốt hơn trước khi có kết quả kiểm thử.
