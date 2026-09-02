# Experiment Manifest — baseline gộp hoàn chỉnh

Manifest này tổng hợp lần train baseline chính thức trên tập train EdgeVision gộp ảnh giao thông Việt Nam đã duyệt. Số liệu chỉ lấy từ run manifest, metric test, benchmark và artifact chọn threshold cục bộ.

| Trường | Faster R-CNN | RetinaNet |
| --- | --- | --- |
| Loại lần chạy | `baseline` | `baseline` |
| Thời gian chạy UTC+07:00 | 02-09-2026, 00:04–03:28 | 02-09-2026, 03:43–06:23 |
| Config | `configs/final_combined_faster_rcnn.yaml` | `configs/final_combined_retinanet.yaml` |
| Git commit khi train | `65c7bcb5c1f60eb7aa8a764d0fd2db3d24c739bc` | `65c7bcb5c1f60eb7aa8a764d0fd2db3d24c739bc` |
| Train / validation / test | 2.126 / 360 / 359 ảnh | 2.126 / 360 / 359 ảnh |
| Train gộp | 1.673 EdgeVision + 449 Việt Nam đã duyệt + 4 Wikimedia Việt Nam | Giống Faster R-CNN |
| Frozen manifest SHA-256 | `0c5c4ddddefb046634a7b851293050a0a33f9f8457f744adc29d5facca3e41b4` | Giống Faster R-CNN |
| Cấu hình chung | 20 epoch, batch size 1, SGD, LR 0,0025, ảnh 512–768 px, AMP | Giống Faster R-CNN |
| Checkpoint tốt nhất | Epoch 8, `outputs/final_combined/faster_rcnn/checkpoints/best_map.pth` | Epoch 8, `outputs/final_combined/retinanet/checkpoints/best_map.pth` |
| SHA-256 checkpoint tốt nhất | `e81f73ba8cfe46b543d9141e211d4922548e07f7429ba523a70d228808997f55` | `f93c48cbf1c9cf4afcc25d7d2fafb767040c9cdb412c6ac170ffaf1f72951f1f` |
| Validation mAP@0.5:0.95 tại best epoch | 0,6548 | 0,6403 |
| Test mAP@0.5:0.95 | 0,6599 | 0,6425 |
| Test mAP@0.5 / mAP@0.75 / mAR@100 | 0,9078 / 0,7581 / 0,7371 | 0,8965 / 0,7276 / 0,7413 |
| Test AP `NoHelmet` | 0,5613 | 0,5325 |
| Threshold demo theo lớp | BikeWithRider 0,80; NoHelmet 0,70; Helmet 0,65 | BikeWithRider 0,60; NoHelmet 0,55; Helmet 0,60 |
| Latency trung bình / FPS | 162,70 ms / 6,15 FPS | 72,25 ms / 13,84 FPS |
| Giao thức benchmark | 20 warm-up + 100 ảnh validation, batch size 1 | Giống Faster R-CNN |
| Phần cứng | NVIDIA GeForce RTX 2050 | NVIDIA GeForce RTX 2050 |
| Phần mềm | Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121 | Giống Faster R-CNN |

## Kiểm tra ngoài train trên `vn_validation`

Tập `vn_validation` có 118 ảnh Việt Nam, được giữ ngoài train và không dùng để chọn checkpoint/threshold. Đây là đánh giá bổ sung, không thay thế EdgeVision test chính thức.

| Mô hình | mAP@0.5:0.95 | mAP@0.5 | AP `NoHelmet` | AP `Helmet` |
| --- | ---: | ---: | ---: | ---: |
| Faster R-CNN | 0,8023 | 0,8814 | 0,7076 | 0,7443 |
| RetinaNet | 0,8136 | 0,9055 | 0,7363 | 0,7754 |

## Lưu ý diễn giải

- Checkpoint và threshold được chọn trên EdgeVision validation; test được chạy sau khi hai lượt train hoàn tất.
- Faster R-CNN cao hơn 0,0174 mAP@0.5:0.95 và 0,0289 AP `NoHelmet` trên EdgeVision test. RetinaNet nhanh hơn khoảng 2,25 lần theo latency trung bình.
- `vn_validation` có ít box `NoHelmet` (31), vì vậy không đủ để khẳng định khả năng tổng quát cho mọi bối cảnh giao thông Việt Nam.
- Latency gồm chuyển tensor CPU–GPU, transform nội bộ, forward và NMS; không gồm đọc tệp, giao diện hoặc ghi kết quả.
- Checkpoint, dataset, log và JSON metric là artifact cục bộ; không commit lên GitHub. Khi bàn giao cần đối chiếu SHA-256.

## Nguồn artifact cục bộ

- `outputs/final_combined/faster_rcnn/run_manifest.json`, `outputs/final_combined/retinanet/run_manifest.json`
- `outputs/final_combined/faster_rcnn/metrics/test_metrics.json`, `outputs/final_combined/retinanet/metrics/test_metrics.json`
- `outputs/final_combined/faster_rcnn/metrics/vn_validation_metrics.json`, `outputs/final_combined/retinanet/metrics/vn_validation_metrics.json`
- `outputs/final_combined/faster_rcnn/metrics/latency_validation.json`, `outputs/final_combined/retinanet/metrics/latency_validation.json`
- `outputs/final_combined/faster_rcnn/metrics/validation_threshold_selection.json`, `outputs/final_combined/retinanet/metrics/validation_threshold_selection.json`
- `outputs/final_combined/comparison/test_comparison.json`
