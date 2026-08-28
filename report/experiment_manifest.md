# Experiment Manifest

Manifest này tổng hợp **lần baseline đã hoàn thành** từ run manifest, metric test, benchmark và artifact chọn threshold. Không dùng số liệu chưa được xuất từ artifact.

| Trường | Faster R-CNN | RetinaNet |
| --- | --- | --- |
| Loại lần chạy | `baseline` | `baseline` |
| Thời gian chạy (UTC+07:00) | 26-08-2026, 14:47–17:23 | 26-08-2026, 19:35–21:50 |
| Config | `configs/faster_rcnn.yaml` | `configs/retinanet.yaml` |
| Git commit khi train | `3504cee686e1277e57ff4ac958bc71374022016e` | `3504cee686e1277e57ff4ac958bc71374022016e` |
| Dataset | EdgeVision v1 cục bộ, annotation COCO đã xử lý | EdgeVision v1 cục bộ, annotation COCO đã xử lý |
| Train / validation / test | 1.673 / 360 / 359 ảnh | 1.673 / 360 / 359 ảnh |
| Frozen split manifest SHA-256 | `a00370c5ee6413aa4a7f6f88da7dcdec16b0e04e1e63c1c40c961e775a947854` | `a00370c5ee6413aa4a7f6f88da7dcdec16b0e04e1e63c1c40c961e775a947854` |
| Cấu hình chung | 20 epoch, batch size 1, SGD, LR 0,0025, ảnh 512–768 px | 20 epoch, batch size 1, SGD, LR 0,0025, ảnh 512–768 px |
| Checkpoint tốt nhất | Epoch 9, `outputs/faster_rcnn/checkpoints/best_map.pth` | Epoch 8, `outputs/retinanet/checkpoints/best_map.pth` |
| SHA-256 checkpoint tốt nhất | `27fc925e68cd908e82b3865f3781ea01ee643c67674a392e62d7893d59f92682` | `5f3e4cb963e2c079094254b261dec15e21b0b2784d5aa1fd34756ff006ed5ed5` |
| Validation mAP@0.5:0.95 tại best epoch | 0,6519 | 0,6443 |
| Test mAP@0.5:0.95 | 0,6562 | 0,6472 |
| Test mAP@0.5 / mAP@0.75 / mAR@100 | 0,9070 / 0,7400 / 0,7317 | 0,8990 / 0,7457 / 0,7436 |
| Confidence threshold cho demo | 0,85, chọn trên validation theo F1 lớp `NoHelmet` | 0,60, chọn trên validation theo F1 lớp `NoHelmet` |
| Latency trung bình / FPS | 163,59 ms / 6,11 FPS | 75,24 ms / 13,29 FPS |
| Giao thức benchmark | 20 warm-up + 100 ảnh validation, batch size 1 | 20 warm-up + 100 ảnh validation, batch size 1 |
| Phần cứng | NVIDIA GeForce RTX 2050 | NVIDIA GeForce RTX 2050 |
| Phần mềm | Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121 | Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121 |

## Lưu ý diễn giải

- Checkpoint được chọn trên validation; tập test chỉ được dùng cho lần đánh giá cuối cùng.
- Latency gồm chuyển tensor CPU–GPU, biến đổi nội bộ của Torchvision, forward và NMS; không gồm đọc tệp, giao diện hoặc ghi kết quả.
- Chênh lệch mAP@0.5:0.95 là 0,0090, nên không đủ để kết luận một mô hình vượt trội hoàn toàn. RetinaNet nhanh hơn trong giao thức benchmark này.
- Checkpoint, dataset, log và JSON metric sinh tự động là artifact cục bộ, không được commit lên GitHub. Cần đối chiếu SHA-256 khi bàn giao checkpoint.

## Nguồn artifact cục bộ

- `outputs/faster_rcnn/run_manifest.json`, `outputs/retinanet/run_manifest.json`
- `outputs/faster_rcnn/metrics/test_metrics.json`, `outputs/retinanet/metrics/test_metrics.json`
- `outputs/faster_rcnn/metrics/latency_validation.json`, `outputs/retinanet/metrics/latency_validation.json`
- `outputs/faster_rcnn/metrics/validation_threshold_selection.json`, `outputs/retinanet/metrics/validation_threshold_selection.json`
- `outputs/comparison/test_comparison.json`
