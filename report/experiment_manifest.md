# Experiment Manifest

Manifest này tổng hợp **lần baseline EdgeVision đã hoàn thành** từ run manifest,
metric test, benchmark và artifact chọn threshold. Candidate fine-tune Việt Nam
v6 được ghi riêng trong README; không thay số liệu baseline học thuật.

| Trường | Faster R-CNN | RetinaNet |
| --- | --- | --- |
| Loại lần chạy | `baseline` | `baseline` |
| Thời gian chạy (UTC+07:00) | 26-08-2026, 14:47–17:23 | 26-08-2026, 19:35–21:50 |
| Config | `configs/faster_rcnn.yaml` | `configs/retinanet.yaml` |
| Git commit khi train | `3504cee686e1277e57ff4ac958bc71374022016e` | Giống Faster R-CNN |
| Dataset | EdgeVision v1 cục bộ, annotation COCO đã xử lý | Giống Faster R-CNN |
| Train / validation / test | 1.673 / 360 / 359 ảnh | Giống Faster R-CNN |
| Frozen split manifest SHA-256 | `a00370c5ee6413aa4a7f6f88da7dcdec16b0e04e1e63c1c40c961e775a947854` | Giống Faster R-CNN |
| Cấu hình chung | 20 epoch, batch size 1, SGD, LR 0,0025, ảnh 512–768 px | Giống Faster R-CNN |
| Checkpoint tốt nhất | Epoch 9, `outputs/faster_rcnn/checkpoints/best_map.pth` | Epoch 8, `outputs/retinanet/checkpoints/best_map.pth` |
| SHA-256 checkpoint tốt nhất | `27fc925e68cd908e82b3865f3781ea01ee643c67674a392e62d7893d59f92682` | `5f3e4cb963e2c079094254b261dec15e21b0b2784d5aa1fd34756ff006ed5ed5` |
| Validation mAP@0.5:0.95 tại best epoch | 0,6519 | 0,6443 |
| Test mAP@0.5:0.95 | 0,6562 | 0,6472 |
| Test mAP@0.5 / mAP@0.75 / mAR@100 | 0,9070 / 0,7400 / 0,7317 | 0,8990 / 0,7457 / 0,7436 |
| Test AP@0.5:0.95 lớp `NoHelmet` | 0,5584 | 0,5386 |
| Confidence threshold cho demo | BikeWithRider 0,95; NoHelmet 0,65; Helmet 0,70 | BikeWithRider 0,65; NoHelmet 0,40; Helmet 0,40 |
| Latency trung bình / FPS | 163,59 ms / 6,11 FPS | 75,24 ms / 13,29 FPS |
| Giao thức benchmark | 20 warm-up + 100 ảnh validation, batch size 1 | Giống Faster R-CNN |
| Phần cứng | NVIDIA GeForce RTX 2050 | NVIDIA GeForce RTX 2050 |
| Phần mềm | Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121 | Giống Faster R-CNN |

## Fine-tune Việt Nam v6 đang dùng cho demo

Hai candidate bắt đầu từ checkpoint baseline tương ứng, đóng băng backbone và
fine-tune thêm một epoch trên train gồm 1.673 ảnh EdgeVision, 449 ảnh Việt Nam
đã duyệt và 4 ảnh Wikimedia. Candidate chưa được dùng thay baseline trong báo
cáo; sau khi chốt cho demo, hai checkpoint được đánh giá một lần trên cùng
EdgeVision test để lưu artifact so sánh.

| Candidate v6 | Val mAP@0.5:0.95 | Test mAP@0.5:0.95 | Test AP `NoHelmet` | Checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Faster R-CNN | 0,6512 | 0,6604 | 0,5598 | `6869faff03a30c497fd60d1a61ef624ae2cc41e261b55030efd8816a980f8348` |
| RetinaNet | 0,6404 | 0,6508 | 0,5405 | `d02de4c3a4e76bb4a7898ff8ca04f40104085696532e60a1521f6fa08650263b` |

## Lưu ý diễn giải

- Checkpoint baseline được chọn trên validation; tập test chỉ dùng cho đánh giá cuối cùng.
- Candidate v6 chưa vượt baseline trên EdgeVision validation; evaluation test v6 được chạy sau khi chốt candidate và không dùng để tinh chỉnh.
- Latency gồm chuyển tensor CPU–GPU, biến đổi nội bộ, forward và NMS; không gồm đọc tệp, giao diện hoặc ghi kết quả.
- Checkpoint, dataset, log và JSON metric là artifact cục bộ; không commit lên GitHub. Cần đối chiếu SHA-256 khi bàn giao.

## Nguồn artifact cục bộ

- `outputs/faster_rcnn/run_manifest.json`, `outputs/retinanet/run_manifest.json`
- `outputs/faster_rcnn/metrics/test_metrics.json`, `outputs/retinanet/metrics/test_metrics.json`
- `outputs/faster_rcnn/metrics/latency_validation.json`, `outputs/retinanet/metrics/latency_validation.json`
- `outputs/vietnam_pilot_v6_wikimedia/faster_rcnn/stage1/checkpoints/best_map.pth`
- `outputs/vietnam_pilot_v6_wikimedia/retinanet/stage1/checkpoints/best_map.pth`
- `outputs/vietnam_pilot_v6_wikimedia/faster_rcnn/stage1/metrics/test_metrics.json`
- `outputs/vietnam_pilot_v6_wikimedia/retinanet/stage1/metrics/test_metrics.json`
- `outputs/vietnam_pilot_v6_wikimedia/comparison/test_comparison.json`
