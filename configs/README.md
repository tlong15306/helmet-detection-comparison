# Thư mục `configs/`

Thư mục này chỉ giữ các cấu hình cần để tái lập baseline và chạy hai checkpoint
Việt Nam đang xuất hiện trong demo.

## Cấu hình công khai

- `common.yaml`: lớp nhãn, dữ liệu, kích thước ảnh, seed và evaluator dùng chung.
- `faster_rcnn.yaml`: baseline Faster R-CNN 20 epoch.
- `retinanet.yaml`: baseline RetinaNet 20 epoch.
- `pilot_faster_rcnn.yaml`, `pilot_retinanet.yaml`: smoke/pilot ngắn của pipeline.
- `rider_association.yaml`: tham số ghép đầu với `BikeWithRider`.
- `demo_thresholds.yaml`: threshold, fingerprint checkpoint và hậu xử lý demo.
- `vietnam_v6_wikimedia_faster_rcnn.yaml`: checkpoint Faster R-CNN fine-tune
  trên train gốc đã gộp dữ liệu Việt Nam được duyệt.
- `vietnam_v6_wikimedia_retinanet.yaml`: cấu hình tương ứng của RetinaNet.

Hai config Việt Nam v6 tự kế thừa trực tiếp `common.yaml`, không phụ thuộc các
config thử nghiệm cũ. Checkpoint và dữ liệu vẫn là artifact cục bộ, không commit
lên GitHub.

## Quy tắc

- Không ghi đường dẫn tuyệt đối phụ thuộc máy cá nhân.
- Không điều chỉnh cấu hình dựa trên tập test.
- Không ghi đè checkpoint baseline khi chạy candidate.
- Threshold báo cáo phải được chọn trên validation; hai checkpoint Việt Nam v6
  đang mang nhãn thử nghiệm và dùng threshold kế thừa để kiểm tra trực quan.
