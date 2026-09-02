# Thư mục `configs/`

Thư mục này giữ các cấu hình cần để tái lập baseline EdgeVision và hai
checkpoint Việt Nam v6 đang xuất hiện trong demo.

## Cấu hình công khai

- `common.yaml`: lớp nhãn, dữ liệu, kích thước ảnh, seed và evaluator dùng chung.
- `faster_rcnn.yaml`: baseline Faster R-CNN 20 epoch.
- `retinanet.yaml`: baseline RetinaNet 20 epoch.
- `pilot_faster_rcnn.yaml`, `pilot_retinanet.yaml`: smoke/pilot ngắn của pipeline.
- `rider_association.yaml`: tham số ghép đầu với `BikeWithRider`.
- `demo_thresholds.yaml`: threshold, fingerprint checkpoint và hậu xử lý demo.
- `vietnam_v6_wikimedia_faster_rcnn.yaml`: Faster R-CNN fine-tune từ baseline
  trên train gốc có bổ sung dữ liệu Việt Nam đã duyệt.
- `vietnam_v6_wikimedia_retinanet.yaml`: cấu hình tương ứng của RetinaNet.
- `final_combined_*.yaml`: artifact train gộp được lưu để đối chiếu, không phải
  cấu hình demo hiện tại.

Hai config Việt Nam v6 tự kế thừa trực tiếp `common.yaml`, không phụ thuộc các
config thử nghiệm cũ. Checkpoint và dữ liệu vẫn là artifact cục bộ, không commit
lên GitHub.

## Quy tắc

- Không ghi đường dẫn tuyệt đối phụ thuộc máy cá nhân.
- Không điều chỉnh cấu hình dựa trên tập test.
- Không ghi đè checkpoint baseline khi chạy thử nghiệm khác.
- Threshold demo/báo cáo phải được chọn trên validation, không dùng test để
  điều chỉnh checkpoint hoặc ngưỡng.
