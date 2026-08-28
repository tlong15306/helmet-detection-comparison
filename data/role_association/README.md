# Dữ liệu liên kết tài xế/người ngồi sau

Thư mục này dành cho annotation quan hệ bổ sung; không thay đổi nhãn COCO ba lớp gốc.

## Quy tắc sử dụng

- `annotations/role_dev.json`: ảnh validation dùng để kiểm tra quy tắc và chọn ngưỡng liên kết.
- `annotations/role_test.json`: ảnh độc lập đã đóng băng, chỉ dùng đánh giá cuối.
- `metadata/labeling_log.csv`: nguồn ảnh, người gán nhãn, trạng thái kiểm tra chéo và ghi chú che khuất.
- Không dùng `role_test` để điều chỉnh thuật toán.
- Chỉ commit annotation/metadata nhỏ có quyền sử dụng rõ ràng; không commit ảnh/video gốc nếu không cần.

## Schema quan hệ dự kiến

```json
{
  "image_id": 0,
  "vehicle_group_id": "vehicle_001",
  "bike_box_id": "...",
  "head_box_id": "...",
  "helmet_status": "helmet | no_helmet | unknown",
  "role": "driver | passenger | unknown",
  "occluded": false,
  "reviewer": "..."
}
```

Không tạo nhãn vai trò bằng prediction của model. Nhãn phải được người gán kiểm tra trên ảnh gốc.

## Quy trình review hiện tại

1. Tạo task pending bằng `python -m tools.create_role_dev_tasks`.
2. Mở web demo cục bộ và kéo đến mục **Review nhãn tài xế/người ngồi sau**.
3. Dựa trên tay lái/tư thế điều khiển, gán mỗi đầu là `driver`, `passenger` hoặc `unknown`.
4. Chỉ có một `driver` trong một task. Nếu không đủ bằng chứng, đặt mọi đầu là `unknown`.
5. Bấm **Lưu để kiểm tra chéo**. Hệ thống lưu tên người review, thời gian và chuyển task sang `needs_second_review`.
6. Sau khi hoàn tất, chạy validator trước khi dùng `role_dev` để chọn quy tắc.

### Báo cáo chẩn đoán vòng một

Sau khi có nhãn review, chạy:

```powershell
.venv\Scripts\python.exe -m tools.evaluate_role_association
```

Kết quả cục bộ nằm tại `outputs/role_association/role_dev_provisional_metrics.json`.
Chỉ số `candidate_precision` và `candidate_recall` chỉ đánh giá gợi ý
`driver_candidate` của quy tắc hình học trên `role_dev`; không phải mAP của
detector và không được dùng làm kết luận cuối cùng. Task ở trạng thái
`needs_second_review` vẫn cần người thứ hai kiểm tra trước khi dùng để điều
chỉnh quy tắc.

Không thay đổi `data/splits/test.json` và không dùng Hard Subset để chọn quy tắc vai trò.
