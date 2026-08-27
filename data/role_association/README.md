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
