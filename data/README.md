# Thư mục `data/`

## Mục đích

Quản lý dữ liệu theo từng giai đoạn mà không làm thay đổi bản gốc và không gây rò rỉ giữa train, validation, test.

## Các thư mục con

- `raw/`: dataset tải về ở trạng thái nguyên gốc.
- `processed/`: annotation đã xử lý, có nhật ký thay đổi, dùng làm đầu vào để chia tập.
- `splits/`: ba tệp COCO JSON cố định cho train, validation và test.
- `samples/`: ảnh trực quan hóa phục vụ kiểm tra chất lượng.
- `challenge/`: tập ảnh khó tách biệt dùng đánh giá robustness; không trộn vào split EdgeVision đã đóng băng.
- `processed/vietnam_intake/`: draft, hash và review queue của nguồn Việt Nam;
  không được train trực tiếp.
- `processed/vietnam_pilot/`: candidate đã hoàn tất review, có manifest riêng;
  chỉ train split được gộp với EdgeVision train.

## Quy trình bắt buộc

1. Ghi nguồn, giấy phép, phiên bản và ngày tải.
2. Kiểm tra số ảnh, category ID, tên lớp và bounding box.
3. Nếu ảnh có EXIF Orientation, đồng bộ kích thước annotation với ảnh sau khi xoay đúng chiều.
4. Kiểm tra trực quan ảnh đại diện của từng lớp.
5. Xác định cách tránh rò rỉ giữa các ảnh cùng cảnh/video.
6. Tạo split đúng một lần và lưu bản tóm tắt.
7. Không sửa split khi chuyển từ Faster R-CNN sang RetinaNet.
8. Không dùng `challenge/` để chọn hyperparameter hoặc confidence threshold.
9. Với nguồn annotation một phần, chỉ train sau khi mỗi `BikeWithRider` có đúng
   trạng thái `Helmet` hoặc `NoHelmet` đã duyệt; ảnh mơ hồ phải loại khỏi pilot.

## Thông tin phải đưa vào báo cáo

- Nguồn và giấy phép dataset.
- Số ảnh và bounding box theo lớp.
- Tiêu chí loại bỏ ảnh/annotation lỗi.
- Tỷ lệ và số lượng train/validation/test.
- Cách ngăn dữ liệu cùng cảnh xuất hiện ở nhiều split.

Không đặt dữ liệu có bản quyền không rõ ràng hoặc thông tin cá nhân nhạy cảm vào dự án.
