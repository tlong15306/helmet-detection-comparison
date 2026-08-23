# Thư mục `data/`

## Mục đích

Quản lý dữ liệu theo từng giai đoạn mà không làm thay đổi bản gốc và không gây rò rỉ giữa train, validation, test.

## Các thư mục con

- `raw/`: dataset tải về ở trạng thái nguyên gốc.
- `splits/`: ba tệp COCO JSON cố định cho train, validation và test.
- `samples/`: ảnh trực quan hóa phục vụ kiểm tra chất lượng.

## Quy trình bắt buộc

1. Ghi nguồn, giấy phép, phiên bản và ngày tải.
2. Kiểm tra số ảnh, category ID, tên lớp và bounding box.
3. Kiểm tra trực quan ảnh đại diện của từng lớp.
4. Xác định cách tránh rò rỉ giữa các ảnh cùng cảnh/video.
5. Tạo split đúng một lần và lưu bản tóm tắt.
6. Không sửa split khi chuyển từ Faster R-CNN sang RetinaNet.

## Thông tin phải đưa vào báo cáo

- Nguồn và giấy phép dataset.
- Số ảnh và bounding box theo lớp.
- Tiêu chí loại bỏ ảnh/annotation lỗi.
- Tỷ lệ và số lượng train/validation/test.
- Cách ngăn dữ liệu cùng cảnh xuất hiện ở nhiều split.

Không đặt dữ liệu có bản quyền không rõ ràng hoặc thông tin cá nhân nhạy cảm vào dự án.
