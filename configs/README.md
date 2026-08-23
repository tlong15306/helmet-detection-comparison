# Thư mục `configs/`

## Mục đích

Lưu toàn bộ tham số điều khiển thí nghiệm. Đây là nguồn thông tin chính để biết một checkpoint đã được huấn luyện như thế nào.

## Các tệp

- `common.yaml`: lớp nhãn, đường dẫn dữ liệu, kích thước ảnh, seed, augmentation, evaluator và cấu hình thiết bị dùng chung.
- `faster_rcnn.yaml`: kiến trúc, optimizer, scheduler, epoch và đường dẫn đầu ra của Faster R-CNN.
- `retinanet.yaml`: cấu hình tương ứng của RetinaNet.

## Việc cần làm

1. Xác nhận dataset và label mapping trong `common.yaml`.
2. Chạy kiểm tra môi trường để xác nhận `device`, số worker và mixed precision.
3. Chốt baseline dùng chung trước khi điều chỉnh riêng từng mô hình.
4. Mỗi lần thay đổi cấu hình quan trọng phải tạo experiment ID mới và lưu bản cấu hình cùng kết quả.
5. Khi viết báo cáo, lấy tham số từ các tệp này thay vì nhớ hoặc nhập lại bằng tay.

## Quy tắc

- Không viết đường dẫn tuyệt đối phụ thuộc máy cá nhân.
- Không chỉnh cấu hình dựa trên kết quả của tập test.
- Mọi khác biệt giữa hai mô hình phải có lý do và được nêu trong báo cáo.
- Giá trị hiện tại chỉ là baseline dự kiến, chưa phải cấu hình đã được kiểm chứng.
