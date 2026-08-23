# Thư mục `data/raw/`

## Mục đích

Lưu dataset ngay sau khi tải, trước mọi thao tác chia tập hoặc chuyển đổi annotation.

## Việc cần làm

- Tạo một thư mục riêng cho từng dataset/phiên bản.
- Giữ nguyên tên tệp và cấu trúc nguồn khi có thể.
- Ghi URL/DOI, giấy phép, ngày tải và checksum nếu có.
- Đặt mọi mã chuyển đổi ở `tools/`, không chỉnh file gốc bằng tay.

## Quy tắc bảo vệ dữ liệu

- Không ghi đè ảnh hoặc annotation trong thư mục này.
- Không lưu ảnh augmentation tại đây.
- Không dùng trực tiếp `raw/` làm test split nếu split chưa được cố định.
- Dữ liệu dung lượng lớn không được commit lên Git.
