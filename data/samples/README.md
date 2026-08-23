# Thư mục `data/samples/`

## Mục đích

Lưu ảnh minh chứng phục vụ kiểm tra trực quan dữ liệu, không dùng làm dataset huấn luyện riêng.

## Thư mục con

- `annotated_examples/`: annotation được cho là hợp lệ và đại diện.
- `invalid_examples/`: lỗi hoặc trường hợp chưa rõ cần nhóm quyết định.

## Cách chọn mẫu

- Có đủ ba lớp.
- Có đối tượng nhỏ, che khuất, nhiều người/xe, ánh sáng khác nhau.
- Có cả tình huống dễ và khó.
- Ghi image ID hoặc tên tệp nguồn trên ảnh/metadata để truy ngược được.

Không chỉnh annotation chỉ dựa trên ảnh render trong thư mục này; mọi sửa đổi phải thực hiện qua quy trình làm sạch dữ liệu có ghi nhận.
