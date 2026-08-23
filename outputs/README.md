# Thư mục `outputs/`

## Mục đích

Lưu toàn bộ sản phẩm sinh ra từ huấn luyện, đánh giá, suy luận và benchmark. Đây là nguồn bằng chứng thực nghiệm cho báo cáo.

## Cấu trúc

- `faster_rcnn/`: kết quả riêng của Faster R-CNN.
- `retinanet/`: kết quả riêng của RetinaNet.
- `comparison/`: bảng và biểu đồ chỉ được tạo từ kết quả test chính thức của hai mô hình.

## Quy tắc chung

- Mỗi lần chạy chính thức phải có experiment ID, config và seed.
- Không ghi đè checkpoint tốt nhất nếu chưa lưu manifest của thí nghiệm cũ.
- Metric phải được chương trình xuất ra JSON/CSV; không sửa bằng tay.
- Không dùng kết quả validation thay cho test trong bảng kết luận.
- Checkpoint dung lượng lớn và tệp sinh tự động không đưa lên Git.
- Chỉ ảnh/bảng được chọn cho báo cáo mới được sao chép sang `report/`.
