# Thư mục `data/splits/`

## Mục đích

Lưu annotation COCO cố định cho ba tập dữ liệu dùng chung bởi cả Faster R-CNN và RetinaNet.

## Tệp cần tạo

- `train.json`: dùng để cập nhật trọng số.
- `val.json`: dùng để theo dõi metric, chọn checkpoint và điều chỉnh cấu hình.
- `test.json`: chỉ dùng cho đánh giá cuối cùng sau khi chốt cấu hình.
- `split_summary.json`: số ảnh/box theo lớp và thông tin seed/quy tắc chia.

## Việc cần làm

1. Chia theo nhóm cảnh/video nếu có, không chia ngẫu nhiên từng frame liên tiếp.
2. Giữ phân bố lớp hợp lý giữa ba split.
3. Xác nhận một image ID chỉ thuộc một split.
4. Xác nhận annotation/category metadata đầy đủ trong mỗi COCO JSON.
5. Tạo hash hoặc manifest để phát hiện split bị thay đổi ngoài ý muốn.

## Quy tắc

- Không dùng test để chọn learning rate, epoch, threshold hoặc checkpoint.
- Không tạo split khác nhau cho hai mô hình.
- Nếu bắt buộc sửa split, phải hủy kết quả cũ và đánh giá lại cả hai mô hình.
