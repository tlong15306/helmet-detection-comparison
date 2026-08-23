# Thư mục `report/tables/`

## Mục đích

Lưu bảng trung gian hoặc bản xuất cuối dùng trong phần báo cáo 2.1.2.

## Bảng dự kiến

- Thống kê dataset và phân bố lớp.
- Cấu hình môi trường/phần cứng.
- Hyperparameter của Faster R-CNN và RetinaNet.
- mAP@0.5:0.95, mAP@0.5, AP theo lớp, Precision, Recall.
- Latency, FPS và peak VRAM nếu đo được.

## Quy tắc

- Mỗi giá trị phải truy ngược được tới config, log hoặc JSON/CSV kết quả.
- Ghi đơn vị và tập dữ liệu sử dụng.
- Dùng cùng số chữ số thập phân hợp lý.
- Không nhập lại thủ công nếu có thể tạo bảng từ `outputs/comparison/comparison.csv`.
- Không trộn số liệu validation và test trong cùng cột mà không ghi rõ.

Tên bảng và số thứ tự cuối cùng phải đồng nhất với báo cáo chung của nhóm.
