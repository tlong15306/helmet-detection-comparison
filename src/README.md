# Thư mục `src/`

## Mục đích

Chứa mã nguồn chính của pipeline: đọc dữ liệu, khởi tạo mô hình, huấn luyện, đánh giá, suy luận và so sánh kết quả.

## Vai trò từng tệp

- `dataset.py`: đọc COCO JSON và trả về ảnh, bounding box, label cùng metadata.
- `transforms.py`: áp dụng augmentation đồng bộ lên ảnh và bounding box.
- `models.py`: khởi tạo Faster R-CNN/RetinaNet pretrained và thay detection head.
- `train.py`: điều phối fine-tune, validation, checkpoint và log.
- `evaluate.py`: đánh giá checkpoint trên validation/test và xuất metric.
- `infer.py`: suy luận ảnh hoặc video và lưu kết quả trực quan.
- `metrics.py`: IoU, Precision, Recall và COCO-style mAP.
- `compare_models.py`: tạo bảng/biểu đồ so sánh hai mô hình.
- `utils.py`: đọc cấu hình, quản lý đường dẫn, seed và tiện ích dùng chung.

## Thứ tự hoàn thiện

1. Hoàn thiện và kiểm thử `dataset.py` cùng `transforms.py`.
2. Kiểm tra `models.py` tạo được cả hai kiến trúc với số lớp đúng.
3. Hoàn thiện smoke test trong `train.py`.
4. Hoàn thiện validation/evaluator trước khi train dài.
5. Chỉ triển khai `compare_models.py` sau khi hai mô hình xuất cùng schema metric.

## Tiêu chí hoàn thành

- Không hard-code đường dẫn hoặc số lớp trong nhiều tệp.
- Chạy lại cùng seed và config phải cho quy trình tương đương.
- Mọi kết quả quan trọng được lưu ra tệp, không chỉ in ra console.
- Hai mô hình dùng cùng dataset loader, split và evaluator.
- Mỗi mô-đun có kiểm thử tương ứng trong `tests/`.

## Trạng thái hiện tại

Dataset, model factory và metric cơ bản đã có khung. Vòng lặp train/evaluate/inference vẫn chưa được triển khai hoàn chỉnh.
