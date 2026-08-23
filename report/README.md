# Thư mục `report/`

## Mục đích

Tập hợp bản thảo và tài sản đã được chọn để hoàn thiện phần báo cáo của Long. Thư mục này không thay thế `outputs/`; nó chỉ chứa nội dung đã được kiểm tra và sẵn sàng đưa vào báo cáo.

## Thành phần

- `2.1.2_huan_luyen_va_danh_gia.md`: bản thảo chính của Long.
- `experiment_manifest.md`: liên kết giữa nội dung báo cáo với config, checkpoint và metric thật.
- `references.bib`: tài liệu tham khảo đã kiểm chứng.
- `figures/`: hình và biểu đồ đã chọn.
- `tables/`: bảng dữ liệu dùng trong báo cáo.

## Quy trình viết

1. Viết trước phần phương pháp và định nghĩa metric bằng nguồn đáng tin cậy.
2. Giữ placeholder cho thông tin thực nghiệm chưa có.
3. Sau mỗi thí nghiệm, cập nhật manifest từ tệp cấu hình/log/metric.
4. Chỉ điền kết quả đã xuất tự động và được đối chiếu.
5. Viết nhận xét sau khi cả hai mô hình được đánh giá trên cùng test split.
6. Kiểm tra đánh số mục, hình, bảng, công thức và trích dẫn trước khi ghép DOCX.

## Quy tắc

- Không bịa số liệu, cấu hình, hình chụp hoặc tài liệu tham khảo.
- Không trình bày dự định như việc đã hoàn thành.
- Không kết luận mô hình tốt hơn nếu chưa xét đồng thời độ chính xác, Recall `NoHelmet`, tốc độ và tài nguyên.
- Không sửa trực tiếp các DOCX nguồn từ thư mục dự án gốc khi chưa được yêu cầu.
