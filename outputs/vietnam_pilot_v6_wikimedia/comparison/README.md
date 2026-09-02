# So sánh Faster R-CNN và RetinaNet v6

`test_comparison.json` được tạo bằng `src.compare_models` từ đúng hai file
`test_metrics.json` của v6. Chỉ dùng bảng này khi `comparison_status` là
`comparable`; nếu khác, kiểm tra split hash, class mapping, checkpoint và giao
thức evaluator trước khi viết nhận xét.
