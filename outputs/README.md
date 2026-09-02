# Thư mục `outputs/`

## Mục đích

Lưu toàn bộ sản phẩm sinh ra từ huấn luyện, đánh giá, suy luận và benchmark. Đây là nguồn bằng chứng thực nghiệm cho báo cáo.

## Cấu trúc

- `faster_rcnn/`: kết quả riêng của Faster R-CNN.
- `retinanet/`: kết quả riêng của RetinaNet.
- `comparison/`: bảng và biểu đồ chỉ được tạo từ kết quả test chính thức của hai mô hình.
- `vietnam_pilot_v6_wikimedia/`: artifact của hai checkpoint fine-tune Việt Nam v6 đang hiển thị trên demo.

## Thu thập artifact cho nhóm

Với một lần chạy cần đối chiếu trong báo cáo hoặc demo, thu thập theo thứ tự:

1. Sau train, giữ `run_manifest.json` và checkpoint `best_map.pth` của từng model.
2. Chạy evaluator trên split đã chốt, lưu `metrics/test_metrics.json` cho cả Faster R-CNN và RetinaNet.
3. Chạy `src.compare_models` từ hai file metric đó, lưu `comparison/test_comparison.json`.
4. Chạy benchmark cùng cấu hình warm-up/measurement, lưu `metrics/latency_validation.json`.
5. Chọn cùng một số ảnh test cho hai model, lưu prediction minh họa ở `report/figures/` cùng README ghi checkpoint, ngưỡng và image ID.

Với v6, xem hướng dẫn và đường dẫn chính xác tại
[`vietnam_pilot_v6_wikimedia/README.md`](vietnam_pilot_v6_wikimedia/README.md).

## Quy tắc chung

- Mỗi lần chạy chính thức phải có experiment ID, config và seed.
- Không ghi đè checkpoint tốt nhất nếu chưa lưu manifest của thí nghiệm cũ.
- Metric phải được chương trình xuất ra JSON/CSV; không sửa bằng tay.
- Không dùng kết quả validation thay cho test trong bảng kết luận.
- Không đưa checkpoint, dataset, video hoặc log lớn lên GitHub.
- Có thể đưa lên GitHub JSON nhỏ đã kiểm chứng (manifest, test metric, comparison,
  latency) và vài prediction có README mô tả nguồn; chỉ force-add các file đó.
- Chỉ ảnh/bảng được chọn cho báo cáo mới được sao chép sang `report/`.
