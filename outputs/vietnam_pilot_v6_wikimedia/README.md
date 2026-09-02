# Artifact fine-tune Việt Nam v6

Đây là thư mục artifact của hai model **đang hiển thị trên demo**. Cả hai được
fine-tune một epoch từ checkpoint EdgeVision baseline. Không đặt checkpoint hay
dataset lên GitHub; chỉ các JSON tái lập và prediction minh họa đã kiểm chứng
được version control.

## Cấu trúc cần thu thập

```text
vietnam_pilot_v6_wikimedia/
├── faster_rcnn/stage1/
│   ├── run_manifest.json
│   ├── checkpoints/best_map.pth                 # chỉ lưu cục bộ
│   └── metrics/
│       ├── test_metrics.json
│       └── latency_validation.json
├── retinanet/stage1/                            # cùng cấu trúc
└── comparison/test_comparison.json
```

## Cách thu thập lại

1. Giữ `run_manifest.json` được train tạo ra, không sửa tay.
2. Khi đã chốt checkpoint, chạy `src.evaluate --split test` cho từng config v6
   và lưu đúng đường dẫn `metrics/test_metrics.json`.
3. Chạy `src.compare_models` từ hai JSON test; kết quả phải có
   `comparison_status: comparable` trước khi dùng để so sánh.
4. Chạy `tools.benchmark_inference` với 20 warm-up và 100 ảnh validation cho
   từng model, lưu `metrics/latency_validation.json`.
5. Nếu đưa ảnh lên báo cáo/GitHub, dùng cùng image ID test cho hai model và ghi
   checkpoint, TTA và threshold trong README cạnh prediction.

Các lệnh cụ thể nằm trong README gốc. Không dùng test để chọn lại checkpoint
hoặc threshold; evaluation test chỉ được chạy sau khi candidate đã chốt.
