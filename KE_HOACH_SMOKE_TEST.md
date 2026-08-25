# Kế hoạch smoke test Faster R-CNN và RetinaNet

## 1. Trạng thái và phạm vi

- Người duyệt: **Nguyễn Thành Long**.
- Trạng thái: **Chờ Long duyệt, chưa chạy smoke test**.
- Mục tiêu: chứng minh toàn bộ pipeline có thể đọc dữ liệu, tạo mô hình, thực hiện forward/backward, tính validation metric, lưu/nạp checkpoint và suy luận trên GPU trước khi huấn luyện chính thức.
- Phần cứng: Intel Core i5-12450H, NVIDIA GeForce RTX 2050 4 GB, RAM 16 GB.
- Môi trường đã kiểm tra: Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, CUDA khả dụng.
- Kết quả kiểm thử hiện tại: 21 test đạt, 2 test được bỏ qua do chưa có checkpoint chính thức/fixture tích hợp.
- Smoke test không dùng tập test và không được dùng để kết luận mô hình nào tốt hơn.

## 2. Đầu vào đã xác nhận

- Dataset processed: 2.392 ảnh, 8.274 bbox.
- Ba lớp: `BikeWithRider`, `NoHelmet`, `Helmet`; ID 1–3, background ID 0.
- Train: 1.673 ảnh, 5.800 bbox.
- Validation: 360 ảnh, 1.236 bbox.
- Test: 359 ảnh, 1.238 bbox.
- Tỷ lệ split: 70/15/15, seed 42.
- Không có image ID giao nhau giữa ba split.
- Không phát hiện ảnh trùng tuyệt đối bằng SHA-256.
- Long đã kiểm tra ảnh minh họa, xác nhận bbox/nhãn nhìn hợp lý và ảnh đánh số liên tiếp không phải các frame cùng một cảnh/video.
- Ảnh có EXIF Orientation được xoay đúng chiều trong dataset loader; kích thước COCO processed dùng cùng hệ tọa độ.

## 3. Mục đích của smoke test

Smoke test chỉ trả lời các câu hỏi kỹ thuật sau:

1. Cấu hình có đọc đúng model, lớp, đường dẫn và thiết bị CUDA không?
2. Dataset loader có trả ảnh và bbox hợp lệ sau transform không?
3. Mỗi mô hình có chạy được một batch train trên RTX 2050 4 GB không?
4. Loss có hữu hạn và backward/optimizer step có hoàn thành không?
5. Mỗi mô hình có chạy được một batch validation và trả đúng schema metric không?
6. Checkpoint smoke có lưu, nạp lại và suy luận được không?
7. Hai mô hình có dùng cùng split, class mapping, kích thước đầu vào và evaluator không?

Giá trị loss, Precision, Recall hoặc mAP từ một batch không có ý nghĩa thống kê và không được đưa vào báo cáo kết quả.

## 4. Các điểm mã nguồn cần hoàn thiện trước khi chạy

### 4.1. Truyền đúng cấu hình pretrained weights

Hiện `src/train.py` tạo model nhưng chưa truyền `config["model"]["weights"]` vào `build_model`. Ngoài ra nhánh Faster R-CNN trong `src/models.py` luôn dùng `DEFAULT` dù hàm có tham số `weights`.

Cần sửa để:

- `DEFAULT` dùng trọng số pretrained COCO;
- `NONE` tắt tải trọng số để kiểm thử ngoại tuyến khi cần;
- Faster R-CNN và RetinaNet xử lý hai lựa chọn nhất quán;
- giá trị không hợp lệ phải báo lỗi rõ ràng;
- bổ sung unit test chứng minh YAML thực sự điều khiển lựa chọn weights.

Baseline chính thức vẫn dự kiến dùng `DEFAULT` cho cả hai mô hình. Smoke test đầu tiên cũng dùng `DEFAULT` để kiểm tra đúng đường khởi tạo sẽ dùng khi fine-tune.

### 4.2. Tách artifact smoke khỏi huấn luyện chính thức

Chế độ `--smoke-test` hiện chạy một batch nhưng không lưu checkpoint/history. Cần cho phép lưu artifact vào thư mục riêng:

```text
outputs/smoke/
├── faster_rcnn/
│   ├── checkpoint.pth
│   ├── history.json
│   ├── run_manifest.json
│   ├── environment.json
│   └── smoke.log
└── retinanet/
    ├── checkpoint.pth
    ├── history.json
    ├── run_manifest.json
    ├── environment.json
    └── smoke.log
```

Quy tắc:

- không ghi vào `outputs/<model>/checkpoints/best_map.pth` hoặc `last.pth` của lần train chính thức;
- checkpoint smoke phải ghi rõ `smoke_test: true`;
- không được resume lần train chính thức từ checkpoint smoke;
- checkpoint và log sinh ra không đưa lên GitHub.

### 4.3. Ghi thông tin tài nguyên tối thiểu

Mỗi lần smoke cần ghi:

- tên model và weights;
- Python, PyTorch, Torchvision và CUDA;
- GPU và VRAM;
- Git commit;
- seed, split hash và config;
- AMP bật/tắt;
- train loss, validation metric schema;
- peak CUDA memory allocated/reserved;
- thời gian chạy và trạng thái đạt/rớt.

### 4.4. Kiểm tra nạp checkpoint và inference

Sau khi lưu checkpoint smoke:

1. Tạo lại đúng model từ metadata.
2. Nạp `model_state_dict`.
3. Chạy inference một ảnh validation.
4. Kiểm tra đầu ra có `boxes`, `labels`, `scores`.
5. Kiểm tra tensor hữu hạn; label nằm trong 1–3; box có dạng `[x1, y1, x2, y2]`.

Không yêu cầu prediction đúng ở giai đoạn này vì model mới chỉ cập nhật một batch.

## 5. Cổng 0 - Đóng băng phiên bản dữ liệu

Trước khi gọi model:

1. Tính SHA-256 cho:
   - `data/processed/edgevision/annotations.json`;
   - `data/processed/edgevision/annotation_changes.json`;
   - `data/processed/edgevision/image_hashes.json`;
   - `data/splits/train.json`;
   - `data/splits/val.json`;
   - `data/splits/test.json`.
2. Ghi seed, tỷ lệ, số ảnh/bbox và các hash vào frozen manifest.
3. Chạy lại kiểm tra không giao nhau về image ID/group ID.
4. Kiểm tra mỗi split không có bbox hỏng/vượt biên và đủ cả ba lớp.
5. Không tạo lại split sau cổng này trừ khi phát hiện bug dữ liệu; nếu thay đổi phải hủy toàn bộ kết quả smoke/train cũ.

Đầu ra cục bộ dự kiến: `data/splits/frozen_manifest.json`.

## 6. Cổng 1 - Preflight trước GPU

### 6.1. Kiểm tra môi trường

Chạy và lưu kết quả:

```powershell
python tools/check_environment.py --output outputs/smoke/environment.json
python -m pytest -q
```

Điều kiện đạt:

- Python 3.11.x;
- PyTorch 2.5.1 và Torchvision 0.20.1;
- `cuda_available=true`;
- GPU là RTX 2050, VRAM khoảng 4 GB;
- test liên quan config, model, dataset, transform, metric và checkpoint đều đạt.

### 6.2. Kiểm tra một batch dữ liệu

- Đọc một batch train và một batch validation, batch size 1.
- Ảnh là tensor RGB hữu hạn.
- Bbox nằm trong kích thước ảnh sau transform.
- Label chỉ thuộc 1–3.
- Số boxes bằng số labels/area/iscrowd.
- Không đọc `test.json`.

## 7. Cấu hình smoke ban đầu

Hai model phải dùng chung các điều kiện sau:

| Thuộc tính | Giá trị smoke ban đầu |
|---|---|
| Device | CUDA |
| Batch size | 1 |
| Train batches | 1 |
| Validation batches | 1 |
| Seed | 42 |
| AMP | Bật |
| `min_size` | 512 |
| `max_size` | 768 |
| Trainable backbone layers | 3 |
| Weights | DEFAULT |
| Optimizer | SGD |
| Learning rate | 0,0025 |
| Gradient accumulation | 1 trong smoke |
| Test split | Không sử dụng |

Gradient accumulation 2 được xem xét cho cấu hình train chính thức để đạt effective batch size 2; smoke một batch chỉ kiểm tra tính đúng của pipeline nên chưa dùng accumulation để suy diễn kết quả.

## 8. Thứ tự thực hiện

### Bước 1 - Chạy Faster R-CNN

Lệnh mục tiêu:

```powershell
python -m src.train --config configs/faster_rcnn.yaml --smoke-test --device cuda
```

Kiểm tra:

- model được tạo bằng `fasterrcnn_resnet50_fpn_v2`, 4 lớp gồm background;
- weights thực tế là `DEFAULT`;
- một train batch hoàn thành forward, loss hữu hạn, backward và optimizer step;
- một validation batch trả đủ metric schema;
- checkpoint smoke được lưu và nạp lại;
- inference một ảnh trả boxes/labels/scores hợp lệ;
- peak VRAM được ghi lại.

Sau khi xong phải giải phóng model, gọi garbage collection và `torch.cuda.empty_cache()` trước model tiếp theo.

### Bước 2 - Chạy RetinaNet

Lệnh mục tiêu:

```powershell
python -m src.train --config configs/retinanet.yaml --smoke-test --device cuda
```

Thực hiện cùng các kiểm tra như Faster R-CNN, đồng thời xác nhận classification head RetinaNet dùng 4 lớp và đúng số anchors.

### Bước 3 - So sánh tính nhất quán, không so sánh chất lượng

Đối chiếu hai manifest:

- cùng dataset/split hash;
- cùng seed, class mapping, kích thước ảnh và batch size;
- cùng evaluator và IoU threshold;
- đều dùng CUDA và AMP;
- đều hoàn thành checkpoint round-trip và inference.

Không so sánh smoke mAP, loss hoặc tốc độ để chọn model.

## 9. Tiêu chí đạt/rớt

### 9.1. Đạt

Một model chỉ đạt smoke test khi toàn bộ điều kiện sau đúng:

- [ ] Tạo model và head 4 lớp thành công.
- [ ] Đọc batch train/validation từ split đã đóng băng.
- [ ] Chạy trên CUDA, AMP theo cấu hình.
- [ ] Forward train trả dictionary loss.
- [ ] Tổng loss hữu hạn và lớn hơn hoặc bằng 0.
- [ ] Backward và optimizer step hoàn thành.
- [ ] Validation trả prediction đúng schema.
- [ ] Metric có `map_50_95`, `map_50` và thông tin theo lớp theo schema evaluator hiện tại.
- [ ] Checkpoint smoke lưu và nạp lại được.
- [ ] Inference sau reload trả tensor hữu hạn, label hợp lệ.
- [ ] Không truy cập tập test.
- [ ] Log, manifest và peak VRAM được lưu.

Chỉ chuyển sang huấn luyện chính thức khi **cả hai model** đều đạt trên cùng commit.

### 9.2. Rớt

Smoke test rớt nếu gặp một trong các trường hợp:

- CUDA out of memory;
- loss NaN/Inf;
- bbox/label lỗi sau transform;
- model head sai số lớp;
- validation hoặc evaluator lỗi;
- checkpoint không nạp lại được;
- inference thiếu boxes/labels/scores;
- hai model dùng khác split hoặc khác class mapping;
- phát hiện mã đọc tập test trong quá trình smoke.

Lỗi phải được lưu kèm model, bước xảy ra, stack trace, cấu hình và peak VRAM; không chỉ chạy lại nhiều lần mà không xác định nguyên nhân.

## 10. Quy trình xử lý OOM trên RTX 2050 4 GB

Chạy hai model tuần tự, không mở ứng dụng GPU nặng trong lúc smoke. Khi OOM:

1. Dừng model hiện tại, ghi lại peak VRAM và vị trí lỗi.
2. Giải phóng model/optimizer/scaler, garbage collection và CUDA cache.
3. Chạy lại đúng một lần để loại trừ bộ nhớ còn sót từ tiến trình trước.
4. Nếu vẫn OOM, đổi **cả hai model** sang `min_size=448`, `max_size=672` và chạy lại từ đầu.
5. Nếu vẫn OOM, giảm `trainable_backbone_layers` từ 3 xuống 2 cho **cả hai model**, ghi rõ đây là cấu hình giảm bộ nhớ.
6. Nếu vẫn OOM, dừng và báo Long; không coi chạy CPU là bằng chứng pipeline GPU đã đạt.

Không giảm kích thước hoặc backbone riêng một model rồi dùng cấu hình khác nhau cho so sánh chính thức.

## 11. Kiểm thử cần bổ sung

- Config `weights=DEFAULT/NONE` được truyền đúng cho cả hai model.
- Faster R-CNN `NONE` không tải COCO/ImageNet weights ngoài ý muốn.
- Smoke checkpoint có cờ `smoke_test=true` và không được resume thành official run.
- Smoke output không ghi đè best/last checkpoint chính thức.
- Dataset integration test đọc được một ảnh EXIF Orientation và bbox đúng.
- Frozen manifest phát hiện split hoặc processed annotation bị thay đổi.
- Checkpoint reload inference trả schema hợp lệ.
- Test split không xuất hiện trong luồng `run_training`.

Sau khi bổ sung, toàn bộ test hiện có và test mới phải đạt trước khi chạy GPU.

## 12. Artifact bàn giao

Đưa lên GitHub:

- mã sửa model weights và smoke artifact;
- test tự động;
- cấu hình smoke nếu cần;
- tài liệu hướng dẫn lệnh;
- schema manifest, không chứa số liệu bịa đặt.

Chỉ lưu cục bộ, không đưa lên GitHub:

- dataset;
- split JSON và annotation processed theo quy tắc hiện tại;
- checkpoint `.pth`;
- log/metric/prediction generated;
- environment và cache.

Bản tóm tắt sau smoke cần ghi:

| Model | CUDA | AMP | Train batch | Loss hữu hạn | Val batch | Checkpoint reload | Inference schema | Peak VRAM | Kết quả |
|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |
| RetinaNet | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |

## 13. Thời gian dự kiến

- Hoàn thiện mã và test trước smoke: 30–60 phút.
- Tải pretrained weights lần đầu: phụ thuộc kết nối mạng.
- Smoke Faster R-CNN: khoảng 5–15 phút, chưa cam kết trước khi đo.
- Smoke RetinaNet: khoảng 5–15 phút, chưa cam kết trước khi đo.
- Sửa một lỗi/OOM và chạy lại cả hai: 30–60 phút tùy nguyên nhân.

Thời gian trên là ước lượng vận hành, không phải kết quả benchmark.

## 14. Cổng duyệt của Long

Long cần duyệt các điểm sau trước khi bắt đầu:

1. Đồng ý đóng băng split hiện tại sau khi đã xác nhận ảnh liên tiếp không cùng cảnh/video.
2. Đồng ý smoke đầu tiên dùng pretrained weights `DEFAULT` cho cả hai model.
3. Đồng ý lưu checkpoint smoke riêng, không dùng để resume train chính thức.
4. Đồng ý quy trình OOM: 512/768 → 448/672 → giảm backbone layers đồng đều nếu cần.
5. Đồng ý chỉ chuyển sang huấn luyện chính thức khi cả Faster R-CNN và RetinaNet đạt smoke trên cùng commit.

Sau khi Long duyệt, bước đầu tiên là sửa và kiểm thử bốn điểm mã nguồn tại Mục 4; chưa chạy GPU cho đến khi preflight đạt.
