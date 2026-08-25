# Kế hoạch pilot training Faster R-CNN và RetinaNet

## 1. Trạng thái và phạm vi

- Người phụ trách duyệt: **Nguyễn Thành Long**.
- Trạng thái: **Long đã duyệt; đang hoàn thiện mã/config và preflight, chưa chạy pilot training**.
- Thời điểm lập kế hoạch: 26/08/2026.
- Hạn nội bộ: hoàn thành huấn luyện và đánh giá trước 03/09/2026 để nhóm còn thời gian viết báo cáo.
- Mục tiêu: huấn luyện thử có kiểm soát trong 3 epoch cho cả Faster R-CNN và RetinaNet nhằm xác nhận mô hình thực sự học được, ước lượng thời gian chạy và phát hiện lỗi trước khi chạy baseline chính thức.
- Pilot chỉ dùng `train` để cập nhật trọng số và `validation` để theo dõi. **Không đọc tập test**.
- Kết quả pilot chưa phải kết quả cuối cùng và không dùng để kết luận mô hình nào tốt hơn.

## 2. Điều kiện đầu vào đã đạt

- Smoke test Faster R-CNN và RetinaNet đã chạy thành công trên CUDA.
- Hai mô hình đã hoàn thành forward, backward, validation, lưu checkpoint, nạp lại checkpoint và inference.
- Phần cứng: Intel Core i5-12450H, NVIDIA GeForce RTX 2050 4 GB, RAM 16 GB.
- Môi trường: Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121.
- Bộ test hiện tại: 27 test đạt, 2 test được bỏ qua đúng lý do đã biết.
- Dataset processed: 2.392 ảnh, 8.274 bounding box.
- Lớp nhãn: `BikeWithRider`, `NoHelmet`, `Helmet`; background có ID 0.
- Split đã đóng băng với seed 42:
  - train: 1.673 ảnh, 5.800 bounding box;
  - validation: 360 ảnh, 1.236 bounding box;
  - test: 359 ảnh, 1.238 bounding box.
- `data/splits/frozen_manifest.json` đã được tạo và xác nhận các split không giao nhau.

## 3. Câu hỏi pilot cần trả lời

Pilot training phải cung cấp bằng chứng cho các câu hỏi sau:

1. Mỗi mô hình có thể chạy liên tục qua toàn bộ tập train và validation mà không lỗi/OOM không?
2. Training loss có hữu hạn và có xu hướng giảm qua 3 epoch không?
3. `mAP@0.5` và `mAP@0.5:0.95` trên validation có được tính ổn định không?
4. Checkpoint tốt nhất và checkpoint cuối có được lưu, nạp lại và resume đúng không?
5. Một epoch mất bao lâu và peak VRAM là bao nhiêu trên RTX 2050?
6. Cấu hình 20 epoch hiện tại có khả thi trước hạn nội bộ không?
7. Có dấu hiệu bất thường về dữ liệu, nhãn hoặc prediction cần xử lý trước baseline chính thức không?

## 4. Nguyên tắc so sánh công bằng

Hai mô hình chỉ khác kiến trúc. Các điều kiện sau phải giữ giống nhau:

- cùng phiên bản dataset và frozen split;
- cùng ba lớp nhãn và cách ánh xạ ID;
- cùng seed 42;
- cùng kích thước ảnh `min_size=512`, `max_size=768`;
- cùng augmentation đã khai báo;
- cùng batch size 1;
- cùng pretrained weights `DEFAULT` từ Torchvision;
- cùng số layer backbone được fine-tune là 3;
- cùng optimizer SGD, learning rate, momentum và weight decay;
- cùng AMP trên CUDA;
- cùng evaluator, IoU threshold và quy tắc tính metric;
- cùng 3 epoch và validation sau mỗi epoch;
- cùng điều kiện máy, không chạy đồng thời hai mô hình.

Không điều chỉnh tham số dựa trên tập test. Nếu phải thay cấu hình vì lỗi kỹ thuật, phải áp dụng lại cho **cả hai mô hình** hoặc ghi rõ đây không còn là phép so sánh tương đương.

## 5. Cấu hình pilot dự kiến

| Thuộc tính | Giá trị |
|---|---:|
| Epoch | 3 |
| Batch size | 1 |
| Gradient accumulation | 1 |
| Optimizer | SGD |
| Learning rate | 0,0025 |
| Momentum | 0,9 |
| Weight decay | 0,0005 |
| Scheduler | StepLR |
| Scheduler step size | 7 |
| Scheduler gamma | 0,1 |
| Weights | DEFAULT |
| Trainable backbone layers | 3 |
| Kích thước ảnh | 512–768 |
| AMP | Bật |
| Num workers | 2 |
| Validation | Sau mỗi epoch |
| Primary metric | `mAP@0.5:0.95` |
| Metric bổ sung | `mAP@0.5`, `mAP@0.75`, Recall, AP theo lớp |
| Test split | Không sử dụng |

Vì scheduler có step size 7, learning rate chưa giảm trong pilot 3 epoch. Pilot chủ yếu kiểm tra độ ổn định và khả năng học ban đầu, không dùng để lựa chọn lịch learning rate tối ưu.

## 6. Mã nguồn cần hoàn thiện trước khi chạy

### 6.1. Tạo cấu hình pilot riêng

Tạo hai tệp:

```text
configs/pilot_faster_rcnn.yaml
configs/pilot_retinanet.yaml
```

Hai tệp dùng 3 epoch và ghi output vào thư mục riêng. Không sửa hoặc ghi đè cấu hình/checkpoint baseline 20 epoch.

### 6.2. Tách artifact pilot

Cấu trúc dự kiến:

```text
outputs/pilot/
├── faster_rcnn/
│   ├── checkpoints/
│   │   ├── best_map.pth
│   │   └── last.pth
│   ├── logs/
│   │   └── history.json
│   ├── run_manifest.json
│   ├── environment.json
│   └── train.log
└── retinanet/
    ├── checkpoints/
    │   ├── best_map.pth
    │   └── last.pth
    ├── logs/
    │   └── history.json
    ├── run_manifest.json
    ├── environment.json
    └── train.log
```

Dataset, checkpoint, log và kết quả sinh ra chỉ lưu cục bộ, không commit lên GitHub.

### 6.3. Bổ sung thông tin theo dõi lần chạy

Pipeline cần ghi tối thiểu:

- loại lần chạy: `pilot`;
- Git commit và thời gian bắt đầu/kết thúc;
- config đầy đủ và hash frozen split;
- Python, PyTorch, Torchvision, CUDA và GPU;
- thời gian từng epoch và tổng thời gian;
- learning rate từng epoch;
- train loss từng epoch;
- metric validation từng epoch;
- peak CUDA memory allocated/reserved;
- đường dẫn và SHA-256 của checkpoint;
- trạng thái hoàn thành, bị ngắt hoặc lỗi.

### 6.4. Thêm log tiến độ

Do một epoch gồm 1.673 batch, cần in tiến độ định kỳ, dự kiến mỗi 50 batch:

- batch hiện tại/tổng batch;
- loss trung bình tạm thời;
- learning rate;
- thời gian đã chạy;
- peak VRAM.

Log tiến độ chỉ phục vụ vận hành, không dùng các giá trị giữa epoch để so sánh chất lượng.

### 6.5. Bảo vệ checkpoint

- Không cho phép resume pilot/baseline từ checkpoint có `smoke_test=true`.
- Checkpoint pilot ghi `run_type=pilot`.
- Chỉ resume đúng model và đúng cấu hình lớp.
- `best_map.pth` chọn theo validation `mAP@0.5:0.95`.
- `last.pth` phải được ghi sau mỗi epoch để có thể tiếp tục nếu tiến trình bị gián đoạn.

## 7. Cổng 0 - Preflight

Trước khi sử dụng GPU:

1. Chạy toàn bộ unit test.
2. Kiểm tra frozen split manifest vẫn khớp dữ liệu hiện tại.
3. Kiểm tra CUDA khả dụng và GPU đúng RTX 2050.
4. Xác nhận không có tiến trình khác chiếm nhiều VRAM.
5. Xác nhận đủ dung lượng đĩa cho ít nhất bốn checkpoint lớn và log.
6. Xác nhận thư mục output pilot không trỏ vào output baseline chính thức.
7. Commit và push mã/config đã kiểm thử lên `main` trước khi bắt đầu để manifest có Git commit xác định.

Chỉ bắt đầu khi toàn bộ preflight đạt.

## 8. Thứ tự chạy

### Bước 1 - Pilot Faster R-CNN

Lệnh mục tiêu:

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m src.train --config configs\pilot_faster_rcnn.yaml --device cuda
```

Sau mỗi epoch kiểm tra:

- loss và metric đều hữu hạn;
- checkpoint `last.pth` tồn tại;
- nếu metric tốt hơn thì `best_map.pth` được cập nhật;
- validation dùng đủ 360 ảnh;
- không có truy cập vào test split;
- thời gian và peak VRAM được ghi.

Sau epoch 3, nạp `best_map.pth` và chạy inference trên một số ảnh validation để kiểm tra định tính. Ảnh minh họa này chỉ phục vụ phát hiện lỗi, không thay thế đánh giá định lượng.

### Bước 2 - Cổng duyệt giữa hai mô hình

Chỉ chuyển sang RetinaNet khi Faster R-CNN:

- hoàn thành đủ 3 epoch hoặc có lỗi đã xác định rõ;
- không có loss NaN/Inf;
- checkpoint nạp lại được;
- lịch sử và manifest đầy đủ;
- thời gian ước lượng cho bước tiếp theo không đe dọa hạn 03/09.

Nếu Faster R-CNN bị lỗi cấu hình chung, sửa và chạy lại từ đầu trước khi dùng cùng bản sửa cho RetinaNet.

### Bước 3 - Pilot RetinaNet

Lệnh mục tiêu:

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m src.train --config configs\pilot_retinanet.yaml --device cuda
```

Áp dụng đúng các kiểm tra và tiêu chí như Faster R-CNN.

### Bước 4 - Đối chiếu pilot

Tạo bảng nội bộ:

| Model | Epoch hoàn thành | Loss cuối | mAP@0.5 | mAP@0.5:0.95 | Recall | Thời gian/epoch | Peak VRAM | Reload checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Faster R-CNN | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |
| RetinaNet | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |

Bảng này dùng để quyết định pipeline đã sẵn sàng và ước lượng thời gian. Không viết kết luận học thuật rằng một mô hình tốt hơn từ pilot 3 epoch.

## 9. Tiêu chí đạt pilot

Một mô hình đạt khi:

- [ ] hoàn thành 3 epoch train và 3 lượt validation;
- [ ] mỗi lượt validation đánh giá đủ 360 ảnh;
- [ ] loss và metric không có NaN/Inf;
- [ ] loss không tăng mất kiểm soát qua các epoch;
- [ ] prediction không rỗng toàn bộ validation;
- [ ] checkpoint best/last tồn tại và nạp lại được;
- [ ] checkpoint chứa đúng model, số lớp, config và loại lần chạy;
- [ ] history, environment và run manifest được lưu đầy đủ;
- [ ] không truy cập hoặc điều chỉnh theo tập test;
- [ ] không xảy ra OOM hoặc lỗi dữ liệu chưa xử lý.

`mAP=0` ở epoch đầu không tự động làm pilot thất bại. Nếu metric vẫn bằng 0 sau epoch 3, cần kiểm tra prediction, confidence, label mapping, bbox và evaluator trước khi quyết định chạy 20 epoch.

## 10. Quy tắc dừng sớm để bảo vệ thời gian

Dừng lần chạy và điều tra ngay nếu gặp một trong các trường hợp:

- CUDA OOM lặp lại;
- loss NaN/Inf;
- loss tăng đột biến liên tục;
- bbox hoặc label ngoài miền hợp lệ;
- validation không đủ 360 ảnh;
- metric schema thiếu hoặc có giá trị không hữu hạn;
- checkpoint không lưu/nạp lại được;
- tốc độ thực tế cho thấy không thể hoàn thành cả hai model trước hạn;
- phát hiện pipeline đọc tập test.

Không dừng chỉ vì mAP của epoch đầu thấp hoặc bằng 0.

## 11. Xử lý lỗi và OOM

### 11.1. OOM

1. Ghi model, epoch, batch và peak VRAM tại thời điểm lỗi.
2. Kết thúc tiến trình để giải phóng toàn bộ CUDA memory.
3. Chạy lại đúng một lần để loại trừ trạng thái bộ nhớ còn sót.
4. Nếu vẫn OOM, giảm đồng đều hai model xuống `min_size=448`, `max_size=672`.
5. Nếu vẫn OOM, giảm `trainable_backbone_layers` từ 3 xuống 2 cho cả hai model.
6. Chạy lại pilot từ đầu với config mới và ghi rõ lý do thay đổi.

Không giảm cấu hình riêng cho một model rồi dùng số liệu đó làm phép so sánh trực tiếp.

### 11.2. Tiến trình bị ngắt

- Nếu checkpoint `last.pth` hợp lệ, resume từ epoch tiếp theo.
- Nếu checkpoint đang ghi dở hoặc hash không khớp, không dùng checkpoint đó.
- Manifest phải ghi rõ lần chạy được resume và checkpoint nguồn.

### 11.3. Metric bất thường

Nếu mAP vẫn bằng 0 sau 3 epoch:

1. Kiểm tra ảnh validation có ground truth đúng không.
2. Vẽ prediction trước và sau confidence filtering.
3. Kiểm tra label ID của prediction và ground truth.
4. Kiểm tra box format `xyxy`, kích thước ảnh và phép biến đổi EXIF.
5. Đối chiếu evaluator với một ví dụ tổng hợp đã biết đáp án.
6. Chỉ thay hyperparameter sau khi loại trừ lỗi pipeline.

## 12. Đầu ra cần bàn giao sau pilot

- Bảng tóm tắt 3 epoch của hai mô hình.
- Lịch sử train loss và validation metric theo epoch.
- Thời gian/epoch, tổng thời gian và peak VRAM.
- Best/last checkpoint cục bộ của từng mô hình.
- Manifest chứng minh cùng split/config/evaluator.
- Một số ảnh validation có ground truth và prediction để kiểm tra định tính.
- Danh sách lỗi/cảnh báo và cách xử lý.
- Khuyến nghị có căn cứ về việc giữ hay điều chỉnh cấu hình trước baseline 20 epoch.

Không đưa checkpoint, dataset hoặc log lớn lên GitHub. Chỉ commit mã nguồn, config, test và tài liệu hướng dẫn.

## 13. Điều kiện chuyển sang baseline chính thức

Chỉ chạy baseline 20 epoch khi:

1. Cả hai pilot đạt tiêu chí tại Mục 9.
2. Không còn lỗi pipeline/dataset/evaluator chưa giải quyết.
3. Thời gian thực tế cho thấy có thể hoàn thành trước 03/09.
4. Cấu hình cuối được áp dụng công bằng cho cả hai mô hình.
5. Long duyệt kết quả pilot và cấu hình baseline.

Sau baseline mới dùng `best_map.pth` của mỗi mô hình để đánh giá **một lần** trên test split, đo latency/FPS và cung cấp số liệu chính thức cho báo cáo.

## 14. Các điểm Long cần duyệt

1. Đồng ý pilot 3 epoch cho mỗi mô hình, chạy tuần tự Faster R-CNN rồi RetinaNet.
2. Đồng ý giữ cấu hình 512–768, batch size 1, AMP, pretrained `DEFAULT` và 3 backbone layers như smoke test.
3. Đồng ý tách hoàn toàn output pilot khỏi checkpoint baseline.
4. Đồng ý bổ sung log tiến độ, thời gian, peak VRAM và run manifest trước khi chạy.
5. Đồng ý không dùng test split và không dùng kết quả pilot để kết luận mô hình tốt hơn.
6. Đồng ý quy trình OOM áp dụng cùng thay đổi cho cả hai model.
7. Đồng ý chỉ chuyển sang baseline 20 epoch sau khi Long xem kết quả pilot.

Sau khi Long duyệt, bước đầu tiên là hoàn thiện mã/config/test tại Mục 6, chạy preflight và push mã lên `main`; chưa khởi chạy GPU cho đến khi preflight đạt.
