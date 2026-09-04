# Công cụ dự án (`tools/`)

## 1. Vai trò của thư mục này

`tools/` chứa các chương trình hỗ trợ bên ngoài pipeline chính ở `src/`. Chúng không thay thế Faster R-CNN, RetinaNet hay các metric chính; nhiệm vụ là chuẩn bị dữ liệu, khóa giao thức, kiểm tra chất lượng, benchmark, tạo tài liệu và kiểm thử demo.

```text
Raw COCO + ảnh
  │
  ├─ inspect_dataset.py → phát hiện lỗi dữ liệu
  ├─ prepare_annotations.py → COCO processed + log thay đổi
  ├─ build_groups.py → nhóm ảnh trùng/gần trùng
  ├─ create_splits.py → train / val / test theo group
  └─ freeze_splits.py → manifest + SHA-256 để khóa đầu vào train
                                      │
                                      ▼
                           src.train / src.evaluate
                                      │
      ┌───────────────────────────────┼──────────────────────────────┐
      ▼                               ▼                              ▼
benchmark_*.py              check_deployment_gate.py      generate_report_*.py
đo tốc độ                   kiểm tra candidate            hình/bảng báo cáo
```

## 2. Nguyên tắc dùng tool

- Chạy lệnh từ **root dự án** để import được `src.*` và các đường dẫn tương đối được hiểu đúng.
- Đọc `--help` trước khi chạy một tool chưa quen, ví dụ: `python tools/inspect_dataset.py --help`.
- Các tool tạo artifact phải ghi ra thư mục mới hoặc đường dẫn output được chỉ định; không sửa `raw` annotation mặc định.
- Không dùng `test` để chọn checkpoint, threshold hoặc candidate. Các tool có liên quan đến việc ra quyết định chỉ dùng validation hoặc split được chỉ định rõ là không phải test.
- JSON kết quả cần giữ cùng checkpoint hash, annotation hash và cấu hình để truy vết. Artifact, checkpoint và dữ liệu lớn không đưa vào Git nếu `.gitignore` đã loại trừ chúng.
- Một số tool tạo contact sheet hoặc review queue chỉ phục vụ con người kiểm tra. Chúng không tự biến pseudo-label thành ground truth.

## 3. Bảng tra nhanh

| Tool | Mục đích | Đầu vào chính | Kết quả tạo ra |
|---|---|---|---|
| `setup_environment.ps1` | Tạo môi trường Python và cài dependency. | Chế độ CPU/GPU. | `.venv/` và gói cần thiết. |
| `check_environment.py` | Ghi nhận môi trường chạy. | Python hiện tại. | JSON/console: Python, Torch, CUDA, GPU, package. |
| `inspect_dataset.py` | Kiểm định cấu trúc COCO và file ảnh. | COCO JSON, thư mục ảnh. | Báo cáo thống kê/lỗi dữ liệu. |
| `prepare_annotations.py` | Làm sạch có lưu vết. | Raw COCO, tùy chọn thư mục ảnh. | COCO processed, change log, problem list. |
| `build_groups.py` | Phát hiện ảnh trùng/gần trùng. | COCO processed, ảnh. | Group manifest, danh sách near-duplicate. |
| `create_splits.py` | Chia train/val/test theo group. | COCO, group manifest, seed/tỉ lệ. | Ba COCO split và manifest. |
| `freeze_splits.py` | Khóa split trước train. | Processed COCO, ba split. | Frozen manifest có SHA-256 và kiểm tra leak. |
| `visualize_annotations.py` | QA trực quan annotation. | COCO, thư mục ảnh. | Ảnh mẫu đã vẽ box/label. |
| `build_challenge_contact_sheet.py` | Duyệt nhanh ảnh challenge. | Thư mục ảnh. | Một ảnh contact sheet. |
| `dedupe_challenge_images.py` | Tìm ảnh challenge trùng/gần trùng. | Thư mục ảnh. | JSON exact/near duplicates. |
| `select_edgevision_hard_subset.py` | Chọn ảnh khó bằng quy tắc dữ liệu. | EdgeVision test + ảnh. | COCO subset, CSV/JSON difficulty. |
| `collect_wikimedia_challenge_images.py` | Thu thập ảnh Wikimedia có provenance. | Query, giới hạn, output. | Ảnh và manifest URL/tác giả/giấy phép/hash. |
| `benchmark_inference.py` | Benchmark end-to-end inference. | Config, checkpoint, validation. | JSON latency/FPS kèm protocol. |
| `benchmark_speed.py` | Benchmark model-forward thuần. | Config, checkpoint, val/test. | JSON latency/FPS kèm protocol. |
| `check_deployment_gate.py` | Quyết định candidate có qua gate không. | History/metric baseline và candidate. | JSON/pass-fail có lý do. |
| `evaluate_fusion.py` | Đánh giá ensemble/TTA. | Hai checkpoint/config, validation. | JSON metric fusion và provenance. |
| `profile_validation_thresholds.py` | Khảo sát threshold ngoài tập test. | Checkpoint, annotation non-test. | JSON Precision/Recall/F1 theo ngưỡng. |
| `audit_rider_association.py` | Audit quy tắc ghép đầu–xe trên ground truth. | COCO split. | JSON thống kê ghép hình học. |
| `create_role_dev_tasks.py` | Tạo task review vai trò thủ công. | COCO annotation. | JSON task + manifest/hash. |
| `build_role_dev_contact_sheet.py` | Vẽ crop để người review role. | JSON task, ảnh. | Các trang contact sheet. |
| `evaluate_role_association.py` | Chẩn đoán heuristic role. | Role tasks đã review. | JSON candidate precision/recall và limitation. |
| `generate_report_figures.py` | Sinh hình SVG cho báo cáo. | JSON metric/manifest. | SVG biểu đồ metric, latency, train. |
| `generate_report_png_charts.ps1` | Sinh biểu đồ PNG cho báo cáo. | Artifact thí nghiệm. | PNG chart. |
| `build_long_2_1_2_docx.py` | Dựng mục 2.1.2 của báo cáo Word. | Artifact đã kiểm chứng, figure. | DOCX định dạng sẵn. |
| `run_local_demo_image_batch.py` | Chạy hàng loạt ảnh qua API local. | Thư mục ảnh, API đang chạy. | PNG kết quả và `results.json`. |

## 4. Chi tiết từng nhóm công cụ

### 4.1. Môi trường và khả năng tái lập

#### `setup_environment.ps1`

Tạo virtual environment `.venv` và cài dependency theo lựa chọn `-InstallMode cpu` hoặc `-InstallMode gpu`. Đây là tool **có thay đổi môi trường**: chỉ chạy khi cần tạo/cài lại môi trường.

Ví dụ:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_environment.ps1 -InstallMode gpu
```

**Đầu ra:** `.venv/` chứa Python và dependency độc lập với Python hệ thống. Sau đó dùng `.\.venv\Scripts\python.exe` để bảo đảm đúng môi trường.

#### `check_environment.py`

Đây là tool chỉ đọc. Nó thu thập phiên bản Python, PyTorch, Torchvision, CUDA, GPU và các package liên quan. Mục đích là ghi lại môi trường thực nghiệm thay vì chỉ nói “chạy bằng GPU”.

```powershell
.\.venv\Scripts\python.exe .\tools\check_environment.py
```

**Đầu ra:** JSON trên console hoặc tệp output nếu truyền tham số tương ứng. Khi báo cáo benchmark, nên lưu kết quả tool này cùng artifact benchmark.

### 4.2. Kiểm tra, làm sạch và khóa dữ liệu

#### `inspect_dataset.py`

Kiểm tra COCO JSON và ảnh trước khi train. Tool rà các lỗi như thiếu khóa COCO, id bị trùng, annotation trỏ tới ảnh/category không tồn tại, box không hợp lệ, file ảnh bị thiếu/hỏng hoặc kích thước ảnh không khớp metadata.

**Thuật toán xử lý:** đọc các danh sách `images`, `annotations`, `categories`; lập chỉ mục theo id; kiểm tra quan hệ tham chiếu; sau đó mở mẫu/tất cả ảnh theo lựa chọn để xác minh file thực tế.

**Đầu ra:** thống kê số ảnh/annotation/category và danh sách vấn đề. Không sửa annotation. Dùng tool này trước và sau `prepare_annotations.py`.

#### `prepare_annotations.py`

Tạo bản COCO processed thay vì sửa raw dataset. Tool có thể:

- kiểm tra kiểu số của box;
- bỏ annotation không thể sử dụng;
- cắt nhẹ box vượt biên ảnh nếu độ tràn không lớn hơn ngưỡng;
- ghi lỗi thay vì âm thầm sửa trường hợp bất thường;
- đồng bộ width/height hiển thị theo EXIF khi người dùng chủ động bật tùy chọn và cung cấp thư mục ảnh.

**Đầu ra:** ba tệp riêng: processed COCO, `change log` ghi mọi thay đổi và `problem list` cho các trường hợp cần xử lý thủ công. Chỉ processed COCO mới nên đi vào bước chia split.

#### `build_groups.py`

Tạo group ảnh để cùng ảnh hoặc ảnh gần trùng không bị rơi sang các split khác nhau.

- **Trùng chính xác:** tính SHA-256 từng file; file cùng hash vào cùng `group_id`.
- **Gần trùng tùy chọn:** tạo average hash 8×8 grayscale, tính Hamming distance giữa hash. Các cặp dưới ngưỡng chỉ được xuất để review, không tự coi là cùng ảnh nếu chưa có quyết định.

**Đầu ra:** group manifest gán `image_id → group_id` và một danh sách ứng viên near-duplicate. `create_splits.py` dùng manifest này để chống data leakage.

#### `create_splits.py`

Chia COCO thành train, validation và test theo **group**, không theo từng ảnh độc lập. Nhờ vậy các ảnh trùng sẽ không xuất hiện đồng thời ở train và test.

**Thuật toán xử lý:** gom annotation theo group, dùng seed cố định và điểm chênh lệch so với tỉ lệ mục tiêu để phân group vào split. Sau đó tool tự kiểm tra group không giao nhau giữa ba split.

**Đầu ra:** ba COCO JSON và manifest nêu seed, tỉ lệ, số ảnh/annotation/lớp. Khi đã bắt đầu train chính thức, không tự tạo lại split chỉ để cải thiện kết quả.

#### `freeze_splits.py`

“Đóng băng” các đầu vào train/validation/test. Tool đọc processed annotation và ba split, kiểm tra split disjoint, sau đó ghi SHA-256 và thống kê vào frozen manifest.

**Đầu ra:** `frozen_manifest.json`. `src.train` kiểm tra hash của processed/train/val trước khi chạy; nếu file đã đổi sau khi freeze, train bị chặn để artifact cũ không bị hiểu nhầm là thuộc dữ liệu mới.

#### `visualize_annotations.py`

Chọn ngẫu nhiên ảnh theo từng category bằng seed, vẽ box và nhãn lên ảnh. Có thể đưa thêm problem list để ưu tiên vẽ những ảnh cần QA.

**Đầu ra:** thư mục ảnh JPEG đã vẽ annotation. Đây là bước kiểm tra bằng mắt, rất hữu ích để phát hiện nhãn đúng format nhưng sai vị trí.

### 4.3. Challenge set, ảnh khó và provenance

#### `build_challenge_contact_sheet.py`

Ghép các ảnh JPG/JPEG/PNG trong một thư mục thành contact sheet có số thứ tự và tên tệp. Tool resize ảnh để vừa ô, không sửa ảnh gốc.

**Đầu ra:** một JPEG preview. Dùng để chọn nhanh ảnh cần duyệt kỹ hơn.

#### `dedupe_challenge_images.py`

Tìm ảnh trùng exact bằng SHA-256 và gần trùng bằng average hash/Hamming distance trong một thư mục challenge. Tool chỉ đánh dấu cặp nghi ngờ; người dùng vẫn phải xem ảnh trước khi loại.

**Đầu ra:** JSON chứa hash file, các nhóm trùng và cặp gần trùng. Đây là kiểm soát chất lượng challenge set, không phải metric mô hình.

#### `select_edgevision_hard_subset.py`

Tạo tập con ảnh khó từ EdgeVision test **không sử dụng prediction của model**, vì vậy không tạo feedback để chỉnh model theo test. Độ khó được tính từ annotation và ảnh: cảnh đông đối tượng, head nhỏ, box đầu chồng nhau, ảnh tối, box chạm biên.

**Thuật toán chọn:** tính difficulty score theo các tín hiệu trên; sau đó ưu tiên đa dạng tag (`crowded`, `small_object`, `overlapping_boxes`, `low_light`, `edge_truncation`) trước khi chọn các ảnh score cao còn lại.

**Đầu ra:** COCO subset cùng CSV/JSON có tag, score, brightness và SHA-256. Chỉ dùng cho phân tích lỗi sau đánh giá, không để tune checkpoint/threshold.

#### `collect_wikimedia_challenge_images.py`

Tìm và tải ảnh ứng viên từ Wikimedia Commons qua API, kiểm tra ảnh hợp lệ và lưu provenance. Tool giữ URL nguồn, query, tác giả/metadata giấy phép khi có và SHA-256 file tải về.

**Đầu ra:** ảnh challenge và manifest nguồn. Ảnh chỉ được dùng sau khi nhóm kiểm tra nội dung, license và quy tắc sử dụng của môn học.

### 4.4. Đo tốc độ, đánh giá mở rộng và gate triển khai

#### `benchmark_inference.py`

Đo latency/FPS gần với đường đi inference thực tế. Tool dùng ảnh validation đã có nhãn nhưng không đọc nhãn trong lúc predict.

- warm-up một số ảnh để giảm ảnh hưởng khởi tạo GPU;
- thời gian đo gồm chuyển tensor CPU → GPU, forward và NMS nội bộ Torchvision;
- không gồm đọc ảnh từ ổ đĩa hoặc render giao diện;
- đồng bộ CUDA trước/sau đo để thời gian không bị sai do kernel chạy bất đồng bộ.

**Đầu ra:** JSON `inference-benchmark-1.0`: mean/median/p95/min/max latency, FPS, checkpoint hash, annotation hash, số warm-up/số lần đo, thiết bị và phiên bản thư viện.

#### `benchmark_speed.py`

Đo một giao thức hẹp hơn: chỉ `model([tensor])`, tức forward + hậu xử lý Torchvision. Tool nạp ảnh/tensor trước khi bắt đầu đo nên **loại** đọc file, chuyển PIL → tensor và CPU → GPU copy khỏi latency.

Tool có thể đo ảnh từ split COCO cố định hoặc tensor zero synthetic. Synthetic chỉ phục vụ kiểm tra kỹ thuật, không đại diện cảnh giao thông.

**Đầu ra:** JSON có mean/median/p95 latency, FPS, số warm-up/runs, manifest thứ tự input, image shape và checkpoint/config. Không so trực tiếp con số này với `benchmark_inference.py` nếu không nói rõ hai giao thức khác nhau.

#### `check_deployment_gate.py`

Kiểm tra candidate trước khi cho demo dùng checkpoint mới. Tool đọc lịch sử validation, fingerprint annotation COCO theo nội dung và metric baseline/candidate; sau đó quyết định pass/fail theo tiêu chí được truyền vào.

**Đầu ra:** JSON có trạng thái gate, metric đã đối chiếu, khác biệt protocol/fingerprint và lý do pass/fail. Gate dùng validation, không chọn candidate dựa trên test.

#### `evaluate_fusion.py`

Đánh giá TTA hoặc ensemble gồm Faster R-CNN và RetinaNet trên validation. Tool nạp các checkpoint, chạy nguồn prediction, sử dụng `src.prediction_fusion` để hợp nhất box cùng lớp theo weighted box fusion, rồi chấm bằng evaluator chuẩn.

**Đầu ra:** JSON metric fusion kèm đường dẫn/hash checkpoint, thông số IoU fusion, source weights và protocol. Fusion là một hệ suy luận mới, nên phải được đánh giá lại; không được lấy mAP từng checkpoint rồi cộng/trung bình.

#### `profile_validation_thresholds.py`

Khảo sát confidence threshold trên một annotation được truyền rõ ràng. Tool từ chối file có tên chứa `test`; nó chạy model một lần, cache prediction CPU, sau đó tính TP/FP/FN, Precision, Recall, F1 trên lưới threshold.

**Đầu ra:** JSON `external-validation-threshold-profile-1.0` gồm lưới threshold, IoU, checkpoint hash và ngữ cảnh split. Mục đích là chẩn đoán dữ liệu ngoài/validation, không tự ghi ngưỡng triển khai demo.

### 4.5. Association đầu–xe và review vai trò

#### `audit_rider_association.py`

Audit rule hình học trong `src.rider_association` bằng **ground truth COCO**, không chạy detector. Annotation `BikeWithRider`, `Helmet`, `NoHelmet` được đổi thành record có confidence `1.0`, rồi đưa qua hàm ghép đầu–xe.

**Đầu ra:** JSON tổng số xe/đầu được ghép, đầu chưa ghép, trường hợp mơ hồ và thống kê theo ảnh. Kết quả nói lên phạm vi áp dụng của rule hình học, không đo “độ đúng tài xế/người ngồi sau”.

#### `create_role_dev_tasks.py`

Tạo danh sách task cho người review gán role thủ công từ ground truth. Tool chọn các tình huống có đầu và xe liên quan, thêm tag để phân tầng, phân bổ vòng tròn nhằm tránh task đầu tiên chỉ thuộc một kiểu cảnh.

**Đầu ra:** JSON schema `role_association_tasks_v1`, manifest/hash tệp nguồn và các trường review ban đầu để trống. Tool không tự gán nhãn `driver`.

#### `build_role_dev_contact_sheet.py`

Đọc role task và render crop vùng xe/đầu thành các trang contact sheet có mã task. Người review sử dụng các trang này để xem ảnh và ghi quyết định vào JSON task, không gán nhãn chỉ dựa vào tên tệp.

**Đầu ra:** một hoặc nhiều ảnh contact sheet trong output directory.

#### `evaluate_role_association.py`

Gọi `src.role_annotations.validate_role_tasks()` để kiểm tra schema review, sau đó gọi `src.role_evaluation.evaluate_role_candidate_baseline()` để so gợi ý role hình học với review của con người.

**Đầu ra:** JSON chẩn đoán candidate precision/recall, số abstention và giới hạn. Không phải mAP của Faster R-CNN/RetinaNet, không phải kết quả cuối trên role test, không dùng để train detector.

### 4.6. Báo cáo và kiểm thử demo

#### `generate_report_figures.py`

Đọc JSON metric/benchmark/run manifest đã được kiểm chứng và sinh SVG không phụ thuộc thư viện biểu đồ nặng. Các hình gồm so sánh test metric, latency/FPS và đường training.

**Đầu ra:** SVG đặt tại đường dẫn chỉ định. Vì hình đọc từ artifact, khi số liệu đổi phải tạo lại hình thay vì sửa số trực tiếp trong ảnh.

#### `generate_report_png_charts.ps1`

Phiên bản PowerShell để tạo biểu đồ PNG dùng trong báo cáo/trình chiếu. Nó là bước trình bày artifact, không tính lại metric mô hình.

**Đầu ra:** tệp PNG. Kiểm tra nhãn trục, đơn vị, split và chiều tốt hơn của metric trước khi chèn báo cáo.

#### `build_long_2_1_2_docx.py`

Tạo tài liệu Word cho mục 2.1.2 (huấn luyện và đánh giá) từ artifact đã xác minh. Script đặt font, heading, bảng, công thức, hình, caption, số trang và chèn các kết quả đã có provenance.

**Đầu ra:** DOCX hoàn chỉnh để nộp/ghép báo cáo. Tool không được dùng để tạo số liệu mới: nếu artifact nguồn thiếu thì cần bổ sung dữ liệu trước.

#### `run_local_demo_image_batch.py`

Gửi lần lượt mọi ảnh JPG/JPEG/PNG trong thư mục đến API FastAPI cục bộ `POST /api/infer/image`. Mặc định chạy cả `faster_rcnn` và `retinanet`; có thể chỉ định model. API phải được mở trước khi chạy tool.

**Đầu ra:** một PNG kết quả cho mỗi cặp ảnh–model và `results.json` gồm latency, detection, threshold, alert và unknown cases. Tool tạo output directory mới, không ghi đè output đã tồn tại.

Ví dụ:

```powershell
# Cửa sổ 1: chạy API trước
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000

# Cửa sổ 2: chạy batch demo
.\.venv\Scripts\python.exe .\tools\run_local_demo_image_batch.py `
  --input-dir .\data\demo_images `
  --output-dir .\outputs\demo_batch
```

## 5. Thứ tự chạy đề xuất

### Lần đầu chuẩn bị dữ liệu

1. `setup_environment.ps1` → `check_environment.py`.
2. `inspect_dataset.py` trên raw COCO.
3. `prepare_annotations.py` để sinh processed COCO và đọc problem list.
4. `inspect_dataset.py` + `visualize_annotations.py` trên processed COCO.
5. `build_groups.py` → review near-duplicate nếu có.
6. `create_splits.py` → `freeze_splits.py`.
7. Chỉ sau đó mới chạy `src.train`.

### Sau khi có checkpoint

1. `src.evaluate` tạo JSON metric trên split đã định.
2. `benchmark_inference.py` hoặc `benchmark_speed.py`, chọn một giao thức và ghi rõ giao thức đó.
3. `check_deployment_gate.py` trên validation trước khi thay checkpoint demo.
4. `run_local_demo_image_batch.py` để QA trực quan; nếu cần review role thì dùng nhóm tool association.
5. `generate_report_figures.py`/`generate_report_png_charts.ps1` rồi mới `build_long_2_1_2_docx.py`.

## 6. Ranh giới quan trọng khi trình bày

- Tool có chữ **benchmark** chỉ đo tốc độ theo giao thức đã ghi; không phải mọi con số latency đều có cùng phạm vi đo.
- Tool có chữ **fusion** tạo hệ suy luận kết hợp; nó khác checkpoint Faster R-CNN hoặc RetinaNet đơn lẻ.
- Tool có chữ **role/association** là hậu xử lý hình học và review thủ công, không phải phần detector học sâu.
- Tool có chữ **challenge/hard subset** phục vụ phân tích lỗi. Không lấy kết quả từ đó để tinh chỉnh model trên test.
- `best_map.pth` là trọng số fine-tune để suy luận; tool không “huấn luyện lại” checkpoint trừ các bước train trong `src.train`.
