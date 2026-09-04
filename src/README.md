# Tài liệu mã nguồn `src/`

## 1. Mục đích

Thư mục `src/` chứa pipeline học sâu của đề tài phát hiện người đi xe máy có/không đội mũ bảo hiểm. Pipeline hỗ trợ hai detector được huấn luyện và đánh giá theo cùng một quy trình:

- **Faster R-CNN ResNet-50 FPN v2**: detector hai giai đoạn. Mô hình tạo vùng nghi ngờ bằng RPN, sau đó phân loại và tinh chỉnh khung giới hạn.
- **RetinaNet ResNet-50 FPN v2**: detector một giai đoạn. Mô hình dự đoán trực tiếp lớp và khung trên các anchor đa tỉ lệ; phần phân loại sử dụng Focal Loss trong kiến trúc RetinaNet của Torchvision.

Mã nguồn không coi hai kết quả là có thể so sánh chỉ vì cùng tên metric. Các công cụ đánh giá và so sánh đều lưu/kiểm tra split, annotation, lớp, IoU và các điều kiện liên quan.

## 2. Luồng xử lý tổng quát

```text
COCO JSON + ảnh
       │
       ▼
dataset.py ──► transforms.py ──► DataLoader
                                      │
                                      ▼
                              models.py: Faster R-CNN / RetinaNet
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
        train.py                 evaluate.py                infer.py
     fine-tune/checkpoint        metric JSON               ảnh đã vẽ box
              │                       │                        │
              ▼                       ▼                        ▼
        run_manifest.json     compare_models.py     postprocess.py (tùy chọn)
                               comparison JSON      ghép đầu–xe/cảnh báo an toàn
```

### Dữ liệu đi qua các mô-đun

- **Ảnh**: `PIL.Image` khi mới đọc; sau `transforms.py` là tensor `float32` dạng `[C, H, W]`, giá trị trong `[0, 1]`.
- **Nhãn thật (`target`)**: dictionary gồm `boxes` theo định dạng `[x1, y1, x2, y2]`, `labels`, `image_id`, `area`, `iscrowd`.
- **Dự đoán (`prediction`)**: dictionary của Torchvision gồm `boxes`, `labels`, `scores`. Một prediction là một khung giới hạn, lớp dự đoán và độ tin cậy.
- **Tệp cấu hình**: YAML trong `configs/`. Các đường dẫn tương đối được hiểu từ root dự án, không phải từ thư mục đang chạy lệnh.

## 3. Thuật toán xử lý chính

### 3.1. Fine-tuning detector

`train.py` thực hiện **transfer learning / fine-tuning**, không huấn luyện kiến trúc từ đầu:

1. `models.py` nạp trọng số pretrained `DEFAULT` của Torchvision hoặc khởi tạo không trọng số khi cấu hình yêu cầu.
2. Detection head được thay để số lớp đầu ra khớp với `model.num_classes` (bao gồm lớp background có id `0`).
3. `dataset.py` đọc ảnh và ground truth từ COCO JSON; `transforms.py` chỉ augmentation trên train.
4. Trong mỗi batch train, detector trả dictionary các thành phần loss. `train_one_epoch()` cộng các loss, gọi `backward()` và optimizer cập nhật các tham số đang được phép học.
5. Sau mỗi chu kỳ theo cấu hình, `validate()` chạy suy luận trên validation, tính mAP và Precision/Recall. Validation được dùng để chọn `best_map.pth`; test không được `train.py` đọc.
6. `last.pth`, `history.json`, `run_manifest.json` và `environment.json` được ghi để tái lập lần chạy.

Với pilot/fine-tune, cấu hình có thể đóng băng `backbone.body`. Khi đó FPN và detection head vẫn học, còn phần thân ResNet không cập nhật. Optimizer được chọn rõ trong YAML (`SGD` hoặc `AdamW`); scheduler hỗ trợ `StepLR`; AMP chỉ bật trên CUDA khi cấu hình cho phép.

### 3.2. Suy luận và lọc kết quả

`infer.py` chuẩn hóa ảnh RGB, chuyển thành tensor, chạy `model.eval()` trong `torch.inference_mode()`, sau đó:

1. Torchvision detector tạo `boxes`, `labels`, `scores`; NMS nội bộ của detector đã nằm trong bước forward.
2. `filter_predictions()` bỏ prediction có score thấp hơn confidence threshold chung hoặc threshold riêng theo lớp.
3. `draw_detections()` vẽ khung, tên lớp và score lên bản sao ảnh đầu vào.
4. `prediction_records()` chuyển tensor thành record JSON dễ gửi frontend.

`predict_image()` đo latency từ lúc đưa tensor lên thiết bị đến hết forward/NMS. Số đo **không** gồm đọc tệp ảnh hoặc vẽ giao diện.

### 3.3. Đánh giá công bằng

`metrics.py` và `evaluate.py` dùng cùng quy ước:

- **IoU**: diện tích giao của hai box chia cho diện tích hợp của chúng.
- **TP/FP/FN cho Precision và Recall**: prediction được sắp theo score giảm dần, chỉ ghép với một ground truth chưa ghép cùng lớp và có IoU đạt ngưỡng. Đây là ghép tham lam một-một.
- **mAP**: `MeanAveragePrecision` theo chuẩn COCO, `mAP@0.5:0.95` dùng các ngưỡng IoU từ 0.50 đến 0.95, bước 0.05. `mAP@0.5`, `mAP@0.75`, `mAR@100` và AP theo lớp cũng được xuất nếu xác định được.

`compare_models.py` từ chối so sánh mặc định nếu hai JSON kết quả khác split, annotation SHA-256, class mapping, định nghĩa mAP, backend metric hoặc quy tắc Precision/Recall. Đây là cơ chế bảo vệ để không rút kết luận từ hai điều kiện thử nghiệm khác nhau.

## 4. Bảng tra nhanh từng tệp

| Tệp | Làm gì | Đầu vào chính | Đầu ra chính |
|---|---|---|---|
| `__init__.py` | Đánh dấu gói mã nguồn. | — | Cho phép import `src.*`. |
| `utils.py` | Đọc/gộp YAML, giải quyết đường dẫn, đặt random seed. | Đường dẫn, dictionary YAML, seed. | Config hoàn chỉnh, đường dẫn tuyệt đối. |
| `dataset.py` | Đọc COCO JSON và ảnh. | Thư mục ảnh, annotation COCO. | `(image, target)` và `collate_fn`. |
| `transforms.py` | Augmentation đồng bộ ảnh–box. | Ảnh PIL/tensor, `target`. | Ảnh tensor và box đã cập nhật. |
| `models.py` | Tạo Faster R-CNN hoặc RetinaNet. | Tên model, số lớp, trọng số, cỡ ảnh. | `torch.nn.Module` detector. |
| `train.py` | Fine-tune, validation, checkpoint, manifest. | YAML train/val, dataset, model. | `.pth`, lịch sử JSON, manifest môi trường. |
| `metrics.py` | IoU, Precision/Recall, COCO mAP. | Prediction và ground truth. | Dictionary metric tổng và theo lớp. |
| `evaluate.py` | Đánh giá checkpoint trên split cố định. | Config, checkpoint, val/test/challenge. | JSON `evaluation-result-1.0`. |
| `compare_models.py` | Kiểm tra protocol rồi so sánh hai JSON. | JSON Faster R-CNN, JSON RetinaNet. | JSON `model-comparison-1.0`. |
| `infer.py` | Suy luận một ảnh và trực quan hóa. | Model đã nạp, ảnh, threshold. | Prediction lọc, latency, PNG có box. |
| `threshold_selection.py` | Chọn threshold demo trên validation. | Config, checkpoint, danh sách ngưỡng. | JSON quét ngưỡng; tùy chọn cập nhật YAML demo. |
| `prediction_fusion.py` | Ensemble/TTA và weighted box fusion. | Prediction từ >= 2 nguồn. | Prediction hợp nhất. |
| `postprocess.py` | Xử lý xung đột Helmet/NoHelmet, ghép đầu–xe, tạo cảnh báo có điều kiện. | Prediction đã lọc, config association/role. | Detection sau xử lý, unknown cases, nhóm xe–đầu, alert. |
| `rider_association.py` | Baseline hình học ghép box đầu với box xe. | Detection records, ngưỡng cấu hình. | Nhóm người–xe, candidate/unknown; không tự khẳng định tài xế. |
| `role_annotations.py` | Kiểm tra schema nhãn vai trò thủ công. | JSON task `role_dev`. | Thống kê task hoặc lỗi schema. |
| `role_evaluation.py` | Chẩn đoán heuristic vai trò trên nhãn người review. | `role_dev` hợp lệ. | Precision/recall candidate và giới hạn áp dụng. |

## 5. Chi tiết từng mô-đun

### `utils.py` — cấu hình và khả năng tái lập

- `resolve_project_path()`: đổi đường dẫn tương đối thành đường dẫn tính từ root dự án.
- `load_yaml()`: đọc YAML bằng `safe_load` và bắt buộc nội dung cấp cao nhất là dictionary.
- `deep_merge()` và `load_config()`: cho phép config con dùng `base_config`, sau đó chỉ ghi đè các khóa cần thay đổi mà không sửa config gốc.
- `set_seed()`: đặt seed cho Python, NumPy và PyTorch/CUDA nếu có.

**Đầu ra:** dictionary config đã gộp và môi trường random nhất quán hơn giữa các lần chạy. Seed không bảo đảm tái lập tuyệt đối trên mọi phần cứng/CUDA, nhưng là điều kiện nền cần thiết.

### `dataset.py` — chuyển COCO thành dữ liệu cho detector

`CocoBoxDataset` đọc các trường `images`, `annotations`, `categories` của COCO JSON. Mỗi annotation COCO có box `[x, y, width, height]` được đổi thành `[x1, y1, x2, y2]`. Box có chiều rộng hoặc chiều cao không dương bị bỏ.

Khi gọi `dataset[index]`, mô-đun mở đúng ảnh, tự sửa hướng EXIF, đổi sang RGB và trả:

```python
image, {
    "boxes": FloatTensor[N, 4],
    "labels": Int64Tensor[N],
    "image_id": Int64Tensor[],
    "area": FloatTensor[N],
    "iscrowd": Int64Tensor[N],
}
```

`collate_fn()` không ép các ảnh về cùng kích thước; nó giữ chúng trong list/tuple để detector Torchvision tự xử lý.

### `transforms.py` — augmentation không làm sai annotation

- `Compose`: gọi lần lượt các phép biến đổi.
- `RandomHorizontalFlip`: lật ảnh với xác suất cấu hình và cập nhật tọa độ x của mọi box theo `x1' = W - x2`, `x2' = W - x1`.
- `RandomColorJitter`: thay đổi nhẹ brightness, contrast, saturation, hue; không đổi box vì đây chỉ là biến đổi quang học.
- `ToFloatTensor`: đổi ảnh sang `float32` trong `[0, 1]`.
- `build_transforms(train=...)`: train dùng flip/color jitter nếu được bật; validation/test chỉ đổi tensor, không augmentation ngẫu nhiên.

**Đầu ra:** ảnh và ground truth vẫn cùng hệ tọa độ, tránh lỗi huấn luyện do box không còn trỏ đúng vật thể.

### `models.py` — factory tạo kiến trúc

`build_model()` chỉ chấp nhận hai tên nằm trong `SUPPORTED_MODELS`:

- `fasterrcnn_resnet50_fpn_v2`: nạp Faster R-CNN, sau đó thay `roi_heads.box_predictor` bằng `FastRCNNPredictor` có đúng số lớp.
- `retinanet_resnet50_fpn_v2`: nạp RetinaNet, sau đó thay `classification_head` bằng `RetinaNetClassificationHead` có đúng số lớp và số anchor mỗi vị trí.

Hàm nhận `min_size`, `max_size`, số layer backbone có thể train và lựa chọn weights `DEFAULT`/`NONE`. `num_classes` luôn tính cả background theo quy ước Torchvision.

**Đầu ra:** detector PyTorch tương thích cùng API: khi train nhận `(images, targets)` và trả losses; khi eval nhận `images` và trả predictions.

### `train.py` — điều phối fine-tune

Đây là mô-đun điều phối chính, có thể chạy bằng `python -m src.train --config <config.yaml>`.

- `validate_config()`: bắt lỗi thiếu đường dẫn train/val, số lớp, epoch, batch size, checkpoint output và cấu hình sampling trước khi chạy tốn GPU.
- `build_loaders()`: dựng train loader và validation loader; không nhận test annotation. Hỗ trợ ba cách lấy mẫu tùy config: random mặc định, `WeightedRandomSampler` cho lớp/vật thể nhỏ, hoặc sampler giữ tỷ lệ nguồn dữ liệu Việt Nam.
- `build_optimizer()`: tạo `SGD` hoặc `AdamW`; `build_scheduler()` tạo `StepLR` hoặc không dùng scheduler.
- `configure_trainable_parameters()`: có thể đóng băng `backbone.body` nhưng giữ FPN và detection head cho fine-tune.
- `train_one_epoch()`: forward train → tổng các loss → kiểm tra loss hữu hạn → backward → optimizer step. Hỗ trợ gradient accumulation và mixed precision CUDA.
- `validate()`: chuyển model sang eval, chạy `DetectionEvaluator`, không cập nhật trọng số.
- `run_training()`: kiểm tra frozen manifest train/val, đặt seed, nạp pretrained hoặc checkpoint khởi tạo, chạy các epoch, lưu checkpoint tốt nhất theo `primary_metric` và checkpoint cuối.

**Tệp sinh ra:**

- `best_map.pth`: checkpoint có metric validation chính tốt nhất.
- `last.pth`: checkpoint sau epoch cuối đã chạy.
- `logs/history.json`: loss, learning rate, thời gian epoch, validation của từng epoch.
- `run_manifest.json`: cấu hình, phiên bản thư viện, hash đầu vào train/val, GPU, checkpoint và lịch sử để truy vết.
- `environment.json`: Python, Torch, Torchvision, CUDA/GPU.

`--smoke-test` chỉ chạy tối đa một batch train và validation trong thư mục output riêng, sau đó nạp lại checkpoint để kiểm tra schema prediction. Nó **không** tạo checkpoint chính thức.

### `metrics.py` — chỉ số detection chung

- `pairwise_iou()`: tạo ma trận IoU giữa các prediction box và ground-truth box.
- `precision_recall()`: tính `Precision = TP/(TP+FP)` và `Recall = TP/(TP+FN)`.
- `build_map_metric()`: tạo evaluator mAP TorchMetrics dùng backend `pycocotools`.
- `DetectionEvaluator.update()`: chuẩn hóa, kiểm tra dữ liệu và cộng dồn một batch prediction/target.
- `_update_precision_recall()`: ghép tham lam một-một theo lớp; prediction score cao được xét trước. Một ground truth chỉ có thể tạo TP một lần.
- `compute()`: xuất `map_50_95`, `map_50`, `map_75`, `mar_100`, TP/FP/FN, Precision/Recall và AP theo lớp.

**Đầu ra:** dictionary có thể JSON hóa. mAP dùng toàn bộ prediction theo chuẩn COCO; confidence threshold chỉ áp dụng cho Precision/Recall nếu được cấu hình.

### `evaluate.py` — đánh giá checkpoint đã chốt

Chạy bằng `python -m src.evaluate --config <config.yaml> --split test` (hoặc `val`, `challenge`).

Mô-đun kiểm tra checkpoint, class mapping trong COCO JSON, thiết bị và annotation. Sau đó nạp checkpoint ở chế độ strict mặc định, chạy toàn bộ split trong `torch.inference_mode()`, rồi gọi `DetectionEvaluator`.

**Tệp sinh ra:** JSON schema `evaluation-result-1.0`, gồm:

- metadata model, checkpoint và SHA-256 checkpoint;
- evaluation protocol: split, annotation path/SHA-256, lớp, định nghĩa mAP, IoU, ngưỡng confidence;
- runtime: Python, Torch, Torchvision, thiết bị, batch size, seed;
- metric tổng và theo lớp.

Tùy chọn `--allow-partial-load` chỉ dành cho chẩn đoán checkpoint không khớp; JSON sẽ lưu các key thiếu/thừa. Nó không nên dùng để công bố kết quả chính thức.

### `compare_models.py` — so sánh có kiểm tra điều kiện

Chạy bằng `python -m src.compare_models --faster-rcnn <json> --retinanet <json> --output <json>`.

Mô-đun đọc hai `evaluation-result-1.0`, so sánh các khóa protocol bắt buộc. Nếu có khác biệt, mặc định chương trình dừng với lỗi thay vì tạo bảng so sánh gây hiểu nhầm. Cờ `--allow-protocol-mismatch` chỉ xuất JSON chẩn đoán và đánh dấu `comparison_status: protocol_mismatch`.

**Tệp sinh ra:** `model-comparison-1.0` gồm hai số gốc và hiệu `RetinaNet - Faster R-CNN` cho mAP, mAR và các metric lớp `NoHelmet` nếu có.

### `infer.py` — suy luận ảnh và vẽ kết quả

Mô-đun có API tái sử dụng cho demo/backend và CLI xử lý một ảnh:

```text
PIL Image → RGB/tensor → detector eval + NMS → lọc threshold → records/vẽ PNG
```

- `normalize_pil_image()` và `image_to_tensor()` chuẩn hóa ảnh đầu vào.
- `predict_image()` trả `(prediction_da_loc, latency_ms)`.
- `filter_predictions()` hỗ trợ một threshold chung hoặc mapping threshold theo class id.
- `summarize_detections()` đếm kết quả theo lớp.
- `prediction_records()` đổi output tensor thành record `detection_id`, `class_id`, `class_name`, `confidence`, `box` cho frontend/API.
- `draw_detections()` vẽ box, nhãn, score; có thể nhận annotation hiển thị từ hậu xử lý nhưng không sửa prediction gốc.
- `encode_png()` đưa ảnh đã vẽ thành bytes PNG.

**CLI output:** ảnh PNG/JPEG đã vẽ tại `--output` và một JSON trên console chứa đường dẫn output, latency, metadata checkpoint.

### `threshold_selection.py` — chọn ngưỡng hiển thị, không dùng test

Threshold hiển thị không được chọn trên test. Mô-đun này cố định split là **validation**, chạy inference đúng một lần, giữ output trên CPU rồi quét nhiều threshold mà không phải chạy lại model.

Với từng threshold và từng lớp, `metrics_at_threshold()` ghép TP/FP/FN giống quy ước IoU một-một. `select_best_threshold()` chọn theo thứ tự: F1 cao nhất → Recall → Precision → threshold cao hơn. `select_thresholds_per_class()` làm việc đó riêng cho mỗi lớp.

**Tệp sinh ra:** JSON `demo-threshold-selection-1.0` gồm toàn bộ candidate, policy chọn, checkpoint hash và ngưỡng được chọn. Khi truyền `--demo-config` và `--demo-key`, `update_demo_config()` ghi ngưỡng theo lớp vào YAML cấu hình demo cùng bằng chứng rằng chúng được chọn trên validation.

### `prediction_fusion.py` — hợp nhất dự đoán tùy chọn

Mô-đun này không tạo hay thay checkpoint.

- `EnsembleDetector`: chạy tuần tự từ hai detector trở lên trên cùng ảnh, sau đó hợp nhất prediction để giảm đỉnh VRAM.
- `HorizontalFlipTTADetector`: chạy detector trên ảnh gốc và ảnh lật ngang, chuyển box ảnh lật về tọa độ gốc rồi hợp nhất.
- `fuse_predictions()`: một biến thể **weighted box fusion**. Box chỉ tạo cụm với box cùng lớp và IoU đủ ngưỡng. Tọa độ box cụm là trung bình có trọng số `score × source_weight`; score cụm phản ánh sự đồng thuận tốt nhất của các nguồn.

**Đầu ra:** prediction chuẩn `boxes`, `labels`, `scores`, đã sắp score giảm dần và giới hạn bởi `max_detections`.

### `postprocess.py` — hậu xử lý an toàn cho demo

Đây là tầng demo sau detector; nó không được dùng để thay metric test hoặc chọn checkpoint.

1. `resolve_head_label_conflicts()` tìm các box chồng lấn IoU cao nhưng có hai nhãn đối nghịch `Helmet`/`NoHelmet`. Nếu score quá sát nhau thì không ép chọn nhãn; nó tạo `unknown_case`. Nếu đủ chênh lệch thì giữ nhãn score cao hơn.
2. `analyze_rider_roles()` từ `rider_association.py` ghép các đầu còn lại với vùng xe.
3. `build_alerts()` chỉ tạo alert khi quy tắc role đã được xác nhận bằng dữ liệu role_dev; `driver_candidate` không đủ điều kiện tạo cảnh báo vi phạm.
4. `build_display_annotations()` chỉ đổi màu/chữ hiển thị, không đổi nhãn/score do detector tạo.

**Đầu ra:** raw detections, prediction sau lọc xung đột, records hiển thị, unknown cases, phân tích rider groups, alerts và annotation cho giao diện.

### `rider_association.py` — ghép đầu với xe bằng hình học

Detector chỉ nhìn thấy các lớp độc lập; nó không tự cung cấp ID người–xe hay nhãn “tài xế”. Vì vậy mô-đun này dùng baseline có chủ đích thận trọng:

- Chỉ xét `BikeWithRider` là vùng xe và `Helmet`/`NoHelmet` là vùng đầu.
- Một đầu có thể ghép với xe khi tâm đầu nằm trong box xe và tỉ lệ diện tích đầu nằm trong xe đạt `min_head_coverage`.
- Điểm ghép là `0.90 × coverage + 0.10 × proximity_to_vehicle_center`; thành phần khoảng cách chỉ phá thế hòa, không suy luận bên trái/phải là tài xế.
- Nếu nhiều xe có điểm quá sát (`ambiguity_margin`) thì giữ trạng thái mơ hồ. Nếu một xe có nhiều đầu, chiến lược mặc định cũng là không chọn tài xế.

`RoleDecisionConfig` chỉ bật quy tắc một-đầu khi có bằng chứng role_dev: cấu hình bật, precision quan sát đạt ngưỡng, đủ support, đúng split validation và SHA-256 tệp nhãn khớp. Nếu không, kết quả chỉ là `driver_candidate` hoặc `unknown`.

**Đầu ra:** `rider_groups`, đầu chưa ghép, trường hợp mơ hồ, thống kê và danh sách limitation. Không được diễn giải `driver_candidate` là kết luận chắc chắn.

### `role_annotations.py` — kiểm tra dữ liệu review vai trò

`validate_role_tasks()` kiểm tra schema `role_association_tasks_v1` do người review nhập:

- id task/ảnh/annotation không trùng;
- trạng thái chỉ là `pending`, `reviewed`, `needs_second_review`;
- role chỉ là `driver`, `passenger`, `unknown` hoặc `None` đúng trường hợp;
- task pending không được mang nhãn role;
- task đã review phải có role cho mọi head và tối đa một head tài xế.

**Đầu ra:** số lượng task/ảnh theo trạng thái nếu hợp lệ; nếu không hợp lệ thì dừng với lỗi mô tả rõ trường sai. Hàm không gán nhãn thay con người.

### `role_evaluation.py` — đánh giá chẩn đoán heuristic role

`evaluate_role_candidate_baseline()` đối chiếu gợi ý `proposed_driver_head_annotation_id` với nhãn role do người review nhập trên `role_dev`. Task pending bị loại; `None` được coi là abstention có chủ đích, không phải dự đoán rằng không có tài xế.

Ngoài candidate precision/recall, mô-đun còn thử các quy tắc ép chọn đơn giản ở cảnh nhiều đầu (`leftmost`, `rightmost`, `highest`, `lowest`, `largest_box`) để cho thấy vì sao baseline mặc định chọn abstain thay vì suy đoán.

**Đầu ra:** JSON-like dictionary `role_candidate_evaluation_v1` gồm trạng thái ground truth, số task dùng/bỏ, candidate precision/recall, tỷ lệ abstention, chẩn đoán multi-head và các giới hạn. Đây không phải mAP của Faster R-CNN/RetinaNet, không dùng để train, chọn checkpoint hay kết luận trên role_test.

## 6. Những ranh giới cần giữ khi thuyết trình

- **Fine-tuning là kỹ thuật huấn luyện**, không phải detector thứ ba. Hai detector là Faster R-CNN và RetinaNet.
- **NMS** là hậu xử lý nội bộ của detector Torchvision để loại box trùng, khác với `postprocess.py` là tầng logic demo.
- **Validation** dùng theo dõi/chọn checkpoint/chọn threshold; **test** chỉ dùng đánh giá cuối sau khi chốt cấu hình.
- Association đầu–xe là heuristic hình học hậu xử lý. Nó không biến detector thành mô hình nhận dạng tài xế.
- Ensemble/TTA là tùy chọn khi suy luận; không được gộp metric của nó với metric checkpoint đơn lẻ nếu chưa đánh giá lại theo một protocol rõ ràng.

## 7. Kiểm tra nhanh sau khi sửa mã

```powershell
python -m compileall src
python -m src.train --config configs/faster_rcnn.yaml --smoke-test --device cuda
python -m src.train --config configs/retinanet.yaml --smoke-test --device cuda
```

Smoke test chỉ xác nhận đường đi dữ liệu, forward/backward, validation tối thiểu, ghi/nạp checkpoint và schema prediction. Nó không chứng minh chất lượng mô hình hay thay thế đánh giá trên test.
