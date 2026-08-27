# Phân tích Hard Subset v1 từ EdgeVision test

- **Phạm vi:** phân tích độ khó của một tập con 50 ảnh lấy từ `data/splits/test.json` đã đóng băng.
- **Người phụ trách:** Nguyễn Thành Long.
- **Trạng thái:** số liệu đã được xuất từ JSON; cần Long duyệt contact sheet trước khi đưa hình vào báo cáo chính.
- **Nguồn đầu vào:** EdgeVision Dataset v1; `data/challenge/annotations/edgevision_hard_subset_v1.coco.json`; checkpoint tốt nhất của Faster R-CNN và RetinaNet.
- **Lưu ý phương pháp:** đây không phải external challenge set và không dùng để chọn checkpoint, threshold hay hyperparameter. Hai model được chạy trên cùng 50 ảnh/349 bounding box với COCO evaluator như nhau.

## 1. Cách tạo Hard Subset

Hard Subset v1 được chọn bằng đặc trưng của ảnh và annotation, không dùng prediction của model: số lượng annotation, số đầu người nhỏ, chồng lấn giữa các box đầu, độ sáng trung bình thấp và box bị cắt mép ảnh. Tập bao gồm 50 ảnh có 349 bounding box: 154 `BikeWithRider`, 131 `NoHelmet` và 64 `Helmet`.

Các thẻ độ khó có thể chồng lấn. Kết quả chọn được 48 ảnh `crowded`, 46 ảnh `small_object`, 29 ảnh `overlapping_boxes`, 10 ảnh `low_light` và 11 ảnh `edge_truncation`. Kiểm tra COCO không phát hiện image ID/annotation ID trùng, box ngoài ảnh, category lạ hay ảnh thiếu tệp. Báo cáo hash không phát hiện file trùng hoặc near-duplicate theo ngưỡng average-hash đã cấu hình.

[CẦN CHÈN HÌNH: contact sheet `data/challenge/previews/edgevision_hard_subset_v1.jpg` - người cung cấp: Long.]

## 2. Giao thức đánh giá

Hai model dùng cùng file annotation với SHA-256 `c7eeab92966999efb2df59efea1669429613cf2fe0488e621a8a2b4d92cd73ab`, batch size 1 và GPU RTX 2050. mAP sử dụng toàn bộ prediction theo COCO mAP@[0.50:0.95]. Precision/Recall dùng ghép một-một cùng lớp ở IoU = 0,50.

Ngưỡng confidence cho Precision/Recall vẫn là ngưỡng đã chọn từ validation, không chọn từ Hard Subset: Faster R-CNN dùng 0,85; RetinaNet dùng 0,60. Vì vậy mAP có thể so sánh trực tiếp trên cùng evaluator; Precision/Recall phản ánh cấu hình demo đã chốt riêng cho mỗi model.

## 3. Kết quả

| Chỉ số | Faster R-CNN | RetinaNet |
|---|---:|---:|
| mAP@0.5:0.95 | 0,6206 | 0,5877 |
| mAP@0.5 | 0,8866 | 0,8533 |
| AP@0.5:0.95 — BikeWithRider | 0,7663 | 0,7498 |
| AP@0.5:0.95 — NoHelmet | 0,5055 | 0,4644 |
| AP@0.5:0.95 — Helmet | 0,5900 | 0,5489 |
| Precision NoHelmet | 0,9278 | 0,8889 |
| Recall NoHelmet | 0,6870 | 0,5496 |

Trên Hard Subset này, Faster R-CNN có mAP@0.5:0.95 cao hơn 0,0330 và Recall `NoHelmet` cao hơn khoảng 13,7 điểm phần trăm so với RetinaNet tại các threshold validation tương ứng. Đây là quan sát trên 50 ảnh khó thuộc test EdgeVision, chưa đủ để kết luận mô hình nào tổng quát tốt hơn trong mọi bối cảnh giao thông Việt Nam.

## 4. Ý nghĩa và hạn chế

Kết quả cho thấy các cảnh đông, đầu người nhỏ và box chồng lấn là những điều kiện phù hợp để minh họa hạn chế của hệ thống. Đặc biệt, Recall `NoHelmet` thấp hơn mức kỳ vọng ở ảnh khó cho thấy một số trường hợp không đội mũ có thể bị bỏ sót khi đối tượng nhỏ, bị che hoặc lẫn vào nhiều người trên xe.

Tập này vẫn là một tập con của test EdgeVision nên không đo domain shift độc lập. Challenge Set v1 nguồn mở/bối cảnh Việt Nam sẽ chỉ được gọi là external challenge sau khi đủ ảnh có giấy phép, nguồn và annotation đã duyệt. Không dùng kết quả này để chỉnh threshold hoặc train lại model.

## Nguồn đã sử dụng

- EdgeVision Dataset v1, DOI `10.17632/j82bnw7gsr.1`.
- `outputs/challenge_v1/faster_rcnn/edgevision_hard_subset_v1_metrics.json`.
- `outputs/challenge_v1/retinanet/edgevision_hard_subset_v1_metrics.json`.

## Dữ liệu cần nhóm cung cấp

- Video/ảnh giao thông Việt Nam do nhóm tự thu thập hoặc có quyền sử dụng, nếu muốn mở rộng sang external challenge thật.
- Người kiểm tra chéo tối thiểu 20% annotation khi Challenge Set nguồn ngoài được gán nhãn.

## Điểm cần người phụ trách xác nhận

- Long duyệt contact sheet 50 ảnh và cách gọi “Hard Subset từ EdgeVision test”.
- Không sử dụng các số liệu trong mục này như kết quả benchmark chính của đề tài nếu giảng viên chỉ yêu cầu test split đã định nghĩa.
- Chỉ triển khai adaptation/fine-tune sau khi nhóm duyệt external challenge và dữ liệu bổ sung.
