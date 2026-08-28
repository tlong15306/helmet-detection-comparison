# Kế hoạch phân biệt tài xế và người ngồi sau

## 1. Trạng thái và phạm vi

- **Mục tiêu:** xác định box `NoHelmet`/`Helmet` nào thuộc tài xế, box nào thuộc người ngồi sau, từ đó chỉ cảnh báo đúng trường hợp tài xế không đội mũ.
- **Trạng thái:** Pha 0–1 và gán nhãn `role_dev` đã hoàn thành; nhóm đã xác nhận 80 task. Chưa có `role_test` độc lập và chưa train lại.
- **Phạm vi hiện tại:** bổ sung tầng hậu xử lý quan hệ người–xe–mũ cho kết quả của Faster R-CNN và RetinaNet.
- **Không thay đổi:** ba lớp gốc `BikeWithRider`, `NoHelmet`, `Helmet`, checkpoint hiện tại, split EdgeVision và giao thức so sánh hai mô hình.
- **Nguyên tắc:** không gọi một người là tài xế nếu quan hệ chưa đủ chắc chắn; khi mơ hồ phải trả về `unknown` thay vì tạo cảnh báo sai.

## Cập nhật thực hiện — 28/08

- Đã hoàn thành Pha 0 và baseline Pha 1 cho luồng **ảnh**: `src/rider_association.py`, audit annotation và API trả `rider_analysis`.
- Baseline chỉ ghép đầu–xe theo hình học. Một vùng xe có đúng một đầu được gọi `driver_candidate`; đó **không** là kết luận tài xế và `confirmed_driver_no_helmet` luôn bằng 0.
- Audit trên ground truth validation EdgeVision: 633/670 box đầu (94,48%) ghép được duy nhất, 28/670 (4,18%) mơ hồ và 9/670 (1,34%) chưa ghép. Đây là coverage hình học, không phải accuracy tài xế/người ngồi sau.
- Báo cáo audit cục bộ: `outputs/role_association/edgevision_val_geometry_audit.json`; tái tạo bằng `python -m tools.audit_rider_association`.
- Đã thêm unit/API test. Chưa triển khai video role analysis, chưa hoàn tất review `role_dev` và chưa đổi/huấn luyện lại detector.
- Đã tạo và review `role_dev.pending.json` gồm 80 task validation đa dạng. Nhóm xác nhận toàn bộ task ngày 28/08; metadata ghi rõ không có log vòng hai riêng cho từng task.
- Baseline một-đầu tạo 53 candidate: 51 candidate trùng nhãn tài xế, tương ứng Precision 96,23%; Recall trên 78 task có tài xế là 65,38%. Đây là metric phát triển trên `role_dev`, không phải kết quả `role_test`.

## Kế hoạch triển khai tiếp theo — quy tắc vai trò v2

1. **Đóng băng bằng chứng phát triển:** lưu hash của annotation `role_dev`, support và metric dùng để duyệt quy tắc trong `configs/rider_association.yaml`.
2. **Quy tắc một đầu:** khi một box đầu được ghép duy nhất với một vùng `BikeWithRider`, trả `driver` với trạng thái `rule_based`; API phải kèm Precision/support quan sát trên `role_dev`, không diễn giải như xác suất của từng ảnh.
3. **Cảnh nhiều đầu:** tiếp tục trả `unknown`. Phân tích hiện tại cho thấy các quy tắc đơn giản như trái/phải, cao/thấp hoặc box lớn nhất không đủ ổn định để đạt mức Precision ưu tiên 0,95.
4. **Cơ chế an toàn:** chỉ bật quy tắc khi metric trong cấu hình đạt `minimum_precision = 0.95` và `minimum_support = 50`; nếu cấu hình không đạt hoặc thiếu thì quay về `driver_candidate`.
5. **API và giao diện:** giữ các trường cũ để tương thích, bổ sung số `rule_based_drivers`, `driver_no_helmet_alerts` và giải thích đây là cảnh báo theo quy tắc đã kiểm chứng trên validation.
6. **Kiểm thử:** thêm ca bật/tắt quy tắc, cấu hình không đạt điều kiện, một đầu có/không mũ và nhóm nhiều đầu luôn abstain.
7. **Đánh giá:** xuất so sánh baseline v1/v2 trên `role_dev`; chưa công bố metric cuối cho tới khi có `role_test` đóng băng.

**Tiêu chí duyệt v2:** Precision phát triển của cảnh báo vai trò không thấp hơn 0,95; không cưỡng ép tài xế trong nhóm nhiều đầu; toàn bộ test backend/frontend qua; có commit riêng để rollback.

## 2. Vấn đề cần giải quyết

Mô hình hiện tại phát hiện độc lập:

1. vùng `BikeWithRider`;
2. đầu có mũ `Helmet`;
3. đầu không mũ `NoHelmet`.

Prediction không có `vehicle_id`, `person_id` hoặc `role`. Vì vậy, một box `NoHelmet` nằm trong ảnh chưa đồng nghĩa với “tài xế không đội mũ”. Trong cảnh có trẻ em, người ngồi sau, nhiều xe sát nhau hoặc các box chồng lấn, việc chỉ đếm `NoHelmet` có thể tạo cảnh báo nghiệp vụ sai.

Pipeline cần tạo thêm quan hệ:

```text
BikeWithRider
├── driver: Helmet | NoHelmet | unknown
├── passengers: [Helmet | NoHelmet | unknown]
└── association_status: confirmed | ambiguous | unmatched
```

## 3. Quyết định kỹ thuật đề xuất

### 3.1. Hướng triển khai chính cho bài tập lớn

Trước mắt dùng **tầng liên kết hậu xử lý có kiểm soát**, không train lại Faster R-CNN và RetinaNet:

- ghép các box đầu với box `BikeWithRider` phù hợp;
- chỉ suy luận vai trò tài xế khi quy tắc đã được kiểm chứng trên tập validation có nhãn vai trò;
- trả về `unknown` khi có nhiều cách ghép gần tương đương;
- hiển thị riêng “tài xế không đội mũ”, “người ngồi sau không đội mũ” và “chưa xác định vai trò”.

Hướng này giữ nguyên thí nghiệm so sánh hai mô hình, ít tốn thời gian hơn việc thay đổi bộ lớp và huấn luyện lại toàn bộ.

### 3.2. Hướng nâng cấp nếu quy tắc không đạt yêu cầu

Tạo dữ liệu có nhãn quan hệ hoặc nhãn vai trò, sau đó huấn luyện một thành phần riêng. Không nên tự đổi ba lớp hiện tại thành nhiều lớp mới khi chưa có đủ dữ liệu, vì nếu thay đổi đầu ra detector thì cả Faster R-CNN và RetinaNet phải được huấn luyện lại trên cùng dữ liệu để bảo đảm công bằng.

Hai phương án có thể khảo sát sau:

- giữ detector hiện tại và huấn luyện bộ phân loại `driver/passenger` trên crop người/đầu;
- tạo annotation quan hệ `vehicle_group_id`, `head_id`, `role` rồi huấn luyện mô hình quan hệ chuyên biệt.

## 4. Dữ liệu vai trò cần bổ sung

### 4.1. Kiểm tra ngữ nghĩa annotation hiện tại

Trước khi viết quy tắc, kiểm tra thủ công một mẫu ảnh EdgeVision để xác nhận:

- một box `BikeWithRider` bao gồm một xe hay có thể bao gồm nhiều xe;
- một box có thể chứa bao nhiêu đầu người;
- box tài xế và người ngồi sau thường nằm như thế nào trong vùng xe;
- các trường hợp thiếu box đầu, box bị cắt mép và box giữa hai xe chồng lấn.

Kết quả kiểm tra phải được ghi thành tệp thống kê; không suy đoán quy tắc chỉ từ vài ảnh minh họa.

### 4.2. Tập phát triển quy tắc

Tạo một tập nhỏ từ **validation**, dự kiến 60–100 ảnh có đủ trường hợp một người, hai người, trẻ em, che khuất, đông xe và vật thể nhỏ. Mỗi quan hệ được gán:

```json
{
  "image_id": 0,
  "vehicle_group_id": "...",
  "bike_box_id": "...",
  "head_box_id": "...",
  "helmet_status": "helmet | no_helmet | unknown",
  "role": "driver | passenger | unknown",
  "occluded": false,
  "reviewer": "..."
}
```

Tập này được đặt tên `role_dev` và chỉ dùng để xây dựng quy tắc, chọn ngưỡng liên kết và phân tích lỗi.

### 4.3. Tập kiểm tra vai trò

- Chuẩn bị một tập `role_test` độc lập, ưu tiên ảnh giao thông Việt Nam do nhóm tự thu thập hoặc có quyền sử dụng.
- Đóng băng tập này trước khi đánh giá cuối.
- Không chỉnh quy tắc hoặc ngưỡng dựa trên kết quả `role_test`.
- Hard Subset 50 ảnh từ EdgeVision test hiện tại chỉ dùng để phân tích cuối, không dùng để chọn quy tắc.

## 5. Thuật toán liên kết phiên bản 1

### 5.1. Bước A — Ghép đầu với vùng xe

Với mỗi box `Helmet` hoặc `NoHelmet`:

1. tìm các box `BikeWithRider` chứa tâm box đầu;
2. tính tỷ lệ diện tích box đầu nằm trong từng box xe;
3. kết hợp độ phủ, khoảng cách tương đối và confidence để xếp hạng ứng viên;
4. gán đầu cho đúng một box xe có điểm cao nhất;
5. nếu hai ứng viên gần tương đương hoặc không đạt ngưỡng validation, đánh dấu `ambiguous/unmatched`.

Một xe có thể nhận nhiều box đầu, nhưng một box đầu không được gán đồng thời cho nhiều xe. Công thức điểm và mọi ngưỡng sẽ được chọn trên `role_dev`, không ấn định trước bằng cảm tính.

### 5.2. Bước B — Xác định vai trò

Áp dụng theo mức độ chắc chắn:

- **Một đầu trong vùng xe:** tạo ứng viên tài xế, nhưng vẫn đánh dấu độ tin cậy vai trò để kiểm tra các trường hợp box bị thiếu.
- **Nhiều đầu trong vùng xe:** chưa tự chọn tài xế chỉ dựa vào trái/phải, vì hướng chuyển động có thể thay đổi và box hiện tại không cho biết đầu xe nằm phía nào.
- Chỉ dùng quy tắc vị trí tương đối nếu thống kê `role_dev` chứng minh quy tắc ổn định.
- Nếu không có tín hiệu đủ rõ, trả về `unknown` và không phát cảnh báo “tài xế vi phạm”.

### 5.3. Bước C — Ổn định cho video

Sau khi phiên bản ảnh hoạt động đúng, mới bổ sung theo dõi đơn giản giữa các frame:

- gán `track_id` tạm thời cho vùng xe;
- làm mượt vai trò và trạng thái mũ trong một cửa sổ frame ngắn;
- không cộng lặp một tài xế qua mọi frame vào tổng vi phạm;
- reset track khi mất dấu đủ lâu.

Video không được dùng để che lỗi của thuật toán ảnh; phải kiểm thử phiên bản ảnh trước.

## 6. Thiết kế đầu ra và giao diện

### 6.1. API

Giữ trường `detections` cũ để tương thích và bổ sung `rider_groups`:

```json
{
  "group_id": "bike_1",
  "bike_detection_id": "...",
  "driver": {
    "head_detection_id": "...",
    "helmet_status": "no_helmet",
    "role_confidence": 0.0,
    "status": "confirmed | ambiguous"
  },
  "passengers": [],
  "unassigned_heads": []
}
```

`role_confidence` ở trên chỉ là schema, không phải số liệu thực nghiệm. Cách hiệu chỉnh điểm phải được ghi rõ trước khi dùng như xác suất.

### 6.2. Giao diện

- đỏ: tài xế `NoHelmet` đã đủ điều kiện cảnh báo;
- cam/nét đứt: phát hiện `NoHelmet` nhưng vai trò chưa chắc chắn;
- xanh lá: tài xế có mũ;
- người ngồi sau hiển thị nhãn riêng, không cộng vào số “tài xế vi phạm”;
- thêm dòng giải thích “Hệ thống có thể trả về chưa xác định trong cảnh che khuất hoặc đông xe”.

Không thay đổi cách hiển thị cũ cho đến khi API mới có kiểm thử và có thể bật/tắt bằng cấu hình.

## 7. Cấu trúc mã nguồn dự kiến

```text
src/
└── rider_association.py       # Ghép đầu–xe và suy luận vai trò

configs/
└── rider_association.yaml     # Ngưỡng đã chọn từ role_dev

data/role_association/
├── README.md
├── annotations/
│   ├── role_dev.json
│   └── role_test.json
└── metadata/
    └── labeling_log.csv

tests/
└── test_rider_association.py  # Ca một người, nhiều người, chồng box, unmatched

outputs/role_association/
├── role_dev_metrics.json
├── role_test_metrics.json
└── error_cases/
```

Ảnh gốc và video tiếp tục nằm ngoài Git. Chỉ commit schema, annotation nhỏ có quyền sử dụng, cấu hình, metric và mã nguồn.

## 8. Chỉ số đánh giá riêng cho tầng vai trò

Không trộn các chỉ số dưới đây với mAP của detector:

- **Association coverage:** tỷ lệ box đầu được ghép với một vùng xe.
- **Ambiguity rate:** tỷ lệ box đầu/nhóm xe bị trả về `unknown` hoặc `ambiguous`.
- **Role accuracy hoặc Macro-F1:** khả năng phân biệt `driver`, `passenger`, `unknown`.
- **Driver-NoHelmet Precision:** trong các cảnh báo tài xế không mũ, bao nhiêu cảnh báo đúng.
- **Driver-NoHelmet Recall:** trong các tài xế không mũ thực tế, hệ thống phát hiện được bao nhiêu.
- **Sai số theo tình huống:** một người, nhiều người, trẻ em, đông xe, đầu nhỏ, che khuất và thiếu sáng.

Ưu tiên Precision của cảnh báo tài xế để hạn chế gắn nhầm người ngồi sau thành người vi phạm. Mọi ngưỡng chấp nhận phải được nhóm duyệt trước; chưa có số liệu thì không khẳng định hệ thống đã phân biệt chính xác.

## 9. Kiểm thử bắt buộc

### 9.1. Kiểm thử đơn vị

- một đầu nằm rõ trong một box xe;
- hai đầu thuộc cùng một xe;
- một đầu nằm trong hai box xe chồng lấn;
- đầu không thuộc box xe nào;
- box rỗng, box ngoài ảnh và prediction thiếu trường;
- kết quả không phụ thuộc thứ tự detection đầu vào;
- một đầu không bị gán cho hai xe.

### 9.2. Kiểm thử tích hợp

- cả Faster R-CNN và RetinaNet trả cùng schema quan hệ;
- endpoint ảnh giữ tương thích với frontend hiện tại;
- video không đếm lặp liên tục cùng một track;
- khi tắt tính năng, demo trở về đúng hành vi cũ;
- lỗi liên kết không làm dừng toàn bộ suy luận.

### 9.3. Kiểm tra trực quan

Tạo contact sheet gồm:

- ví dụ ghép đúng;
- ví dụ tài xế/người ngồi sau đúng;
- ví dụ mơ hồ được trả về `unknown`;
- lỗi điển hình để đưa vào phần hạn chế của báo cáo.

## 10. Các pha thực hiện và điểm duyệt

### Pha 0 — Audit dữ liệu

- kiểm tra box EdgeVision và thống kê quan hệ hình học;
- đề xuất schema nhãn vai trò;
- **điểm duyệt:** Long xác nhận cách hiểu tài xế/người ngồi sau.

### Pha 1 — Baseline ghép đầu–xe

- viết `rider_association.py` chỉ làm association;
- chưa tự kết luận tài xế ở nhóm nhiều người;
- thêm unit test và xuất contact sheet;
- **điểm duyệt:** kiểm tra trực quan các ca khó.

### Pha 2 — Gán nhãn `role_dev` và xây quy tắc vai trò

- gán nhãn vai trò trên validation;
- chọn quy tắc/ngưỡng bằng `role_dev`;
- xuất metric và nhóm lỗi;
- **điểm duyệt:** quyết định quy tắc có đủ tốt để tích hợp hay cần mô hình có giám sát.

### Pha 3 — Tích hợp demo

- mở rộng API, ảnh và video;
- thêm chế độ bật/tắt “Phân biệt vai trò”;
- hiển thị `unknown` thay vì cưỡng ép kết luận;
- **điểm duyệt:** Long thử trực tiếp trên ảnh/video Việt Nam.

### Pha 4 — Đánh giá đóng băng

- chạy một lần trên `role_test`;
- lưu config, hash annotation và metric;
- cập nhật báo cáo, không chỉnh lại quy tắc theo test.

### Pha 5 — Nâng cấp có nhãn, nếu cần

Chỉ bắt đầu khi Pha 2 cho thấy quy tắc không đạt yêu cầu và nhóm chấp nhận khối lượng gán nhãn/huấn luyện bổ sung.

## 11. Lịch đề xuất trước ngày 03/09

| Thời gian | Công việc | Đầu ra |
|---|---|---|
| 28–29/08 | Pha 0 | Thống kê box, schema và mẫu nhãn vai trò |
| 29–30/08 | Pha 1 | Baseline association, test và contact sheet |
| 30–31/08 | Pha 2 | `role_dev`, quy tắc vai trò và metric phát triển |
| 31/08–01/09 | Pha 3 | API/UI ảnh, sau đó video |
| 01–02/09 | Pha 4 | Đánh giá đóng băng, ảnh minh họa và nội dung báo cáo |
| Sau 02/09 | Pha 5 nếu còn thời gian | Thử nghiệm có nhãn; không làm ảnh hưởng bản demo ổn định |

Nếu thiếu dữ liệu nhãn vai trò, bản nộp vẫn giữ tính năng phát hiện mũ hiện tại và trình bày phân biệt vai trò như hạn chế/hướng phát triển, thay vì công bố một chức năng chưa được kiểm chứng.

## 12. Rủi ro và cách kiểm soát

| Rủi ro | Kiểm soát |
|---|---|
| Box `BikeWithRider` chứa nhiều xe/người | Audit trước; cho phép `ambiguous` |
| Không biết đầu xe nằm bên trái hay phải | Không dùng quy tắc trái/phải khi chưa có bằng chứng validation |
| Người ngồi sau/trẻ em bị tính là tài xế | Chỉ cảnh báo khi role đủ chắc chắn; còn lại `unknown` |
| Cảnh đông khiến một đầu khớp nhiều xe | So điểm ứng viên và kiểm tra độ chênh; không gán cưỡng ép |
| Tuning trên test | Chỉ chọn quy tắc/ngưỡng từ `role_dev`; đóng băng `role_test` |
| Thay đổi làm hỏng demo hiện tại | Feature flag, giữ API cũ và commit theo từng pha |
| Không đủ thời gian retrain | Tách Pha 5 thành tùy chọn, không chặn bản demo chính |

## 13. Tiêu chí hoàn thành phiên bản phù hợp BTL

- Có annotation vai trò được kiểm tra và tách rõ `role_dev`/`role_test`.
- Ghép đầu–xe xác định, tái lập được và có unit test.
- Không gán một đầu cho nhiều xe; trường hợp mơ hồ có trạng thái riêng.
- Cảnh báo tài xế không mũ được đánh giá bằng Precision/Recall riêng.
- Cả hai detector dùng cùng tầng liên kết và cùng dữ liệu đánh giá.
- Demo ảnh hoạt động ổn định trước khi mở rộng video.
- Báo cáo phân biệt rõ detection metric, relation metric và hạn chế dữ liệu.
- Mỗi pha có commit riêng để có thể quay lại phiên bản ổn định.

## 14. Nội dung cần Long xác nhận trước khi bắt đầu

1. Chấp nhận hướng chính: hậu xử lý có `unknown`, chưa train lại detector.
2. Chấp nhận gán nhãn vai trò trước trên 60–100 ảnh validation.
3. Chỉ cảnh báo “tài xế không đội mũ” khi vai trò đủ chắc chắn; các ca còn lại hiển thị “chưa xác định”.
4. Pha mô hình có nhãn là tùy chọn sau khi đo chất lượng quy tắc và không được làm chậm bản demo ổn định.
