# KẾ HOẠCH BỔ SUNG TẬP ẢNH KHÓ CHO BÀI TOÁN PHÁT HIỆN MŨ BẢO HIỂM

## 1. Thông tin kiểm soát

- **Đề tài:** Ứng dụng và so sánh Faster R-CNN và RetinaNet trong phát hiện người điều khiển xe máy không đội mũ bảo hiểm từ hình ảnh giao thông.
- **Người phụ trách phối hợp:** Nguyễn Thành Long.
- **Ngày lập kế hoạch:** 28/08/2026.
- **Trạng thái:** Chờ Long duyệt; chưa tải ảnh, chưa thay đổi dataset và chưa train lại.
- **Mục tiêu trước mắt:** tạo `Challenge Set v1` gồm khoảng 50 ảnh khó để đo khả năng tổng quát của hai checkpoint hiện tại.
- **Mục tiêu mở rộng:** chỉ tạo dữ liệu adaptation và fine-tune lại nếu Challenge Set v1 cho thấy lỗi có hệ thống và nhóm còn đủ thời gian.

## 2. Lý do cần bổ sung

EdgeVision là dataset chính và các split train/validation/test hiện tại đã được đóng băng. Tuy nhiên ảnh giao thông thực tế tại Việt Nam có nhiều trường hợp khó mà một dataset đơn lẻ có thể chưa bao phủ đầy đủ:

- mật độ xe máy và người tham gia giao thông cao;
- nhiều người ngồi trên cùng một xe, trong đó có trẻ em;
- mũ nửa đầu, mũ 3/4, mũ thời trang, nón lưỡi trai và các vật che đầu dễ gây nhầm;
- đầu người có kích thước nhỏ hoặc bị che khuất;
- ảnh tối, ngược sáng, mưa, bóng đổ, nhiễu và nén mạnh;
- góc nhìn cao, góc xiên, chuyển động nhanh hoặc đối tượng nằm sát mép ảnh;
- các khung `Helmet`, `NoHelmet` và `BikeWithRider` chồng lấn trong cảnh đông người.

Challenge Set không thay thế tập test EdgeVision. Kết quả trên tập này được báo cáo riêng để phân tích khả năng tổng quát và các hạn chế trong bối cảnh ảnh khó.

## 3. Quyết định phương pháp quan trọng

### 3.1. Hai pha độc lập

#### Pha A — Challenge audit, bắt buộc thực hiện trước

- Thu thập và khóa khoảng 50 ảnh khó.
- Gán nhãn theo đúng ba lớp hiện tại.
- Chạy hai checkpoint hiện tại mà không điều chỉnh tham số theo Challenge Set.
- Dùng threshold demo đã chọn trên validation: Faster R-CNN `0,85`, RetinaNet `0,60` khi phân tích Precision/Recall hoặc ảnh định tính.
- Với mAP, dùng evaluator COCO hiện tại và không lọc dự đoán bằng threshold demo nếu evaluator chính thức không yêu cầu.
- Phân tích false positive, false negative và nhầm lớp theo nhóm độ khó.

#### Pha B — Domain adaptation, chỉ thực hiện sau khi duyệt kết quả Pha A

- Không đưa 50 ảnh Challenge Set v1 vào train hoặc validation.
- Thu thập một tập adaptation riêng, mục tiêu tối thiểu 150–300 ảnh.
- Chia adaptation train/validation theo `source_group`, không chia ngẫu nhiên từng frame.
- Fine-tune cả Faster R-CNN và RetinaNet bằng cùng dữ liệu, cùng quy tắc augmentation và cùng giao thức đánh giá.
- Đánh giá lại trên cả test EdgeVision đã đóng băng và Challenge Set v1 đã khóa.

### 3.2. Không sửa split EdgeVision hiện tại

- Không thêm ảnh mới trực tiếp vào `data/splits/train.json`, `val.json` hoặc `test.json` hiện tại.
- Không tạo lại split EdgeVision chỉ vì kết quả ảnh khó chưa tốt.
- Nếu thực hiện Pha B, tạo cấu hình và artifact phiên bản mới, ví dụ `challenge_v1` và `adaptation_v1`.
- Kết quả baseline cũ và kết quả adaptation phải lưu ở thư mục khác nhau để không ghi đè.

## 4. Phạm vi Challenge Set v1

### 4.1. Quy mô

- Tải ban đầu khoảng 80–100 ảnh ứng viên.
- Sau lọc chất lượng, giữ khoảng 50 ảnh chính thức.
- Mỗi ảnh có thể mang nhiều thẻ độ khó; các chỉ tiêu bên dưới được phép chồng lấn.

### 4.2. Chỉ tiêu bao phủ

| Nhóm tình huống | Mục tiêu tối thiểu | Ví dụ |
|---|---:|---|
| Bối cảnh giao thông Việt Nam | 20 ảnh | xe máy phổ biến tại Việt Nam, đường phố Việt Nam, mũ nửa đầu |
| Cảnh đông người | 15 ảnh | từ ba người/xe máy trở lên, nhiều box chồng lấn |
| Vật thể nhỏ hoặc ở xa | 10 ảnh | vùng đầu nhỏ, camera góc rộng hoặc góc cao |
| Che khuất | 10 ảnh | đầu bị người khác, xe, cột hoặc vật thể che một phần |
| Ánh sáng khó | 10 ảnh | thiếu sáng, ngược sáng, bóng đổ, đèn xe hoặc trời mưa |
| Loại mũ/vật che đầu dễ nhầm | 8 ảnh | mũ nửa đầu, nón lưỡi trai, mũ thời trang, khăn trùm |
| Mờ, nhiễu hoặc nén mạnh | 8 ảnh | motion blur, CCTV chất lượng thấp, frame video nén |
| Góc nhìn khác thường | 8 ảnh | nhìn từ trên cao, phía sau, góc xiên hoặc sát mép ảnh |

### 4.3. Bao phủ lớp

- Có đủ `BikeWithRider`, `NoHelmet` và `Helmet`.
- Mục tiêu có ít nhất 20 ảnh chứa `NoHelmet` và 20 ảnh chứa `Helmet`.
- Không ép cân bằng bằng cách đưa vào ảnh không phù hợp; số lượng thực tế phải được thống kê sau gán nhãn.
- Ảnh không có đối tượng hợp lệ chỉ được giữ như ảnh âm nếu nhóm thống nhất evaluator hỗ trợ và có mục đích rõ ràng.

## 5. Nguồn dữ liệu dự kiến

### 5.1. Thứ tự ưu tiên

1. Video/ảnh do nhóm tự quay hoặc được chủ sở hữu cho phép sử dụng.
2. Wikimedia Commons có giấy phép CC BY hoặc CC BY-SA và thông tin attribution đầy đủ.
3. Dataset nghiên cứu có trang công bố, giấy phép và đường dẫn tải chính thức.
4. AI City Challenge 2023 nếu nhóm nhận được quyền truy cập hợp lệ.

### 5.2. Nguồn ứng viên đã kiểm tra sơ bộ

| Nguồn | Giá trị sử dụng | Hạn chế/quyết định |
|---|---|---|
| Wikimedia Commons — giao thông/xe máy tại Việt Nam | Bối cảnh Việt Nam, có metadata tác giả và giấy phép | Có thể dùng khi từng tệp có giấy phép phù hợp; phải lưu attribution |
| AI City Challenge 2023 Track 5 | Cảnh giao thông đông, che khuất; có nhãn tách tài xế/người ngồi sau | Dữ liệu tại Ấn Độ; quyền truy cập phải theo trang challenge; không tải từ bản sao không rõ nguồn |
| HelmetML/Mendeley Data | Nhiều loại mũ, điều kiện ngày và đêm, giấy phép CC BY 4.0 trên trang dataset | Có thể thiên về ảnh gần/phân loại; chỉ chọn ảnh phù hợp object detection giao thông |
| HCMCTrafficDataset | Đúng bối cảnh TP.HCM và mật độ xe máy cao | Trang UEH hiện chỉ có abstract và ghi quyền IEEE/all rights reserved; chưa dùng nếu chưa có quyền/tệp chính thức |
| SHWD/GDUT-HWD | Có nhiều ảnh mũ bảo hộ | Loại khỏi nguồn chính vì là mũ công trường, lệch miền so với mũ bảo hiểm xe máy |

### 5.3. Nguồn không sử dụng

- Ảnh tìm ngẫu nhiên trên Google Images, Facebook, TikTok, báo chí hoặc YouTube nếu không xác định được giấy phép.
- Ảnh có watermark lớn hoặc bị chỉnh sửa làm sai nội dung.
- Dataset re-upload trên Kaggle/Drive không có liên kết về nguồn và giấy phép gốc.
- Ảnh tổng hợp bằng AI dùng làm ground truth thực nghiệm chính thức.
- Ảnh test của nguồn bên ngoài nếu điều khoản chỉ cho phép dùng trên hệ thống đánh giá của họ.

## 6. Cấu trúc thư mục dự kiến

```text
data/
└── challenge/
    ├── README.md
    ├── sources_manifest.csv
    ├── raw/                         # ảnh/video tải nguyên gốc, không commit
    ├── candidates/                  # frame và ảnh ứng viên, không commit
    ├── selected/                    # khoảng 50 ảnh đã chọn, không commit
    ├── annotations/
    │   ├── challenge_v1.coco.json
    │   ├── annotation_review.csv
    │   └── label_policy.md
    ├── metadata/
    │   ├── difficulty_tags.csv
    │   ├── source_groups.csv
    │   ├── duplicate_report.json
    │   └── challenge_v1_manifest.json
    └── previews/                    # contact sheet/ảnh render, không commit

outputs/
└── challenge_v1/
    ├── faster_rcnn/
    ├── retinanet/
    ├── comparison/
    └── error_analysis/
```

### 6.1. Quy tắc Git

- Không commit ảnh, video, contact sheet hoặc tệp đầu ra nặng.
- Có thể commit tài liệu, công cụ, annotation COCO, manifest nguồn và metadata không chứa thông tin nhạy cảm.
- Trước khi thay `.gitignore`, phải kiểm tra chính xác các đường dẫn để không ảnh hưởng EdgeVision hiện tại.

## 7. Manifest nguồn và khả năng truy vết

Mỗi ảnh phải có một dòng trong `sources_manifest.csv` với tối thiểu:

- `challenge_image_id`;
- `local_filename`;
- `source_group_id`;
- `source_title`;
- `source_page_url`;
- `direct_download_url`;
- `creator`;
- `license_name`;
- `license_url`;
- `downloaded_at`;
- `sha256`;
- `intended_use` (`challenge_only` hoặc `adaptation_candidate`);
- `review_status`;
- `notes`.

Không dùng ảnh nếu thiếu trang nguồn, giấy phép hoặc tác giả khi giấy phép yêu cầu attribution.

## 8. Thu thập và trích frame

### 8.1. Ảnh tĩnh

1. Truy vấn nguồn theo nhóm: Việt Nam, đông người, tối, mưa, trẻ em, vật thể nhỏ và góc camera lạ.
2. Tải file gốc thay vì thumbnail nếu giấy phép cho phép.
3. Giữ nguyên file raw; mọi chuyển đổi nằm ở `candidates/`.
4. Tính SHA-256 và ghi metadata ngay khi tải.

### 8.2. Video

1. Ghi một `source_group_id` cho từng video.
2. Trích frame theo khoảng cách thời gian ban đầu khoảng 1–3 giây.
3. Không giữ nhiều frame gần như giống nhau chỉ để đủ số lượng.
4. Các frame từ cùng video luôn thuộc cùng source group.
5. Lưu timestamp frame vào manifest để truy vết về video gốc.

### 8.3. Công cụ dự kiến

- `tools/collect_challenge_images.py`: tải ảnh qua API/trang nguồn được phép và ghi manifest.
- `tools/extract_challenge_frames.py`: trích frame video, đặt tên theo source group và timestamp.
- `tools/dedupe_challenge_images.py`: tính hash chính xác và near-duplicate.
- `tools/build_challenge_contact_sheet.py`: tạo contact sheet để Long duyệt nhanh.

Mọi công cụ phải có chế độ dry-run, không ghi đè file và xuất log lỗi rõ ràng.

## 9. Lọc chất lượng và chống trùng lặp

### 9.1. Loại ngay

- file hỏng hoặc không đọc được;
- ảnh không có xe máy/người trên xe phù hợp phạm vi;
- độ phân giải quá thấp khiến không thể gán nhãn đầu người;
- ảnh trùng chính xác;
- nhiều frame liền nhau gần như giống hệt;
- giấy phép hoặc trang nguồn không đủ thông tin;
- nội dung có rủi ro riêng tư cao và không phù hợp sử dụng học thuật.

### 9.2. Kiểm tra trùng

- SHA-256 để phát hiện file trùng chính xác.
- Perceptual hash hoặc đặc trưng ảnh để tìm near-duplicate.
- So sánh với EdgeVision nếu ảnh EdgeVision cục bộ sẵn có.
- Tạo báo cáo ứng viên trùng; không xóa tự động nếu chưa duyệt.

### 9.3. Chọn 50 ảnh chính thức

- Xem contact sheet theo từng nhóm độ khó.
- Ưu tiên đa dạng nguồn, cảnh, thời gian, góc nhìn và loại mũ.
- Không để một video/series chiếm quá nhiều Challenge Set nếu còn nguồn khác.
- Gán `selection_reason` và các thẻ độ khó cho mỗi ảnh được giữ.

## 10. Quy chuẩn gán nhãn

### 10.1. Xác nhận semantics trước khi gán

Trước khi viết guideline, phải kiểm tra trực quan annotation EdgeVision hiện tại để xác nhận chính xác:

- `BikeWithRider` bao quanh xe và người ở phạm vi nào;
- `Helmet` bao quanh chiếc mũ hay vùng đầu có mũ;
- `NoHelmet` bao quanh đầu hay toàn người không đội mũ;
- trẻ em và người ngồi sau có được gán nhãn hay không;
- cách xử lý đối tượng bị cắt mép hoặc che khuất.

Không tự suy đoán semantics từ tên lớp.

### 10.2. Nguyên tắc

- Dùng đúng category ID hiện tại: `1 BikeWithRider`, `2 NoHelmet`, `3 Helmet`.
- Bounding box bám sát đối tượng theo policy đã xác nhận.
- Không tạo lớp mới như `DriverNoHelmet` trong Challenge Set v1.
- Không dùng dự đoán của Faster R-CNN/RetinaNet làm ground truth mà không có người kiểm tra.
- Có thể dùng hợp của dự đoán hai mô hình để gợi ý box, sau đó người gán nhãn phải sửa và xác nhận từng box.
- Các trường hợp không chắc chắn phải ghi `ambiguous_reason`, không ép gán nhãn.

### 10.3. Kiểm tra annotation

- Long hoặc người được giao duyệt 100% ảnh sau gán nhãn.
- Một thành viên khác kiểm tra chéo tối thiểu 20% ảnh và toàn bộ trường hợp mơ hồ.
- Render annotation và xem lại trực quan trước khi khóa.
- Kiểm tra category ID, box ngoài ảnh, box có diện tích bằng 0, ID trùng và ảnh thiếu annotation.

## 11. Gắn thẻ độ khó

Mỗi ảnh có thể có nhiều thẻ:

- `crowded`;
- `child_passenger`;
- `small_object`;
- `occlusion`;
- `low_light`;
- `backlight`;
- `rain_or_glare`;
- `motion_blur`;
- `compression_noise`;
- `unusual_helmet`;
- `hat_or_cap_confusion`;
- `high_angle`;
- `oblique_view`;
- `edge_truncation`;
- `overlapping_boxes`.

Thẻ phải dựa trên tiêu chí mô tả trong `difficulty_tags.csv`; không gắn tùy ý chỉ dựa trên cảm giác “ảnh khó”.

## 12. Đóng băng Challenge Set v1

Sau khi annotation được duyệt:

1. Xuất `challenge_v1.coco.json`.
2. Tạo manifest gồm danh sách ảnh, SHA-256 ảnh, SHA-256 annotation, nguồn và source group.
3. Ghi số ảnh và số box từng lớp.
4. Chạy validator và render mẫu.
5. Đánh dấu `status: frozen` và ngày khóa.
6. Không thay ảnh/annotation sau khi xem kết quả mô hình; nếu phát hiện lỗi ground truth, sửa có log và tăng version.

## 13. Giao thức đánh giá hai mô hình

### 13.1. Điều kiện công bằng

- Cùng 50 ảnh và cùng annotation.
- Cùng evaluator COCO và IoU config.
- Cùng kích thước xử lý theo pipeline hiện tại.
- Không thay backbone, checkpoint hoặc hậu xử lý chỉ cho một mô hình.
- Không chọn confidence threshold mới trên Challenge Set.

### 13.2. Chỉ số

- mAP@0.5:0.95;
- mAP@0.5;
- AP theo `BikeWithRider`, `NoHelmet`, `Helmet`;
- Precision, Recall và F1 của `NoHelmet` tại IoU 0,50 với threshold đã khóa từ validation;
- số false positive và false negative theo nhóm độ khó;
- latency/FPS chỉ ghi như quan sát bổ sung nếu đo theo đúng protocol benchmark.

### 13.3. Cách báo cáo

- Bảng kết quả EdgeVision test và Challenge Set phải tách riêng.
- Không gộp hai tập thành một con số mAP chung.
- Không kết luận mô hình tốt hơn chỉ từ 50 ảnh; nêu đây là phân tích robustness quy mô nhỏ.
- Chọn một số ảnh đúng, bỏ sót và phát hiện nhầm để phân tích nguyên nhân.

## 14. Phân tích lỗi

Mỗi lỗi được ghi tối thiểu:

- model;
- image ID;
- difficulty tag;
- lớp ground truth;
- loại lỗi: `false_positive`, `false_negative`, `wrong_class`, `poor_localization`;
- confidence;
- IoU tốt nhất;
- nhận xét nguyên nhân có thể;
- ảnh render minh họa.

Các nhóm cần phân tích riêng:

- mũ nửa đầu/nón dễ nhầm;
- trẻ em và người ngồi sau;
- người nhỏ/xa;
- cảnh đông và box chồng lấn;
- thiếu sáng/ngược sáng;
- che khuất/mờ;
- box `NoHelmet` không thể hiện quan hệ ai là người điều khiển.

## 15. Điều kiện chuyển sang fine-tune bổ sung

Chỉ triển khai Pha B khi thỏa đồng thời:

- Challenge Set v1 đã khóa và có kết quả cho cả hai mô hình;
- lỗi xuất hiện lặp lại theo một hoặc nhiều nhóm độ khó;
- có khả năng thu thập thêm tối thiểu 150 ảnh adaptation không trùng Challenge Set;
- nhóm còn thời gian gán nhãn và kiểm tra chất lượng;
- cả hai mô hình sẽ được fine-tune trong cùng điều kiện;
- kết quả baseline hiện tại được giữ nguyên để so sánh trước/sau.

Không fine-tune chỉ dựa trên vài ảnh demo trông chưa đẹp.

## 16. Phương án Pha B nếu được duyệt

### 16.1. Dữ liệu

- `adaptation_train`: dữ liệu mới dùng cập nhật trọng số.
- `adaptation_val`: dữ liệu mới dùng chọn epoch/checkpoint/threshold.
- `challenge_v1`: giữ nguyên, chỉ đánh giá cuối.
- Nhóm theo nguồn/video trước khi chia.

### 16.2. Huấn luyện

- Tạo config phiên bản mới, không sửa trực tiếp config baseline đã dùng.
- Chọn rõ chiến lược: warm-start từ checkpoint baseline hoặc train lại từ cùng pretrained weights.
- Áp dụng cùng chiến lược cho hai model.
- Learning rate, epoch và scheduler được chọn trên validation, không dựa trên Challenge Set.
- Lưu checkpoint, log và manifest vào thư mục mới.

### 16.3. Đánh giá sau adaptation

- EdgeVision test: kiểm tra có làm giảm chất lượng miền gốc hay không.
- Challenge Set v1: kiểm tra cải thiện miền ảnh khó.
- So sánh before/after cho từng model và từng nhóm độ khó.
- Chỉ kết luận cải thiện khi số liệu thực tế hỗ trợ.

## 17. Quyền riêng tư và sử dụng ảnh

- Không commit ảnh/video chứa người lên GitHub mặc định.
- Không dùng ảnh cá nhân riêng tư nếu chưa có sự đồng ý phù hợp.
- Biển số và khuôn mặt chỉ làm mờ khi việc làm mờ không phá hỏng mục tiêu nhận diện; mọi biến đổi phải có log.
- Ảnh chèn báo cáo phải ghi nguồn/giấy phép hoặc “Nguồn: Nhóm tác giả thu thập”.
- Tuân thủ điều kiện attribution, share-alike hoặc hạn chế phân phối của từng nguồn.

## 18. Kiểm thử kỹ thuật cần có

- Kiểm thử parser manifest nguồn.
- Kiểm thử không chấp nhận ảnh thiếu license/source.
- Kiểm thử SHA-256 và phát hiện file trùng.
- Kiểm thử COCO schema và category mapping.
- Kiểm thử box nằm trong kích thước ảnh.
- Kiểm thử một ảnh chỉ xuất hiện một lần trong Challenge Set.
- Kiểm thử `source_group_id` đầy đủ.
- Chạy thử evaluator bằng prediction giả trước khi dùng checkpoint thật.
- Chạy toàn bộ `pytest` sau khi thêm công cụ.

## 19. Sản phẩm bàn giao

### Sau Pha A

- Cấu trúc `data/challenge/` và `.gitignore` an toàn.
- Khoảng 50 ảnh Challenge Set v1 lưu cục bộ.
- Manifest nguồn và attribution.
- COCO annotation ba lớp đã duyệt.
- Metadata nhóm độ khó và source group.
- Báo cáo duplicate/quality.
- Kết quả Faster R-CNN và RetinaNet trên Challenge Set.
- Bảng so sánh và bộ ảnh phân tích lỗi.
- Đoạn nội dung báo cáo về robustness trong bối cảnh ảnh khó.

### Sau Pha B, nếu thực hiện

- Dataset adaptation riêng và split manifest.
- Config huấn luyện phiên bản mới.
- Checkpoint/log before–after của cả hai model.
- Bảng đánh giá EdgeVision test và Challenge Set v1.
- Cập nhật demo sử dụng checkpoint mới chỉ sau khi kết quả được duyệt.

## 20. Lịch thực hiện đề xuất

| Ngày | Công việc | Cổng nghiệm thu |
|---|---|---|
| 28/08 | Duyệt kế hoạch, tạo cấu trúc và manifest; xác nhận policy nhãn EdgeVision | Chưa tải khi chưa duyệt; policy nhãn rõ |
| 29/08 | Thu thập 80–100 ứng viên, tạo contact sheet và lọc nguồn | Nguồn/giấy phép đầy đủ |
| 30/08 | Chọn khoảng 50 ảnh, gắn thẻ độ khó, dedupe | Đủ bao phủ, không trùng đáng kể |
| 31/08 | Gán nhãn, render và kiểm tra chéo | Annotation COCO hợp lệ |
| 01/09 | Khóa Challenge Set v1, đánh giá hai model | Có metric và artifact kiểm chứng |
| 02/09 | Phân tích lỗi, cập nhật báo cáo và ảnh minh họa | Hoàn thành Pha A trước hạn kỹ thuật |
| Sau Pha A | Quyết định có làm adaptation hay không | Không tự động train lại |

Nếu thời gian thực tế ngắn hơn dự kiến, ưu tiên hoàn thành Challenge Set v1 có nguồn và annotation tốt; không giảm kiểm tra chất lượng để chạy fine-tune vội.

## 21. Cổng duyệt

### Cổng 0 — Duyệt kế hoạch

- Long xác nhận phạm vi Pha A trước, Pha B sau.
- Xác nhận nguồn ảnh ưu tiên và có/không có video nhóm tự quay.
- Xác nhận không commit ảnh lên GitHub.

### Cổng 1 — Duyệt nguồn và ảnh ứng viên

- Xem contact sheet và manifest.
- Loại nguồn không phù hợp hoặc lệch đề tài.

### Cổng 2 — Duyệt 50 ảnh được chọn

- Kiểm tra mức độ khó, tính đa dạng và bối cảnh Việt Nam.
- Chưa chạy model chính thức trước khi khóa lựa chọn.

### Cổng 3 — Duyệt annotation

- Render 100% ảnh.
- Sửa mọi box/nhãn chưa thống nhất và ghi log.

### Cổng 4 — Khóa và đánh giá

- Tạo hash dataset/annotation.
- Chạy hai model bằng cùng protocol.

### Cổng 5 — Quyết định adaptation

- Dựa trên bảng lỗi thực tế để quyết định dừng ở phân tích robustness hay thu thập thêm dữ liệu và fine-tune.

## 22. Tiêu chí hoàn thành Pha A

- [ ] Khoảng 50 ảnh đã được Long duyệt.
- [ ] Mọi ảnh có nguồn và giấy phép truy vết được.
- [ ] Không có ảnh trùng chính xác; near-duplicate đã được xem xét.
- [ ] Ba lớp và category ID khớp EdgeVision.
- [ ] Annotation đã render và duyệt 100%.
- [ ] Challenge Set v1 có manifest và hash đã khóa.
- [ ] Cả hai checkpoint chạy trên cùng dữ liệu/evaluator.
- [ ] Metric Challenge Set được tách khỏi EdgeVision test.
- [ ] Có phân tích lỗi theo nhóm độ khó.
- [ ] Không dùng Challenge Set để chọn threshold hoặc chỉnh model.
- [ ] Không có ảnh/video nặng bị commit lên GitHub.
- [ ] Báo cáo ghi đúng giới hạn quy mô và nguồn dữ liệu.

## 23. Các điểm Long cần xác nhận trước khi triển khai

1. Duyệt hướng **Pha A: chỉ xây và đánh giá Challenge Set v1 trước; chưa train lại**.
2. Duyệt mục tiêu khoảng 50 ảnh chính thức từ 80–100 ứng viên.
3. Duyệt ưu tiên nguồn: ảnh/video nhóm tự có → Wikimedia Commons → dataset nghiên cứu có giấy phép.
4. Xác nhận nhóm có video/ảnh giao thông Việt Nam tự quay để bổ sung hay không.
5. Duyệt quy tắc không commit ảnh/video lên GitHub; chỉ commit manifest, annotation, công cụ và tài liệu.
6. Xác nhận một thành viên ngoài Long có thể kiểm tra chéo tối thiểu 20% annotation hay không.

Chỉ sau khi các điểm trên được Long duyệt mới bắt đầu tạo thư mục, sửa `.gitignore`, tải ảnh hoặc viết công cụ thu thập.

## 24. Liên kết nguồn ứng viên để kiểm tra khi triển khai

- Wikimedia Commons, video xe máy tại Huế: <https://commons.wikimedia.org/wiki/File:Veturado_sur_motorciklo_apud_Hue.webm>
- AI City Challenge 2023, bài tổng quan Track 5: <https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Naphade_The_7th_AI_City_Challenge_CVPRW_2023_paper.pdf>
- Mã nguồn giải pháp Track 5 của VNPT AI: <https://github.com/vnptai/AI-City-Challenge-2023>
- Helmet Wearing Image Dataset trên Mendeley Data: <https://data.mendeley.com/datasets/tm72fkfxd5/3>
- HCMCTrafficDataset trên kho dữ liệu UEH: <https://digital.lib.ueh.edu.vn/handle/UEH/78522?mode=full>

Các liên kết này chỉ là danh sách ứng viên. Khi triển khai, giấy phép phải được kiểm tra ở cấp từng file/dataset tại thời điểm tải và được ghi vào manifest; việc xuất hiện trong danh sách không đồng nghĩa tự động được phép tải hoặc phân phối lại.
