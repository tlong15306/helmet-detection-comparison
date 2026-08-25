# Kế hoạch xử lý annotation và tạo tập dữ liệu cố định

## 1. Trạng thái và phạm vi

- Người phụ trách duyệt: **Nguyễn Thành Long**.
- Trạng thái: **Chờ Long duyệt, chưa thực hiện thay đổi dữ liệu**.
- Mục tiêu hoàn thành: **trước khi chạy smoke test và huấn luyện hai mô hình**.
- Thời gian dự kiến cho cổng dữ liệu: **26/08/2026 - 27/08/2026**.
- Phạm vi: kiểm định nhãn EdgeVision, tạo bản annotation đã xử lý, kiểm tra trùng lặp/rò rỉ dữ liệu, tạo một bộ chia train/validation/test cố định và kiểm chứng bộ chia.
- Ngoài phạm vi của bước này: fine-tune, lựa chọn mô hình tốt hơn, công bố mAP/FPS và viết kết luận thực nghiệm.

## 2. Vì sao đây là phần phải làm tiếp theo

Theo `KE_HOACH_HE_THONG.md`, sau khi tiếp nhận dataset, cổng tiếp theo là kiểm tra nhãn và tạo bộ chia cố định trước khi smoke test. Trạng thái hiện tại:

- Có 2.392 ảnh đọc được và tệp `annotations.json`.
- Có 8.275 khung giới hạn thuộc ba lớp: `BikeWithRider`, `NoHelmet` và `Helmet`.
- Không phát hiện ảnh thiếu, ảnh hỏng hoặc ảnh có kích thước không khớp metadata.
- Có 1 annotation có chiều cao bằng 0: ID `2107`, image ID `566`, bbox `[524, 452, 1, 0]`.
- Có 78 bbox vượt biên ảnh; trong đó 40 bbox vượt tối đa 20 pixel và 38 bbox vượt hơn 20 pixel.
- Một số bbox vượt biên rất lớn, tối đa 1.771 pixel. Đây có thể là lỗi tọa độ, xoay ảnh hoặc metadata và không được phép tự động cắt biên hàng loạt.
- `tools/create_splits.py` hiện chỉ là tệp giữ chỗ, chưa tạo split thật.

Nếu huấn luyện ngay, lỗi nhãn có thể làm nhiễu loss và metric. Nếu chia ngẫu nhiên từng ảnh mà không xét cảnh/chuỗi, các khung hình gần giống nhau có thể xuất hiện ở cả train và test, làm kết quả đánh giá lạc quan giả tạo.

## 3. Các nguyên tắc cố định

1. Không sửa `data/raw/edgevision/annotations.json` và ảnh gốc.
2. Mọi thay đổi phải được thực hiện trên bản sao dưới `data/processed/edgevision/` và có nhật ký.
3. Giữ nguyên image ID, annotation ID và category ID khi có thể; không đánh lại ID chỉ vì loại một annotation.
4. Faster R-CNN và RetinaNet phải sử dụng đúng cùng một bộ chia dữ liệu.
5. Tập validation dùng để theo dõi và chọn checkpoint; tập test chỉ dùng sau khi chốt cấu hình.
6. Không công bố metric nếu bộ chia chưa vượt qua toàn bộ kiểm tra rò rỉ và tính hợp lệ.
7. Seed mặc định theo kế hoạch tổng là `42`; tỷ lệ mục tiêu là `70%/15%/15%` theo số ảnh, có xét phân bố lớp và nhóm cảnh.

## 4. Luồng thực hiện

```text
Annotation gốc (chỉ đọc)
          |
          v
Kiểm tra tự động + kiểm tra trực quan
          |
          v
Long duyệt quy tắc xử lý lỗi
          |
          v
Annotation processed + nhật ký thay đổi
          |
          v
Hash/trùng lặp + nhóm ảnh theo cảnh
          |
          v
Tạo split theo nhóm, seed 42
          |
          v
Kiểm chứng không rò rỉ + thống kê lớp
          |
          v
Long duyệt và đóng băng split
          |
          v
Smoke test Faster R-CNN và RetinaNet
```

## 5. Giai đoạn A - Kiểm định annotation

### A1. Xuất danh sách lỗi có thể kiểm tra

Mở rộng công cụ kiểm tra để tạo danh sách chi tiết cho từng annotation bất thường, gồm:

- annotation ID, image ID và tên ảnh;
- category ID và tên lớp;
- kích thước ảnh;
- bbox gốc;
- loại lỗi và số pixel vượt từng biên;
- đường dẫn ảnh minh họa có bbox được vẽ lên.

Đầu ra dự kiến:

- `outputs/dataset_quality/problem_annotations.json`;
- `outputs/dataset_quality/problem_annotations.csv`;
- `outputs/dataset_quality/previews/`.

### A2. Kiểm tra trực quan

Phạm vi kiểm tra tối thiểu:

- toàn bộ 1 bbox có kích thước không hợp lệ;
- toàn bộ 38 bbox vượt biên nghiêm trọng;
- toàn bộ 40 bbox vượt biên nhẹ;
- tối thiểu 30 annotation hợp lệ của mỗi lớp để xác nhận ý nghĩa nhãn và chất lượng bbox;
- một số ảnh có nhiều lớp cùng xuất hiện để kiểm tra quan hệ giữa người lái, xe máy và mũ bảo hiểm.

Mỗi trường hợp lỗi được gán một quyết định: `keep`, `clip`, `correct`, `exclude_annotation`, `exclude_image` hoặc `needs_review`.

### A3. Xác nhận nguyên nhân lỗi nghiêm trọng

Với 38 bbox vượt biên lớn, cần đối chiếu ảnh thật và metadata để trả lời:

- bbox có được tạo theo ảnh trước khi xoay hay không;
- chiều rộng/chiều cao có bị đảo hay không;
- tọa độ có theo định dạng khác COCO `[x, y, width, height]` hay không;
- lỗi chỉ nằm ở bbox hay toàn bộ annotation của ảnh;
- có tồn tại một phép biến đổi xác định, áp dụng đúng cho tất cả trường hợp cùng loại hay không.

Chỉ sửa tọa độ khi phép biến đổi được chứng minh bằng ảnh và kiểm tra lại trực quan. Nếu không xác định chắc chắn, loại annotation hoặc ảnh khỏi bản processed và ghi rõ lý do; không đoán tọa độ.

### Cổng duyệt A

Long xem bảng lỗi và ảnh minh họa, sau đó duyệt quy tắc cho từng nhóm lỗi. Chưa tạo bản processed chính thức trước cổng này.

## 6. Giai đoạn B - Tạo annotation processed có thể truy vết

### B1. Quy tắc xử lý dự kiến

- Bbox có `width <= 0` hoặc `height <= 0`: loại annotation khỏi bản processed và ghi log.
- Bbox vượt biên nhẹ: chỉ clip về biên ảnh nếu kiểm tra trực quan cho thấy đối tượng đúng và phần vượt biên là sai số làm tròn/gán nhãn.
- Bbox vượt biên nghiêm trọng: không auto-clip. Chỉ sửa khi tìm được phép biến đổi chắc chắn; nếu không, loại khỏi tập đủ điều kiện và ghi log.
- Annotation đúng: giữ nguyên.
- Ảnh không còn annotation hợp lệ: đánh dấu rõ là ảnh âm hay ảnh bị loại, không tự suy diễn.

Ngưỡng “vượt biên nhẹ” 20 pixel hiện chỉ dùng để phân nhóm kiểm tra, chưa phải quy tắc sửa tự động. Ngưỡng cuối cùng phải dựa trên bằng chứng trực quan và được Long duyệt.

### B2. Đầu ra dự kiến

- `data/processed/edgevision/annotations.json`: COCO JSON dùng cho bước chia dữ liệu;
- `data/processed/edgevision/annotation_changes.json`: nhật ký trước/sau;
- `outputs/dataset_quality/processed_summary.json`: số lượng giữ, sửa, loại theo lớp và lý do;
- `outputs/dataset_quality/processed_summary.md`: bản tóm tắt để nhóm đưa vào phụ lục nếu cần.

Mỗi bản ghi thay đổi cần có: annotation ID, image ID, hành động, lý do, bbox gốc, bbox mới, thời điểm tạo và phiên bản công cụ.

### B3. Điều kiện đạt

- Không còn bbox có kích thước không dương.
- Tất cả bbox processed nằm trong kích thước ảnh.
- Không mất category hoặc đổi ánh xạ lớp ngoài ý muốn.
- COCO JSON đọc được bằng dataset loader của dự án.
- Tổng số annotation trước/sau và mọi chênh lệch đều giải thích được bằng nhật ký.

## 7. Giai đoạn C - Phát hiện trùng lặp và nhóm ảnh theo cảnh

### C1. Trùng lặp chính xác

- Tính SHA-256 cho từng ảnh.
- Các ảnh có cùng hash phải nằm trong cùng một nhóm.
- Nếu loại bản trùng, giữ một bản đại diện và ghi danh sách alias; không xóa ảnh raw.

### C2. Ảnh gần trùng lặp

- Tính perceptual hash để tạo danh sách ứng viên gần giống.
- Chưa ấn định ngưỡng loại tự động trước khi xem phân bố khoảng cách hash.
- Các cặp gần ngưỡng phải được xem ảnh ghép cạnh nhau.

### C3. Nhóm cảnh/chuỗi

- Kiểm tra tên tệp, thứ tự ảnh và nội dung trực quan để nhận biết ảnh cùng camera, cùng đoạn video hoặc các khung hình liên tiếp.
- Không giả định các tên `Image_xxxxx` chắc chắn là cùng chuỗi nếu chưa có bằng chứng.
- Toàn bộ ảnh thuộc cùng cảnh/chuỗi phải có chung `group_id` và được đưa nguyên nhóm vào một split.

Đầu ra dự kiến:

- `data/processed/edgevision/image_hashes.json`;
- `data/processed/edgevision/group_manifest.json`;
- `outputs/dataset_quality/near_duplicate_candidates.json`.

### Cổng duyệt C

Long duyệt quy tắc nhóm và các trường hợp gần trùng lặp chưa rõ trước khi chạy bộ chia cuối cùng.

## 8. Giai đoạn D - Cài đặt công cụ tạo split

Hoàn thiện `tools/create_splits.py` với các đầu vào rõ ràng:

- annotation processed;
- group manifest;
- tỷ lệ `0.70/0.15/0.15`;
- seed `42`;
- thư mục đầu ra.

Yêu cầu của thuật toán:

1. Chia theo `group_id`, không chia riêng từng ảnh.
2. Cố gắng đưa số ảnh về gần tỷ lệ mục tiêu.
3. Cố gắng giữ phân bố ba lớp gần nhau giữa các split.
4. Cho cùng kết quả khi dùng cùng dữ liệu, cấu hình và seed.
5. Dừng với thông báo rõ ràng nếu phát hiện ảnh thiếu nhóm, ID trùng hoặc annotation không hợp lệ.

Đầu ra dự kiến:

- `data/splits/train.json`;
- `data/splits/val.json`;
- `data/splits/test.json`;
- `data/splits/split_manifest.json`;
- `outputs/dataset_quality/split_summary.json`;
- `outputs/dataset_quality/split_summary.md`.

Mỗi tệp split phải giữ đủ metadata và danh mục lớp cần thiết để có thể dùng trực tiếp với loader.

## 9. Giai đoạn E - Kiểm chứng và đóng băng split

### E1. Kiểm tra bắt buộc

- Không giao nhau về image ID, tên tệp, SHA-256 và `group_id` giữa train/validation/test.
- Hợp của ba tập bằng đúng tập ảnh đủ điều kiện trong annotation processed.
- Mỗi ảnh và annotation xuất hiện đúng số lần quy định.
- Không còn bbox không hợp lệ hoặc vượt biên.
- Cả ba lớp đều có mặt trong từng split, nếu cấu trúc dataset cho phép.
- Báo cáo số ảnh, số bbox và tỷ lệ từng lớp theo split.
- Loader đọc được mẫu từ cả ba tập.
- Faster R-CNN và RetinaNet nhận cùng ánh xạ category và số lớp.

### E2. Kiểm thử mã nguồn

Bổ sung kiểm thử tự động tối thiểu cho:

- tính xác định với cùng seed;
- không rò rỉ group giữa các split;
- phát hiện dữ liệu đầu vào sai;
- tính toàn vẹn của COCO JSON đầu ra;
- tổng ảnh/annotation không bị mất ngoài danh sách loại đã ghi log;
- loader đọc được split sinh từ fixture nhỏ.

### E3. Đóng băng

Sau khi Long duyệt:

- tính SHA-256 cho annotation processed, group manifest và ba tệp split;
- lưu các hash cùng cấu hình chia vào `split_manifest.json`;
- không tạo lại split sau khi smoke test nếu không có lý do được ghi nhận;
- nếu buộc phải thay đổi, tăng phiên bản split và đánh dấu toàn bộ kết quả cũ không còn so sánh trực tiếp được.

Theo `.gitignore` hiện tại, các JSON trong `data/splits/` chưa được đưa lên Git. Ở bước duyệt cuối, Long cần quyết định có lưu các tệp split nhẹ trong repository để tái lập hay chỉ lưu manifest/hash và hướng dẫn tái tạo. Không thay đổi quy tắc Git trước khi có quyết định này.

## 10. Sản phẩm bàn giao của phần này

| Sản phẩm | Mục đích | Được đưa lên Git |
|---|---|---|
| Công cụ kiểm tra annotation | Tạo danh sách lỗi và bằng chứng trực quan | Có |
| Công cụ tạo processed annotation | Áp dụng quy tắc đã duyệt, ghi log | Có |
| Công cụ hash/nhóm cảnh | Chống trùng lặp và rò rỉ dữ liệu | Có |
| `tools/create_splits.py` hoàn chỉnh | Tạo split xác định theo nhóm | Có |
| Kiểm thử tự động | Chứng minh logic xử lý và chia dữ liệu | Có |
| Annotation processed | Đầu vào sạch cho mô hình | Không mặc định; dữ liệu cục bộ |
| Tệp train/val/test | Split dùng chung cho hai mô hình | Chờ Long quyết định |
| Báo cáo chất lượng và hash | Truy vết phiên bản dữ liệu | Có nếu không chứa dữ liệu nhạy cảm/lớn |

## 11. Tiêu chí hoàn thành

- [ ] Long đã duyệt cách xử lý 1 bbox không hợp lệ và 78 bbox vượt biên.
- [ ] Toàn bộ annotation lỗi có ảnh minh họa và quyết định truy vết được.
- [ ] Bản processed vượt qua kiểm tra hình học và COCO schema.
- [ ] Đã kiểm tra trùng lặp chính xác và ứng viên gần trùng lặp.
- [ ] Mọi ảnh đủ điều kiện có `group_id`.
- [ ] Bộ chia 70/15/15, seed 42 được tạo theo nhóm.
- [ ] Không có image/hash/group giao nhau giữa ba tập.
- [ ] Thống kê lớp của từng split đã được xem xét.
- [ ] Loader đọc được cả ba split.
- [ ] Test tự động liên quan đều đạt.
- [ ] Long đã duyệt và hash của split đã được đóng băng.

## 12. Điểm Long cần duyệt

1. Duyệt bằng chứng trực quan và hành động cho từng nhóm annotation lỗi.
2. Duyệt cách nhóm cảnh/chuỗi và danh sách gần trùng lặp khó phân loại.
3. Duyệt thống kê cuối cùng của train/validation/test.
4. Chọn có đưa các JSON split lên Git hay chỉ đưa công cụ, manifest và hash.
5. Cho phép đóng băng split để chuyển sang smoke test hai mô hình.

## 13. Bước ngay sau khi kế hoạch này hoàn thành

Chạy smoke test Faster R-CNN và RetinaNet trên một tập con nhỏ của **cùng split đã đóng băng**. Smoke test chỉ kiểm tra pipeline có thể đọc dữ liệu, forward, backward, lưu checkpoint và đánh giá; không dùng kết quả smoke test để kết luận mô hình nào tốt hơn.
