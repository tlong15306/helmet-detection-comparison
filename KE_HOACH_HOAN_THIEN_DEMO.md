# KẾ HOẠCH HOÀN THIỆN DEMO VÀ BÀN GIAO BTL

## 1. Thông tin kiểm soát

- **Đề tài:** Ứng dụng và so sánh Faster R-CNN và RetinaNet trong phát hiện người điều khiển xe máy không đội mũ bảo hiểm từ hình ảnh giao thông.
- **Người phụ trách demo:** Nguyễn Thành Long.
- **Ngày lập kế hoạch:** 27/08/2026.
- **Hạn kỹ thuật nội bộ:** hoàn thành trước 03/09/2026 để nhóm còn thời gian ghép báo cáo và chuẩn bị trình bày.
- **Trạng thái:** Đang hoàn thiện sau khi Long đã duyệt triển khai.
- **Phạm vi:** hoàn thiện video, camera snapshot, kiểm thử, tài liệu, ảnh minh họa và đưa mã nguồn lên GitHub.
- **Không thuộc phạm vi:** train lại trên nhiều dataset, webcam live tốc độ cao, tracking, nhận dạng biển số, đăng nhập, cơ sở dữ liệu, ONNX/TensorRT và triển khai Internet.

## 2. Hiện trạng đã hoàn thành

- Dataset EdgeVision đã được xử lý và chia cố định train/validation/test.
- Faster R-CNN và RetinaNet đã có checkpoint tốt nhất.
- Hai mô hình đã được đánh giá trên cùng tập test.
- Threshold demo được chọn trên validation:
  - Faster R-CNN: `0,85`.
  - RetinaNet: `0,60`.
- Đã có benchmark latency/FPS riêng cho báo cáo.
- Frontend React đã có ba chế độ: ảnh, video và camera.
- FastAPI, model loader và pipeline suy luận ảnh đã hoạt động với checkpoint thật.
- Chuyển đổi giữa hai mô hình đã được kiểm tra tuần tự để tránh giữ đồng thời hai checkpoint trong VRAM.
- Toàn bộ bộ kiểm thử hiện tại đạt `46 passed, 1 skipped`.
- Frontend production build thành công.
- Bản Word mục 2.1.2 của Long và các biểu đồ so sánh đã có.

## 3. Mục tiêu đầu ra

Khi hoàn tất kế hoạch này, ứng dụng phải cho phép:

1. Tải ảnh và nhận kết quả phát hiện bằng Faster R-CNN hoặc RetinaNet.
2. Tải video, theo dõi tiến độ xử lý và tải video kết quả.
3. Mở camera trong trình duyệt, chụp một ảnh và chạy cùng pipeline suy luận ảnh.
4. Hiển thị bounding box, nhãn lớp, confidence, số detection và latency quan sát được.
5. Đổi mô hình mà không gây hết VRAM.
6. Báo lỗi rõ ràng khi backend tắt, checkpoint thiếu, tệp hỏng, định dạng không hỗ trợ hoặc camera bị từ chối quyền.
7. Có ảnh chụp giao diện và ví dụ định tính phục vụ báo cáo.
8. Có hướng dẫn cài đặt, bật/tắt ứng dụng và chạy kiểm thử.

## 4. Nguyên tắc kỹ thuật

- React chỉ đảm nhiệm giao diện; PyTorch/Torchvision tiếp tục chạy trong FastAPI.
- Ảnh, video và camera dùng chung model loader, threshold, class mapping và hàm vẽ bounding box.
- Chỉ một mô hình được giữ trên GPU tại một thời điểm.
- Không sử dụng test để điều chỉnh threshold hoặc tính năng demo.
- Latency/FPS hiển thị trong demo chỉ là số quan sát của lần chạy hiện tại, không thay thế benchmark chính thức.
- Không đưa số liệu giả vào giao diện hoặc báo cáo.
- Các tệp video tạm và kết quả sinh ra không được commit lên GitHub.
- Demo tiếp tục chạy cục bộ vì checkpoint và GPU nằm trên máy của Long; không triển khai lên Internet trong phạm vi bắt buộc.

## 5. Các giai đoạn thực hiện

### Giai đoạn 0 — Khóa mốc demo ảnh

#### Công việc

- Kiểm tra `git diff` và tách thay đổi thuộc demo khỏi dữ liệu/checkpoint/artifact lớn.
- Chạy lại toàn bộ Python test và frontend build.
- Kiểm tra `.gitignore` loại trừ `.venv`, `node_modules`, dataset, checkpoint, video tạm và kết quả sinh tự động.
- Commit mã nguồn demo ảnh, test, cấu hình threshold và tài liệu hiện có trực tiếp lên `main` theo quy trình nhóm đã chốt.
- Push lên GitHub và xác nhận repository hiển thị thư mục `frontend/`, `app/api.py`, `src/infer.py` và tài liệu chạy.

#### Cổng nghiệm thu

- Working tree không còn thay đổi thuộc mốc demo ảnh chưa được lưu.
- Không có dataset, checkpoint, `.venv` hoặc `node_modules` trong commit.
- GitHub có commit ổn định để quay lại nếu phần video phát sinh lỗi.

### Giai đoạn 1 — Backend xử lý video

#### Thiết kế

Video không chạy trong một request kéo dài. Backend tạo một tác vụ cục bộ và trả `job_id` để frontend theo dõi.

API dự kiến:

- `POST /api/infer/video`: nhận video, model và threshold; trả `job_id`.
- `GET /api/infer/video/jobs/{job_id}`: trả trạng thái `queued/processing/completed/failed`, phần trăm tiến độ và thống kê.
- `GET /api/infer/video/jobs/{job_id}/download`: tải video kết quả khi hoàn thành.
- `DELETE /api/jobs/{job_id}`: dọn tệp tạm sau khi người dùng không cần nữa; chỉ triển khai nếu không làm phức tạp luồng chính.

#### Quy tắc xử lý

- Kiểm tra phần mở rộng, MIME type và khả năng mở video bằng OpenCV.
- Đọc FPS, kích thước, codec và tổng số frame.
- Xử lý tuần tự từng frame với batch size 1.
- Dùng đúng threshold validation mặc định hoặc threshold do người dùng điều chỉnh.
- Vẽ bounding box bằng cùng màu lớp của chế độ ảnh.
- Ghi tệp đầu ra theo FPS và kích thước gốc khi khả thi.
- Ưu tiên MP4 phát được trên trình duyệt; nếu codec máy không hỗ trợ thì vẫn cung cấp tệp tải xuống và thông báo rõ.
- Chỉ cho phép một job GPU chạy tại một thời điểm; job sau chờ trong hàng đợi.
- Không giữ toàn bộ frame trong RAM.
- Luôn đóng `VideoCapture` và `VideoWriter`, kể cả khi có lỗi.

#### Giới hạn đề xuất cho BTL

- Dung lượng upload tối đa: `200 MB`.
- Thời lượng khuyến nghị: tối đa `5 phút`.
- Không yêu cầu xử lý thời gian thực.
- Các giới hạn này phải hiển thị trên giao diện và có thể điều chỉnh sau nếu kiểm thử phần cứng cho thấy phù hợp.

#### Cổng nghiệm thu

- Video ngắn chạy được với cả hai mô hình.
- Tiến độ tăng theo số frame thực tế.
- Tệp kết quả tải xuống và mở được.
- RAM/VRAM không tăng liên tục theo số frame.
- Dừng hoặc lỗi giữa chừng không để khóa file/video handle.

### Giai đoạn 2 — Frontend video

#### Công việc

- Cho phép kéo thả MP4/MOV/AVI và xem trước video đầu vào.
- Hiển thị tên tệp, dung lượng, thời lượng, FPS và kích thước nếu đọc được.
- Gửi model, threshold và video sang FastAPI.
- Poll trạng thái job theo chu kỳ khoảng một giây; chỉ dùng WebSocket nếu polling không đáp ứng.
- Hiển thị progress bar, số frame đã xử lý và trạng thái lỗi.
- Khi hoàn tất, hiển thị video kết quả nếu trình duyệt hỗ trợ.
- Cung cấp nút tải tệp kết quả trong mọi trường hợp hoàn thành.
- Hiển thị tổng số detection theo lớp, thời gian xử lý và FPS quan sát của job.
- Ghi chú rõ FPS của job video không phải benchmark chính thức trong báo cáo.

#### Cổng nghiệm thu

- Không làm treo giao diện trong thời gian xử lý.
- Reload trang không gây chạy lại job ngoài ý muốn.
- Nút chạy bị khóa khi tệp không hợp lệ hoặc backend chưa kết nối.
- Lỗi backend được dịch thành thông báo tiếng Việt dễ hiểu.

### Giai đoạn 3 — Camera snapshot

#### Phạm vi bắt buộc

- Dùng `navigator.mediaDevices.getUserMedia()` để hiển thị preview camera.
- Có nút mở camera, chụp ảnh, chụp lại và dừng camera.
- Frame được chụp thành JPEG/PNG rồi gửi tới endpoint suy luận ảnh hiện có.
- Kết quả dùng cùng bảng thống kê và danh sách detection như ảnh upload.
- Dừng toàn bộ media track khi đổi tab, đóng camera hoặc component bị hủy.

#### Xử lý lỗi

- Trình duyệt không hỗ trợ camera.
- Người dùng từ chối quyền.
- Máy không có camera hoặc camera đang được ứng dụng khác sử dụng.
- Trang không chạy trong secure context phù hợp; `localhost` và `127.0.0.1` được dùng cho demo cục bộ.

#### Phần mở rộng tùy chọn

Camera live gửi frame liên tục chỉ thực hiện nếu còn thời gian sau khi snapshot, video, QA và báo cáo đã hoàn thành. Không đưa live detection vào tiêu chí bắt buộc.

#### Cổng nghiệm thu

- Mở, chụp và dừng camera không để camera tiếp tục hoạt động nền.
- Ảnh camera chạy được với cả hai mô hình.
- Từ chối quyền không làm hỏng hai tab còn lại.

### Giai đoạn 4 — Kiểm thử và ổn định

#### Unit test

- Kiểm tra metadata và validation tệp video.
- Kiểm tra tính phần trăm tiến độ và trạng thái job.
- Kiểm tra dọn tài nguyên khi decoder/writer lỗi.
- Kiểm tra endpoint video bằng model giả, không cần GPU.
- Kiểm tra camera frontend ở mức logic component phù hợp.

#### Integration test

- Dùng cùng một video ngắn cho hai mô hình.
- Chuyển Faster R-CNN → RetinaNet → Faster R-CNN và theo dõi VRAM.
- Kiểm tra ảnh không có detection sau threshold.
- Kiểm tra ảnh/video hỏng và định dạng không hỗ trợ.
- Kiểm tra backend tắt rồi bật lại; frontend phải tự kết nối lại.
- Chạy toàn bộ `pytest` và `npm run build`.

#### Kiểm tra thủ công

- Chrome/Edge hoặc trình duyệt tích hợp trên màn hình desktop.
- Bố cục responsive ở chiều rộng nhỏ.
- Nhãn và bounding box không bị che/cắt ở mép ảnh.
- Tải kết quả ảnh và video thành công.
- Camera được cấp quyền và bị từ chối quyền.

#### Cổng nghiệm thu

- Không còn lỗi chặn ảnh, video hoặc camera snapshot.
- Không xuất hiện số liệu mẫu giả.
- Các cảnh báo còn lại được ghi rõ và không ảnh hưởng luồng chính.

### Giai đoạn 5 — Tài sản báo cáo và tài liệu bàn giao

#### Ảnh cần chụp

1. Giao diện chính và khu vực chọn mô hình.
2. Faster R-CNN chạy trên một ảnh minh họa.
3. RetinaNet chạy trên cùng ảnh minh họa.
4. Video đang xử lý với progress bar.
5. Video đã hoàn tất và nút tải kết quả.
6. Camera snapshot và kết quả phát hiện.
7. Một trường hợp dự đoán đúng.
8. Một trường hợp bỏ sót hoặc phát hiện nhầm để phân tích hạn chế.

Mỗi ảnh đưa vào báo cáo phải ghi model, threshold và nguồn dữ liệu; không dùng ảnh test để lựa chọn tham số mới.

#### Tài liệu cần cập nhật

- `README.md`: trạng thái, tính năng và lệnh chạy.
- `HUONG_DAN_CAI_DAT.md`: Python, Node.js, FastAPI và cách bật hai server.
- `frontend/README.md`: cấu hình frontend và backend URL.
- Phụ lục báo cáo: cấu trúc thư mục, checkpoint, threshold, phần cứng và hướng dẫn demo.
- Bản Word mục 2.1.2: chỉ bổ sung nội dung ứng dụng/ảnh minh họa thuộc phạm vi Long; không sửa số liệu đã kiểm chứng nếu không có nguồn mới.

#### Cổng nghiệm thu

- Thành viên khác clone repository có thể cài và chạy theo README.
- Ảnh chụp đủ rõ để đưa vào báo cáo và slide.
- Không có placeholder không rõ nghĩa trong phần bàn giao của Long.

### Giai đoạn 6 — Commit, push và đóng mốc

- Chạy toàn bộ test và build lần cuối.
- Kiểm tra không commit tệp lớn hoặc dữ liệu nhạy cảm.
- Commit theo nhóm thay đổi hợp lý: video, camera, QA/tài liệu.
- Push trực tiếp lên `main` theo quy trình một người code đã chốt.
- Kiểm tra repository GitHub sau push.
- Ghi lại commit cuối dùng để nộp và trình bày.

## 6. Lịch thực hiện đề xuất

| Ngày | Công việc | Kết quả cần có |
|---|---|---|
| 27/08 | Khóa mốc demo ảnh, commit/push; xây backend video | Commit ổn định và API video chạy bằng model giả |
| 28/08 | Nối frontend video; kiểm thử hai checkpoint | Video ngắn chạy được với Faster R-CNN và RetinaNet |
| 29/08 | Camera snapshot và xử lý quyền camera | Camera chụp và suy luận được |
| 30/08 | QA toàn hệ thống, sửa lỗi, build | Ảnh/video/camera không còn lỗi chặn |
| 31/08 | Chụp ảnh, cập nhật README và phụ lục | Đủ tài sản cho báo cáo |
| 01–02/09 | Dự phòng, ghép báo cáo và chuẩn bị trình bày | Bản nộp ổn định trước 03/09 |

## 7. Thứ tự ưu tiên khi thiếu thời gian

1. Demo ảnh ổn định và có commit GitHub.
2. Video upload → progress → tải kết quả.
3. Camera snapshot.
4. QA và ảnh chụp báo cáo.
5. Camera live là tùy chọn cuối cùng.

Không cắt giảm kiểm tra hash checkpoint, threshold validation, giải phóng VRAM hoặc tài liệu chạy vì đây là các yếu tố ảnh hưởng trực tiếp đến tính đúng đắn và khả năng trình bày của BTL.

## 8. Tiêu chí hoàn thành cuối

- [x] Mốc demo ảnh đã commit và push (`67bed52`).
- [x] Video chạy được với cả hai mô hình (smoke test 2 frame đã hoàn tất với Faster R-CNN và RetinaNet trên RTX 2050).
- [x] Video có progress, trạng thái lỗi và nút tải kết quả.
- [ ] Camera snapshot chạy được với cả hai mô hình (đã hoàn thiện mã; cần xác nhận quyền camera trên trình duyệt của Long).
- [ ] Ảnh, video và camera dùng chung class mapping và threshold config.
- [ ] Chỉ một model nằm trên GPU tại một thời điểm.
- [ ] Toàn bộ test pass; skip còn lại có lý do rõ ràng.
- [ ] Frontend production build thành công.
- [ ] Backend tắt/bật lại không bắt buộc khởi động lại frontend.
- [ ] Có đủ ảnh minh họa và ví dụ hạn chế cho báo cáo.
- [ ] README và phụ lục cho phép thành viên khác chạy lại.
- [ ] Không commit dataset, checkpoint, `.venv`, `node_modules` hoặc artifact lớn.
- [ ] Commit cuối đã push lên `main` và được kiểm tra trên GitHub.

## 9. Các điểm Long cần duyệt

1. Video dùng job nền cục bộ và frontend polling tiến độ khoảng một giây.
2. Giới hạn đề xuất: tối đa `200 MB`, khuyến nghị video không quá `5 phút`.
3. Camera snapshot là yêu cầu bắt buộc; camera live chỉ làm nếu còn thời gian.
4. Hoàn thành và nghiệm thu từng cổng trước khi chuyển sang giai đoạn tiếp theo.
5. Commit/push mốc demo ảnh trước khi bắt đầu sửa phần video.

Sau khi Long duyệt, công việc đầu tiên là **Giai đoạn 0 — kiểm tra, commit và push mốc demo ảnh**, sau đó triển khai backend video.
