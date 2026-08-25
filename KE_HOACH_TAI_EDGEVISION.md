# Kế hoạch tải và kiểm tra EdgeVision v1

## 1. Trạng thái và phạm vi

- **Người phụ trách thực hiện:** Nguyễn Thành Long, với hỗ trợ kỹ thuật từ Codex.
- **Trạng thái:** Chờ Long duyệt để bắt đầu tải.
- **Phạm vi của lượt này:** tải đúng EdgeVision v1, lưu dữ liệu gốc tại `data/raw/edgevision/`, xác nhận có ảnh và tệp annotation COCO `annotations.json`.
- **Chưa thực hiện trong lượt này:** tạo split train/validation/test, sửa annotation, tiền xử lý dữ liệu hoặc huấn luyện mô hình.
- **Nguồn chính thức:** <https://data.mendeley.com/datasets/j82bnw7gsr/1>
- **DOI:** `10.17632/j82bnw7gsr.1`

## 2. Cấu trúc đích bắt buộc

Sau khi hoàn tất, hai đường dẫn mà mã nguồn sử dụng phải tồn tại:

```text
data/raw/edgevision/
├── images/
│   ├── <ảnh thứ nhất>
│   └── ...
└── annotations.json
```

Nếu gói tải về có nhiều định dạng annotation, chỉ **sao chép hoặc sắp xếp nguyên trạng** bản COCO về cấu trúc trên. Không chuyển đổi annotation và không sửa nội dung JSON trong bước tiếp nhận dữ liệu.

## 3. Nguyên tắc an toàn

1. Dữ liệu tải về, tệp nén, ảnh và annotation không được commit lên GitHub.
2. Trước khi tải, cập nhật `.gitignore` để bỏ qua toàn bộ nội dung cục bộ trong `data/raw/edgevision/`, ngoại trừ tệp hướng dẫn và `.gitkeep`.
3. Không giải nén đè lên dữ liệu đã tồn tại. Nếu thư mục đích có dữ liệu, phải dừng và kiểm tra trước.
4. Kiểm tra danh sách tệp trong gói nén trước khi giải nén nhằm phát hiện đường dẫn bất thường hoặc cấu trúc lồng nhiều cấp.
5. Dữ liệu trong `data/raw/` được coi là bản gốc chỉ đọc. Mọi xử lý về sau phải tạo đầu ra ở vị trí khác.

## 4. Các bước thực hiện

### Bước 1 — Kiểm tra trước khi tải

- Xác nhận đang ở đúng repository `helmet_detection_project` và đúng nhánh làm việc.
- Kiểm tra `data/raw/edgevision/` chưa chứa dataset cũ.
- Kiểm tra dung lượng trống của ổ đĩa.
- Kiểm tra `.gitignore` bằng một tệp giả hoặc lệnh Git để chắc chắn toàn bộ dữ liệu thô sẽ bị bỏ qua.

**Điều kiện qua bước:** đường dẫn đích an toàn, không có dữ liệu cần bảo toàn và quy tắc Git hoạt động.

### Bước 2 — Tải đúng phiên bản từ nguồn chính thức

- Mở bản ghi Mendeley có DOI `10.17632/j82bnw7gsr.1`.
- Ghi lại ngày tải, phiên bản, tên từng tệp cung cấp và giấy phép được công bố.
- Tải đầy đủ các tệp cần thiết cho ảnh và annotation COCO.
- Tính SHA-256 cho từng tệp tải về trước khi giải nén.

**Điều kiện qua bước:** tệp tải hoàn tất, không phải trang HTML lỗi, có dung lượng hợp lý và có checksum để đối chiếu.

### Bước 3 — Kiểm tra và giải nén

- Liệt kê nội dung gói nén trước khi bung tệp.
- Giải nén vào một thư mục tạm nằm trong `data/raw/edgevision/`.
- Xác định chính xác thư mục chứa ảnh và tệp JSON định dạng COCO.
- Đưa chúng về đường dẫn đích `images/` và `annotations.json`.
- Giữ lại tên tệp ảnh đúng như annotation tham chiếu; không đổi tên hàng loạt.

**Điều kiện qua bước:** không có lỗi giải nén; không ghi tệp ra ngoài thư mục đích; cấu trúc đích tồn tại.

### Bước 4 — Kiểm tra nhanh ảnh và `annotations.json`

Thực hiện các kiểm tra tối thiểu sau:

- `annotations.json` tồn tại, mở được bằng UTF-8 và parse được dưới dạng JSON.
- JSON có ba trường COCO chính: `images`, `annotations`, `categories`.
- Thư mục `images/` tồn tại và có ít nhất một tệp ảnh hỗ trợ (`.jpg`, `.jpeg`, `.png` hoặc định dạng thực tế của bộ dữ liệu).
- Mỗi mục ảnh được kiểm tra có `id`, `file_name`, `width`, `height` hợp lệ.
- Mỗi annotation được kiểm tra có `id`, `image_id`, `category_id` và `bbox` gồm bốn giá trị.
- Tên ảnh mà JSON tham chiếu thực sự tồn tại trên đĩa.
- Mở thử một mẫu ảnh để xác nhận tệp không hỏng.
- In thống kê thực tế: số ảnh trong JSON, số ảnh trên đĩa, số bounding box và danh sách lớp.

Không dùng số liệu công bố làm kết quả kiểm tra. Chỉ ghi số liệu đọc trực tiếp từ bản dữ liệu đã tải.

### Bước 5 — Chạy công cụ kiểm định và lưu bằng chứng

- Chạy `tools/inspect_dataset.py` với `data/raw/edgevision/annotations.json`.
- Nếu công cụ hiện tại chưa kiểm tra đủ ảnh, bổ sung chức năng theo phạm vi tối thiểu ở Bước 4 trước khi chạy.
- Lưu báo cáo máy đọc được tại `outputs/dataset_report.json`.
- Lưu metadata tiếp nhận gồm nguồn, DOI, phiên bản, ngày tải, tên tệp và SHA-256 tại `outputs/dataset_intake.json`.
- Chụp hoặc lưu danh sách lỗi nếu có; không âm thầm bỏ qua lỗi.

### Bước 6 — Kiểm tra Git và bàn giao

- Xác nhận `git status` không liệt kê ảnh, archive hoặc `annotations.json` là tệp sẽ commit.
- Báo cáo rõ một trong hai kết quả:
  - **Đạt:** có ảnh, có `annotations.json`, JSON đọc được và ảnh được tham chiếu tồn tại.
  - **Chưa đạt:** nêu chính xác tệp thiếu, lỗi giải nén, lỗi cấu trúc hoặc lỗi annotation.
- Chỉ sau khi đạt mới chuyển sang bước kiểm định sâu và tạo split 70/15/15 với seed 42.

## 5. Đầu ra cần bàn giao

| Đầu ra | Vị trí | Có đưa lên GitHub không? |
|---|---|---|
| Ảnh gốc | `data/raw/edgevision/images/` | Không |
| Annotation COCO gốc | `data/raw/edgevision/annotations.json` | Không |
| Báo cáo kiểm tra | `outputs/dataset_report.json` | Không, trừ khi nhóm thống nhất lưu bản đã loại thông tin lớn |
| Metadata và checksum | `outputs/dataset_intake.json` | Không theo quy tắc hiện tại |
| Mã kiểm tra được bổ sung | `tools/inspect_dataset.py` | Có |
| Hướng dẫn/cập nhật `.gitignore` | `.gitignore` và tài liệu liên quan | Có |

## 6. Tiêu chí hoàn thành

- [ ] Dataset được tải từ đúng bản ghi EdgeVision v1 chính thức.
- [ ] Có checksum SHA-256 của tệp tải về.
- [ ] Có ít nhất một ảnh đọc được trong `data/raw/edgevision/images/`.
- [ ] Có `data/raw/edgevision/annotations.json` parse được.
- [ ] JSON có `images`, `annotations` và `categories`.
- [ ] Đã đối chiếu đường dẫn ảnh trong JSON với tệp trên đĩa.
- [ ] Đã ghi thống kê thực tế, không sao chép số liệu công bố để thay cho kiểm tra.
- [ ] Dataset và tệp nén không bị Git theo dõi.
- [ ] Chưa tạo split, chưa sửa dữ liệu gốc và chưa bắt đầu train.

## 7. Điểm cần Long duyệt trước khi triển khai

1. Cho phép tải bộ dữ liệu từ Mendeley về máy; quá trình có thể tiêu tốn thời gian, băng thông và dung lượng ổ đĩa.
2. Cho phép cập nhật `.gitignore` và, nếu cần, hoàn thiện `tools/inspect_dataset.py` để kiểm tra ảnh cùng các tham chiếu trong COCO JSON.
3. Sau khi Long duyệt, thực hiện tuần tự sáu bước trên và dừng báo lỗi nếu cấu trúc thực tế khác cấu trúc dự kiến.
