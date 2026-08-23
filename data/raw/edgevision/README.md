# Thư mục `data/raw/edgevision/`

## Dataset dự kiến

EdgeVision Dataset, phiên bản 1, DOI `10.17632/j82bnw7gsr.1`.

## Cấu trúc cần có sau khi tải

```text
edgevision/
├── README.md
├── images/
│   ├── <image_001>.jpg
│   └── ...
└── annotations.json
```

Nếu gói tải về sử dụng tên hoặc cấu trúc khác, chưa đổi tên ngay. Trước tiên phải kiểm tra các đường dẫn `file_name` trong COCO JSON.

## Kiểm tra sau khi tải

1. Xác nhận giấy phép CC BY 4.0 và thông tin trích dẫn.
2. Xác nhận số ảnh, số annotation và ba lớp dự kiến.
3. Kiểm tra category ID có đúng `1, 2, 3` hay không.
4. Kiểm tra mọi `file_name` đều trỏ tới ảnh tồn tại.
5. Phát hiện bounding box âm, rỗng hoặc vượt biên ảnh.
6. Vẽ ít nhất một số mẫu của mỗi lớp để xác minh ý nghĩa nhãn.
7. Kiểm tra ảnh có được lấy liên tiếp từ cùng cảnh hay không trước khi chia tập.

Không thay đổi dữ liệu gốc tại đây. Kết quả chia tập phải đi vào `data/splits/`.
