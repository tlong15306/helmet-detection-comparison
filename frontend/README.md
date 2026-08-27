# Frontend Helmet Detection AI

Giao diện React + TypeScript cho demo Faster R-CNN và RetinaNet.

## Trạng thái

- Chế độ ảnh, video và camera snapshot đã kết nối FastAPI và chạy checkpoint Faster R-CNN/RetinaNet thật.
- Video chạy trong hàng đợi cục bộ, hiển thị tiến độ và tạo tệp MP4 đã gắn nhãn để tải về.
- Camera chỉ gửi frame sau khi người dùng chụp ảnh; không có suy luận camera live liên tục.
- Threshold mặc định được đọc từ cấu hình đã chọn trên validation.

## Chạy frontend

```powershell
npm install
npm run dev
```

Frontend mặc định gọi backend tại `http://127.0.0.1:8000`. Có thể đặt `VITE_API_URL` nếu cần dùng địa chỉ khác.

## Chạy backend

Từ thư mục gốc dự án:

```powershell
.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

## Build kiểm tra

```powershell
npm run build
```

Lần suy luận đầu tiên của mỗi mô hình sẽ chậm hơn do phải kiểm tra hash và nạp checkpoint vào GPU.

## Giới hạn video

- Chấp nhận MP4, MOV, AVI; tối đa 200 MB và 5 phút.
- Video được xử lý tuần tự từng frame, vì vậy không dùng FPS/latency trên giao diện thay cho benchmark trong báo cáo.
- Khi video đang chạy, không gửi thêm tác vụ mới hoặc đổi mô hình để tránh cạnh tranh GPU.
