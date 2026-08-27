# Frontend Helmet Detection AI

Giao diện React + TypeScript cho demo Faster R-CNN và RetinaNet.

## Trạng thái

- Chế độ ảnh đã kết nối FastAPI và chạy checkpoint Faster R-CNN/RetinaNet thật.
- Chế độ video và camera đã có giao diện, chưa kết nối backend.
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
