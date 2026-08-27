# Hướng dẫn cài đặt môi trường cho nhóm

## 1. Phiên bản đã khóa

| Thành phần | Phiên bản |
|---|---|
| Python | 3.11.x, 64-bit |
| PyTorch | 2.5.1 |
| Torchvision | 0.20.1 |
| CUDA wheel cho GPU | cu121 |
| TorchMetrics | 1.7.4 |
| pycocotools | 2.0.8 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.52.4 |
| Node.js | 20.19 trở lên |

Không tự nâng phiên bản trong khi hai mô hình đang được so sánh. Nếu bắt buộc thay đổi, cả nhóm phải cập nhật cùng lúc và ghi lại trong experiment manifest.

## 2. Cài Python

1. Tải Python 3.11 bản 64-bit từ <https://www.python.org/downloads/>.
2. Khi cài, chọn **Add python.exe to PATH**.
3. Mở PowerShell mới và kiểm tra:

```powershell
python --version
```

Kết quả phải bắt đầu bằng `Python 3.11`.

## 3. Clone dự án

```powershell
git clone https://github.com/tlong15306/helmet-detection-comparison.git
cd helmet-detection-comparison
```

Thành viên đã clone trước đó chỉ cần cập nhật nhánh đang làm từ `main` theo quy trình Git của nhóm.

## 4. Chọn chế độ cài theo vai trò

### GPU — Long, Thao và Sinh khi kiểm thử detection trên máy NVIDIA

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_environment.ps1 -InstallMode gpu
```

Chế độ này cài PyTorch/Torchvision CUDA 12.1 và toàn bộ thư viện dự án.

### CPU — Thành khi viết evaluator hoặc thành viên không có NVIDIA GPU

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_environment.ps1 -InstallMode cpu
```

Chế độ này vẫn chạy được unit test, metric và suy luận nhỏ nhưng không dùng để benchmark tốc độ GPU.

### Data — Tùng khi chỉ xử lý dữ liệu

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_environment.ps1 -InstallMode data
```

Chế độ này không tải PyTorch nên nhẹ hơn. Nếu Tùng cần chạy DataLoader có tensor PyTorch, chuyển sang chế độ CPU.

## 5. Kích hoạt môi trường

Mỗi lần mở PowerShell mới tại thư mục dự án:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chạy cho phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Không commit hoặc gửi thư mục `.venv` cho thành viên khác.

## 6. Kiểm tra môi trường

```powershell
python tools/check_environment.py --output outputs/environment.json
```

Với máy GPU, cần kiểm tra:

- `pytorch_available` là `true`;
- `torch` là `2.5.1+cu121` hoặc biến thể tương đương của wheel cu121;
- `torchvision` là `0.20.1+cu121`;
- `cuda_available` là `true`;
- tên GPU và VRAM đúng với máy.

Chạy kiểm thử:

```powershell
python -m pytest
```

## 7. Kaggle

Kaggle Notebook phải ghi lại phiên bản Python, PyTorch, Torchvision, CUDA và tên GPU trước khi train. Nếu môi trường Kaggle khác phiên bản đã khóa, cài lại đúng cặp PyTorch/Torchvision hoặc cập nhật có kiểm soát cho cả hai model; không để Faster R-CNN và RetinaNet dùng hai môi trường khác nhau.

Checkpoint phải được lưu sau mỗi epoch và tải ra khỏi runtime để tránh mất dữ liệu khi phiên bị ngắt.

## 8. Lỗi thường gặp

### Frontend không kết nối được backend

- Xác nhận FastAPI đang chạy tại `http://127.0.0.1:8000`.
- Xác nhận frontend đang chạy tại `http://127.0.0.1:5173`.
- Kiểm tra `http://127.0.0.1:8000/api/health` trả trạng thái `ready`.

### `python` không được nhận diện

Cài lại Python 3.11 và chọn **Add python.exe to PATH**, sau đó mở PowerShell mới.

### `torch.cuda.is_available()` trả về `False`

- Xác minh máy có NVIDIA GPU bằng `nvidia-smi`.
- Xác minh đã dùng `-InstallMode gpu`, không phải `cpu`.
- Kiểm tra đúng `.venv` của dự án đang được kích hoạt.
- Không cài đè wheel CPU sau khi đã cài wheel CUDA.

### CUDA out of memory

Không cài lại CUDA Toolkit. Giảm batch size theo cấu hình chung của nhóm và áp dụng cùng quy tắc cho cả hai mô hình.

### `pycocotools` không cài được

Xác minh đang dùng Python 3.11 64-bit và pip đã được script cập nhật. Không tự chuyển evaluator sang thư viện khác trước khi báo nhóm.
