# Quy chuẩn nhãn Challenge Set v1

## Trạng thái

**Chưa khóa.** Trước khi gán nhãn chính thức, phải kiểm tra trực quan annotation EdgeVision hiện có và điền các quyết định dưới đây. Không suy đoán chỉ từ tên lớp.

## Category ID phải dùng

| ID | Lớp | Quyết định về phạm vi box |
|---:|---|---|
| 1 | `BikeWithRider` | `[CẦN XÁC NHẬN TỪ EDGEVISION]` |
| 2 | `NoHelmet` | `[CẦN XÁC NHẬN TỪ EDGEVISION]` |
| 3 | `Helmet` | `[CẦN XÁC NHẬN TỪ EDGEVISION]` |

## Trường hợp phải chốt trước khi gán nhãn

- Người ngồi sau và trẻ em có được gán `Helmet`/`NoHelmet` không?
- Box `Helmet` bao quanh mũ hay vùng đầu có mũ?
- Box `NoHelmet` bao quanh đầu hay toàn người?
- `BikeWithRider` bao quanh phần xe, người lái hay tổ hợp xe-người?
- Đối tượng bị che khuất/cắt mép ảnh cần nhìn thấy tối thiểu bao nhiêu để gán nhãn?
- Khi mũ/nón không chắc chắn, dùng nhãn nào hay đánh dấu mơ hồ?

## Nguyên tắc sau khi policy được khóa

1. Một người gán nhãn tạo annotation; Long duyệt 100% ảnh.
2. Prediction của model chỉ là gợi ý, không là ground truth.
3. Box phải nằm trong ảnh, có diện tích dương và dùng đúng category ID.
4. Trường hợp mơ hồ được ghi vào `annotation_review.csv`, không tự loại âm thầm.
5. Khi annotation đã khóa, mọi sửa đổi tạo version mới và ghi lý do.
