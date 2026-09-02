# Prediction minh họa fine-tune Việt Nam v6 trên EdgeVision test

Thư mục này chứa prediction của hai checkpoint **fine-tune Việt Nam v6 đang
hiện trên demo** trên cùng ba ảnh thuộc split `test` EdgeVision: image ID 2
(`Image_00002.jpg`), 5 (`Image_00005.jpg`) và 10 (`Image_00010.jpg`).

- `faster_rcnn_v6_*.png`: `outputs/vietnam_pilot_v6_wikimedia/faster_rcnn/stage1/checkpoints/best_map.pth`,
  horizontal-flip TTA và threshold demo theo lớp: BikeWithRider 0,95,
  NoHelmet 0,65, Helmet 0,70.
- `retinanet_v6_*.png`: `outputs/vietnam_pilot_v6_wikimedia/retinanet/stage1/checkpoints/best_map.pth`,
  horizontal-flip TTA và threshold demo theo lớp: BikeWithRider 0,65,
  NoHelmet 0,40, Helmet 0,40.

Đây là ví dụ trực quan, không thay thế `test_metrics.json`. Ảnh nguồn thuộc
EdgeVision v1; xem nguồn dataset trong README gốc của dự án. Các metric v6
được tạo bằng evaluation không dùng threshold demo; ảnh minh họa dùng đúng TTA
và threshold theo lớp của demo.
