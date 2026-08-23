"""Điểm vào cho pipeline fine-tune.

Vòng lặp huấn luyện sẽ được triển khai sau khi dataset và môi trường CUDA được
xác nhận. Tệp hiện tại chỉ kiểm tra và hiển thị cấu hình đã chọn.
"""

from __future__ import annotations

import argparse

from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune object detector")
    parser.add_argument("--config", required=True, help="Đường dẫn tới cấu hình mô hình")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy kiểm thử pipeline ngắn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = config["model"]["name"]
    print(f"Đã đọc cấu hình cho: {model_name}")
    print("Pipeline huấn luyện chưa được triển khai; cần xác nhận dataset và CUDA trước.")


if __name__ == "__main__":
    main()
