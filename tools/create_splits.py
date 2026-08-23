"""Tạo train/validation/test split sau khi quy tắc chia dữ liệu được xác nhận.

Không triển khai chia ngẫu nhiên ngay trong scaffold vì cần kiểm tra dataset có
các ảnh liên tiếp từ cùng cảnh/video hay không. Chia theo từng frame có thể gây
rò rỉ dữ liệu giữa train và test.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed COCO splits")
    parser.add_argument("--config", default="configs/common.yaml")
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("Chưa tạo split: cần kiểm tra cấu trúc nguồn/cảnh của dataset trước.")


if __name__ == "__main__":
    main()
