"""Khung công cụ vẽ bounding box ground truth lên ảnh mẫu."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize COCO annotations")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", default="data/samples/annotated_examples")
    parser.add_argument("--count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("Công cụ trực quan hóa chưa được triển khai; cần xác nhận dataset trước.")


if __name__ == "__main__":
    main()
