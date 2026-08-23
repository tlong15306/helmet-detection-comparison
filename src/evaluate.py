"""Điểm vào đánh giá checkpoint trên validation hoặc test."""

from __future__ import annotations

import argparse

from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate object detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Đã đọc cấu hình: {config['model']['name']}; split={args.split}")
    print("Pipeline đánh giá chưa được triển khai; chưa có checkpoint hoặc dataset.")


if __name__ == "__main__":
    main()
