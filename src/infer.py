"""Giao diện suy luận dự kiến cho ảnh và video."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("Pipeline suy luận chưa được triển khai; cần checkpoint đã đánh giá.")


if __name__ == "__main__":
    main()
