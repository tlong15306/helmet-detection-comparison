"""Khung benchmark latency/FPS theo cùng giao thức cho hai mô hình."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector speed")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("Benchmark chưa được triển khai; cần checkpoint đã đánh giá.")


if __name__ == "__main__":
    main()
