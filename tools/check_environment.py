"""Thu thập thông tin môi trường phục vụ tái lập thí nghiệm."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "Pillow",
    "PyYAML",
    "matplotlib",
    "opencv-python",
    "torchmetrics",
    "pycocotools",
    "streamlit",
    "pytest",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra môi trường dự án")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Tệp JSON nhận kết quả; nếu bỏ trống chỉ in ra màn hình",
    )
    return parser.parse_args()


def collect_package_versions() -> dict[str, str | None]:
    """Lấy phiên bản package mà không import toàn bộ thư viện nặng."""
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": collect_package_versions(),
    }

    try:
        import torch
        import torchvision
    except ImportError as error:
        result["pytorch_available"] = False
        result["import_error"] = str(error)
        return result

    result.update(
        {
            "pytorch_available": True,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    )

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["gpu_name"] = properties.name
        result["gpu_vram_gb"] = round(properties.total_memory / 1024**3, 2)
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_compute_capability"] = [properties.major, properties.minor]

    return result


def main() -> None:
    args = parse_args()
    serialized = json.dumps(collect_environment(), ensure_ascii=False, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
