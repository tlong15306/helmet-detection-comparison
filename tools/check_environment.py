"""In thông tin môi trường phục vụ tái lập thí nghiệm."""

from __future__ import annotations

import json
import platform
import sys


def collect_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
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
        }
    )

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["gpu_name"] = properties.name
        result["gpu_vram_gb"] = round(properties.total_memory / 1024**3, 2)
        result["gpu_count"] = torch.cuda.device_count()

    return result


def main() -> None:
    print(json.dumps(collect_environment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
