"""Đo latency/FPS inference của checkpoint detector theo một giao thức cố định.

Benchmark dùng ảnh validation đã có nhãn nhưng không đọc nhãn trong lúc suy luận.
Thời gian bao gồm chuyển tensor CPU -> GPU, transform/NMS nội bộ của Torchvision và
forward model; không gồm đọc ảnh từ ổ đĩa hoặc render kết quả giao diện.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torchvision

from src.dataset import CocoBoxDataset
from src.evaluate import (
    build_evaluation_model,
    choose_device,
    file_sha256,
    load_checkpoint_into_model,
    project_relative,
)
from src.transforms import build_transforms
from src.utils import load_config, resolve_project_path, set_seed


BENCHMARK_SCHEMA_VERSION = "inference-benchmark-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Torchvision detector inference")
    parser.add_argument("--config", required=True, help="YAML config của model")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint best_map.pth")
    parser.add_argument("--output", required=True, help="JSON artifact cần ghi")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-images", type=int, default=20)
    parser.add_argument("--measure-images", type=int, default=100)
    return parser.parse_args()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Không thể tính percentile cho danh sách rỗng")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def summarize_latencies(latencies_seconds: Sequence[float]) -> dict[str, float]:
    """Trả latency ms và FPS từ từng ảnh đã được synchronize GPU."""
    if not latencies_seconds or any(value <= 0 for value in latencies_seconds):
        raise ValueError("Latency phải có ít nhất một giá trị dương")
    milliseconds = [value * 1000 for value in latencies_seconds]
    mean_ms = sum(milliseconds) / len(milliseconds)
    return {
        "mean_ms": mean_ms,
        "median_ms": _percentile(milliseconds, 50),
        "p95_ms": _percentile(milliseconds, 95),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "fps_from_mean_latency": 1000 / mean_ms,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "auto",
    warmup_images: int = 20,
    measure_images: int = 100,
) -> dict[str, Any]:
    """Chạy warm-up rồi benchmark trên tập con validation cố định."""
    if warmup_images < 0 or measure_images < 1:
        raise ValueError("warmup_images >= 0 và measure_images >= 1 là bắt buộc")

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = choose_device(device_name)
    image_root = resolve_project_path(config["data"]["image_root"])
    validation_annotations = resolve_project_path(config["data"]["val_annotations"])
    checkpoint_file = resolve_project_path(checkpoint_path)
    result_file = resolve_project_path(output_path)
    for path, label in (
        (image_root, "thư mục ảnh"),
        (validation_annotations, "validation annotations"),
        (checkpoint_file, "checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy {label}: {path}")

    dataset = CocoBoxDataset(
        image_root, validation_annotations, transforms=build_transforms(train=False)
    )
    required_images = warmup_images + measure_images
    if len(dataset) < required_images:
        raise ValueError(
            f"Validation chỉ có {len(dataset)} ảnh, cần tối thiểu {required_images} ảnh"
        )

    model = build_evaluation_model(config, device)
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint_file, device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for index in range(warmup_images):
        image, _target = dataset[index]
        _ = model([image.to(device, non_blocking=True)])
    _synchronize(device)

    latencies: list[float] = []
    prediction_count = 0
    for index in range(warmup_images, required_images):
        image, _target = dataset[index]
        _synchronize(device)
        started = time.perf_counter()
        prediction = model([image.to(device, non_blocking=True)])[0]
        _synchronize(device)
        latencies.append(time.perf_counter() - started)
        prediction_count += int(prediction["boxes"].shape[0])

    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model"]["name"],
            "num_classes": int(config["model"]["num_classes"]),
            "min_size": int(config["image"]["min_size"]),
            "max_size": int(config["image"]["max_size"]),
        },
        "checkpoint": checkpoint_metadata,
        "protocol": {
            "data_split": "val",
            "annotation_path": project_relative(validation_annotations),
            "annotation_sha256": file_sha256(validation_annotations),
            "warmup_images": warmup_images,
            "measure_images": measure_images,
            "batch_size": 1,
            "timing_includes": [
                "CPU-to-device tensor transfer",
                "Torchvision detector internal transform",
                "model forward pass",
                "Torchvision post-processing and NMS",
            ],
            "timing_excludes": ["disk image decoding", "visualization and file writing"],
        },
        "runtime": {
            "device": str(device),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "latency": summarize_latencies(latencies),
        "predictions_after_model_threshold": prediction_count,
        "cuda_memory_bytes": (
            {
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
            if device.type == "cuda"
            else None
        ),
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = benchmark(
        load_config(args.config),
        args.checkpoint,
        args.output,
        device_name=args.device,
        warmup_images=args.warmup_images,
        measure_images=args.measure_images,
    )
    latency = result["latency"]
    print(
        f"Đã benchmark {result['model']['name']}: "
        f"mean={latency['mean_ms']:.2f} ms, FPS={latency['fps_from_mean_latency']:.2f}"
    )


if __name__ == "__main__":
    main()
