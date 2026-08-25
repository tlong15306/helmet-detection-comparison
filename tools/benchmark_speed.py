"""Đo latency/FPS detector theo giao thức được lưu đầy đủ trong JSON.

Mặc định benchmark dùng ảnh từ split COCO đã cố định. Thời gian đo chỉ gồm
``model([tensor])`` (forward và hậu xử lý của Torchvision); đọc ảnh, chuyển
đổi PIL sang tensor, copy dữ liệu host-to-device và khởi tạo model bị loại ra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from src.dataset import CocoBoxDataset
from src.evaluate import (
    build_evaluation_model,
    choose_device,
    default_checkpoint,
    file_sha256,
    load_checkpoint_into_model,
    project_relative,
)
from src.transforms import build_transforms
from src.utils import load_config, resolve_project_path, set_seed


def synchronize(device: torch.device) -> None:
    """Đồng bộ CUDA để số đo wall-clock bao gồm hoàn thành kernel GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    """Nội suy percentile đơn giản, không phụ thuộc NumPy."""
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def input_manifest_sha256(names: list[str]) -> str:
    """Fingerprint thứ tự ảnh đầu vào để hai lần benchmark đối chiếu được."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def load_inputs(
    config: Mapping[str, Any],
    split: str,
    max_images: int,
    synthetic: bool,
    synthetic_height: int | None,
    synthetic_width: int | None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Nạp sẵn tensor đầu vào; việc nạp này không nằm trong latency đo."""
    if max_images < 1:
        raise ValueError("max_images phải lớn hơn hoặc bằng 1")
    if synthetic:
        height = synthetic_height or int(config["image"]["min_size"])
        width = synthetic_width or int(config["image"]["min_size"])
        if height < 1 or width < 1:
            raise ValueError("Kích thước synthetic phải lớn hơn 0")
        return [torch.zeros((3, height, width), dtype=torch.float32)], {
            "type": "synthetic_zero_tensor",
            "image_count": 1,
            "image_sizes_chw": [[3, height, width]],
            "warning": "Synthetic input chỉ phù hợp kiểm tra kỹ thuật, không đại diện dữ liệu giao thông.",
        }

    annotation_path = resolve_project_path(config["data"][f"{split}_annotations"])
    image_root = resolve_project_path(config["data"]["image_root"])
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy annotation của split {split}: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_root}")
    dataset = CocoBoxDataset(image_root, annotation_path, transforms=build_transforms(train=False))
    if not dataset:
        raise ValueError(f"Split {split} không có ảnh để benchmark")
    indices = range(min(max_images, len(dataset)))
    inputs = [dataset[index][0] for index in indices]
    names = [str(dataset.images[index]["file_name"]) for index in indices]
    return inputs, {
        "type": "fixed_coco_split",
        "split": split,
        "annotation_path": project_relative(annotation_path),
        "annotation_sha256": file_sha256(annotation_path),
        "image_count": len(inputs),
        "image_names_sha256": input_manifest_sha256(names),
        "image_sizes_chw": [list(image.shape) for image in inputs],
    }


def benchmark(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
    device_name: str = "auto",
    warmup: int = 20,
    runs: int = 100,
    max_images: int = 100,
    synthetic: bool = False,
    synthetic_height: int | None = None,
    synthetic_width: int | None = None,
) -> dict[str, Any]:
    """Trả kết quả latency/FPS, không ghi file để dễ dùng trong test."""
    if warmup < 0:
        raise ValueError("warmup không được âm")
    if runs < 1:
        raise ValueError("runs phải lớn hơn 0")
    if split not in {"val", "test"}:
        raise ValueError("split chỉ được là val hoặc test")

    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = choose_device(device_name)
    checkpoint_file = resolve_project_path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_file}")
    inputs, input_metadata = load_inputs(
        config, split, max_images, synthetic, synthetic_height, synthetic_width
    )
    model = build_evaluation_model(config, device)
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint_file, device)
    model.eval()
    device_inputs = [image.to(device) for image in inputs]

    with torch.inference_mode():
        for index in range(warmup):
            model([device_inputs[index % len(device_inputs)]])
        synchronize(device)

        latencies_ms: list[float] = []
        for index in range(runs):
            image = device_inputs[index % len(device_inputs)]
            synchronize(device)
            started = time.perf_counter()
            model([image])
            synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    mean_latency_ms = statistics.fmean(latencies_ms)
    return {
        "schema_version": "speed-benchmark-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model"]["name"],
            "num_classes": int(config["model"]["num_classes"]),
            "min_size": int(config["image"]["min_size"]),
            "max_size": int(config["image"]["max_size"]),
        },
        "checkpoint": checkpoint_metadata,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "protocol": {
            "batch_size": 1,
            "warmup_runs": warmup,
            "measured_runs": runs,
            "timing": "wall-clock synchronized around model([tensor])",
            "included": "model forward and Torchvision detection post-processing",
            "excluded": "disk I/O, image decode, PIL-to-tensor transform, host-to-device copy, model/checkpoint initialization",
            "input": input_metadata,
        },
        "metrics": {
            "mean_latency_ms": mean_latency_ms,
            "std_latency_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            "p50_latency_ms": percentile(latencies_ms, 0.50),
            "p95_latency_ms": percentile(latencies_ms, 0.95),
            "fps_from_mean_latency": 1000.0 / mean_latency_ms if mean_latency_ms > 0 else None,
        },
    }


def default_output(config: Mapping[str, Any], split: str) -> str:
    directory = config.get("output", {}).get("directory")
    if not directory:
        raise ValueError("Thiếu output.directory; hãy truyền --output")
    return str(Path(directory) / "benchmark" / f"{split}_speed.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector latency/FPS with recorded protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-height", type=int, default=None)
    parser.add_argument("--synthetic-width", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = benchmark(
        config=config,
        checkpoint_path=args.checkpoint or default_checkpoint(config),
        split=args.split,
        device_name=args.device,
        warmup=args.warmup,
        runs=args.runs,
        max_images=args.max_images,
        synthetic=args.synthetic,
        synthetic_height=args.synthetic_height,
        synthetic_width=args.synthetic_width,
    )
    output_path = resolve_project_path(args.output or default_output(config, args.split))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    metrics = result["metrics"]
    print(
        f"Đã benchmark {result['model']['name']}: "
        f"latency={metrics['mean_latency_ms']:.2f} ms, FPS={metrics['fps_from_mean_latency']:.2f}"
    )


if __name__ == "__main__":
    main()
