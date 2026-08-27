"""Đánh giá checkpoint Faster R-CNN hoặc RetinaNet trên split COCO cố định.

Kết quả được ghi theo schema JSON dùng chung để ``compare_models`` chỉ so
sánh các lần chạy có cùng split và giao thức metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision
from torch.utils.data import DataLoader

from .dataset import CocoBoxDataset, collate_fn
from .metrics import DetectionEvaluator
from .models import build_model
from .transforms import build_transforms
from .utils import PROJECT_ROOT, load_config, resolve_project_path, set_seed


EVALUATION_PROTOCOL_VERSION = "1.0"


def file_sha256(path: Path) -> str:
    """Tính SHA-256 theo luồng để metadata không phụ thuộc kích thước tệp."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    """Đưa đường dẫn về dạng ổn định tương đối với root dự án nếu có thể."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def choose_device(requested: str) -> torch.device:
    """Chọn CPU/CUDA; không âm thầm đổi từ CUDA sang CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng PyTorch không nhận diện được GPU CUDA")
    return torch.device(requested)


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Lấy state dict từ hai định dạng checkpoint thông dụng của PyTorch."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint phải là mapping hoặc state_dict PyTorch")

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and all(
            isinstance(name, str) for name in candidate
        ):
            return candidate

    if checkpoint and all(isinstance(name, str) for name in checkpoint):
        return checkpoint
    raise ValueError(
        "Không tìm thấy state dict. Hỗ trợ key model_state_dict, state_dict, model "
        "hoặc checkpoint chính là state_dict."
    )


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    allow_partial_load: bool = False,
) -> dict[str, Any]:
    """Nạp checkpoint và trả thông tin kiểm chứng để lưu vào kết quả."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Tương thích PyTorch cũ chưa có weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    incompatible = model.load_state_dict(
        checkpoint_state_dict(checkpoint), strict=not allow_partial_load
    )
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if (missing or unexpected) and not allow_partial_load:
        raise RuntimeError("Checkpoint không khớp kiến trúc mô hình")

    metadata: dict[str, Any] = {
        "path": project_relative(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "partial_load": allow_partial_load,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    if isinstance(checkpoint, Mapping):
        for key in ("epoch", "best_metric", "config", "training_config"):
            if key in checkpoint and isinstance(checkpoint[key], (str, int, float, bool, type(None))):
                metadata[key] = checkpoint[key]
    return metadata


def build_evaluation_model(config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    """Khởi tạo kiến trúc đúng config trước khi nạp checkpoint.

    Checkpoint sẽ ghi đè toàn bộ tham số, vì vậy không tải pretrained weights
    lần nữa khi chỉ cần tạo kiến trúc để đánh giá hoặc nạp checkpoint smoke.
    """
    model_config = config["model"]
    image_config = config["image"]
    model = build_model(
        name=model_config["name"],
        num_classes=int(model_config["num_classes"]),
        min_size=int(image_config["min_size"]),
        max_size=int(image_config["max_size"]),
        trainable_backbone_layers=int(model_config.get("trainable_backbone_layers", 3)),
        weights="NONE",
    )
    return model.to(device)


def class_names_from_config(config: Mapping[str, Any]) -> dict[int, str]:
    """Lấy các nhãn detection, loại background id 0."""
    classes = config.get("classes", {})
    result = {int(class_id): str(name) for class_id, name in classes.items() if int(class_id) != 0}
    if not result:
        raise ValueError("Cấu hình classes phải có ít nhất một lớp khác background")
    return result


def default_checkpoint(config: Mapping[str, Any]) -> str:
    """Lấy checkpoint tốt nhất được quy ước trong config mô hình."""
    checkpoint = config.get("output", {}).get("best_checkpoint")
    if not checkpoint:
        raise ValueError("Thiếu output.best_checkpoint; hãy truyền --checkpoint")
    return str(checkpoint)


def default_output(config: Mapping[str, Any], split: str) -> str:
    """Vị trí JSON chuẩn, được .gitignore bỏ qua như artifact thực nghiệm."""
    directory = config.get("output", {}).get("directory")
    if not directory:
        raise ValueError("Thiếu output.directory; hãy truyền --output")
    return str(Path(directory) / "metrics" / f"{split}_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate object detector on a fixed COCO split")
    parser.add_argument("--config", required=True, help="Cấu hình Faster R-CNN hoặc RetinaNet")
    parser.add_argument("--split", choices=("val", "test", "challenge"), default="val")
    parser.add_argument(
        "--annotations",
        default=None,
        help="COCO annotation tùy chọn, bắt buộc khi --split challenge",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint cần đánh giá")
    parser.add_argument("--output", default=None, help="Tệp JSON kết quả")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Chỉ áp dụng cho Precision/Recall; mAP luôn dùng toàn bộ prediction",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Chỉ dùng để chẩn đoán checkpoint; metadata sẽ ghi rõ missing/unexpected keys.",
    )
    return parser.parse_args()


def evaluate(
    config: Mapping[str, Any],
    split: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    device_name: str = "auto",
    batch_size: int = 1,
    num_workers: int | None = None,
    allow_partial_load: bool = False,
    annotations_override: str | Path | None = None,
    confidence_threshold_override: float | None = None,
) -> dict[str, Any]:
    """Đánh giá toàn bộ một split và ghi JSON tái lập được.

    mAP dùng toàn bộ prediction (chuẩn COCO). Precision/Recall theo lớp dùng
    ngưỡng IoU và confidence nêu trong ``evaluation`` của config.
    """
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn hoặc bằng 1")
    if split not in {"val", "test", "challenge"}:
        raise ValueError("split chỉ được là val, test hoặc challenge")
    if split == "challenge" and annotations_override is None:
        raise ValueError("split challenge yêu cầu annotations_override")
    if confidence_threshold_override is not None and not 0.0 <= confidence_threshold_override <= 1.0:
        raise ValueError("confidence_threshold_override phải nằm trong [0, 1]")

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = choose_device(device_name)
    annotation_path = resolve_project_path(
        annotations_override if annotations_override is not None else config["data"][f"{split}_annotations"]
    )
    image_root = resolve_project_path(config["data"]["image_root"])
    checkpoint_file = resolve_project_path(checkpoint_path)
    result_file = resolve_project_path(output_path)

    for required_path, label in ((annotation_path, "annotation"), (image_root, "thư mục ảnh"), (checkpoint_file, "checkpoint")):
        if not required_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {label}: {required_path}")

    dataset = CocoBoxDataset(image_root, annotation_path, transforms=build_transforms(train=False))
    configured_classes = class_names_from_config(config)
    if dataset.categories and dataset.categories != configured_classes:
        raise ValueError(
            "Category mapping trong annotation không khớp config classes. "
            f"annotation={dataset.categories}; config={configured_classes}"
        )
    if num_workers is None:
        num_workers = int(config.get("runtime", {}).get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers không được âm")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = build_evaluation_model(config, device)
    checkpoint_metadata = load_checkpoint_into_model(
        model, checkpoint_file, device, allow_partial_load=allow_partial_load
    )
    model.eval()
    evaluation_config = config["evaluation"]
    confidence_threshold = (
        confidence_threshold_override
        if confidence_threshold_override is not None
        else evaluation_config.get("confidence_threshold")
    )
    evaluator = DetectionEvaluator(
        configured_classes,
        iou_threshold=float(evaluation_config["precision_recall_iou_threshold"]),
        confidence_threshold=confidence_threshold,
    )

    with torch.inference_mode():
        for images, targets in loader:
            device_images = [image.to(device, non_blocking=True) for image in images]
            predictions = model(device_images)
            evaluator.update(predictions, targets)

    metrics = evaluator.compute()
    result = {
        "schema_version": "evaluation-result-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model"]["name"],
            "num_classes": int(config["model"]["num_classes"]),
            "min_size": int(config["image"]["min_size"]),
            "max_size": int(config["image"]["max_size"]),
        },
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "split": split,
            "annotation_path": project_relative(annotation_path),
            "annotation_sha256": file_sha256(annotation_path),
            "class_names": {str(key): value for key, value in configured_classes.items()},
            "map": "COCO mAP@[IoU=0.50:0.95, step=0.05] via TorchMetrics/pycocotools",
            "map_backend": "pycocotools",
            "precision_recall": {
                "matching": "greedy one-to-one, score descending, same class",
                "iou_threshold": float(evaluation_config["precision_recall_iou_threshold"]),
                "confidence_threshold": confidence_threshold,
            },
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": platform.python_version(),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
        },
        "checkpoint": checkpoint_metadata,
        "metrics": metrics,
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    args = parse_args()
    if args.split == "challenge" and args.annotations is None:
        raise ValueError("--annotations là bắt buộc khi --split challenge")
    config = load_config(args.config)
    result = evaluate(
        config=config,
        split=args.split,
        checkpoint_path=args.checkpoint or default_checkpoint(config),
        output_path=args.output or default_output(config, args.split),
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_partial_load=args.allow_partial_load,
        annotations_override=args.annotations,
        confidence_threshold_override=args.confidence_threshold,
    )
    metrics = result["metrics"]
    print(
        f"Đã đánh giá {result['model']['name']}: "
        f"mAP@0.5:0.95={metrics['map_50_95']}, mAP@0.5={metrics['map_50']}"
    )


if __name__ == "__main__":
    main()
