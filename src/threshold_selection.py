"""Chọn confidence threshold cho demo chỉ bằng tập validation.

Script chạy suy luận đúng một lần trên validation, sau đó quét các ngưỡng trên
prediction đã giữ ở CPU. Tập test không nằm trong API hay giao thức này.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .dataset import CocoBoxDataset, collate_fn
from .evaluate import (
    build_evaluation_model,
    choose_device,
    default_checkpoint,
    file_sha256,
    load_checkpoint_into_model,
    project_relative,
)
from .metrics import precision_recall
from .transforms import build_transforms
from .utils import load_config, resolve_project_path, set_seed


THRESHOLD_SELECTION_SCHEMA_VERSION = "demo-threshold-selection-1.0"
DEFAULT_THRESHOLDS = tuple(round(value * 0.05, 2) for value in range(1, 20))


def _configure_console_encoding() -> None:
    """Giữ CLI dùng được trên PowerShell Windows có code page cũ."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def parse_thresholds(value: str) -> tuple[float, ...]:
    """Đọc danh sách threshold, loại trùng và sắp theo thứ tự tăng dần."""
    try:
        thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("Threshold phải là các số ngăn cách bằng dấu phẩy") from error
    if not thresholds or any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise argparse.ArgumentTypeError("Threshold phải nằm trong [0, 1]")
    return tuple(thresholds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chọn confidence threshold cho demo bằng validation, không dùng test"
    )
    parser.add_argument("--config", required=True, help="Cấu hình Faster R-CNN hoặc RetinaNet")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint best_map.pth")
    parser.add_argument("--output", required=True, help="JSON kết quả quét threshold")
    parser.add_argument(
        "--demo-config",
        default="configs/demo_thresholds.yaml",
        help="YAML nhận threshold đã chọn để inference/demo dùng lại",
    )
    parser.add_argument("--target-class", default="NoHelmet", help="Lớp ưu tiên khi chọn threshold")
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Ví dụ: 0.05,0.10,...,0.95",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def _cpu_prediction(prediction: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "boxes": prediction["boxes"].detach().cpu(),
        "labels": prediction["labels"].detach().cpu(),
        "scores": prediction["scores"].detach().cpu(),
    }


def _cpu_target(target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu(),
    }


def collect_validation_predictions(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
    batch_size: int = 1,
    num_workers: int | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], dict[str, Any], dict[str, Any]]:
    """Suy luận một lần trên validation; không nhận tham số split để tránh dùng test."""
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn hoặc bằng 1")
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = choose_device(device_name)
    annotation_path = resolve_project_path(config["data"]["val_annotations"])
    image_root = resolve_project_path(config["data"]["image_root"])
    checkpoint_file = resolve_project_path(checkpoint_path)
    for path, label in (
        (annotation_path, "validation annotations"),
        (image_root, "thư mục ảnh"),
        (checkpoint_file, "checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy {label}: {path}")
    if num_workers is None:
        num_workers = int(config.get("runtime", {}).get("num_workers", 0))

    dataset = CocoBoxDataset(image_root, annotation_path, transforms=build_transforms(train=False))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = build_evaluation_model(config, device)
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint_file, device)
    model.eval()

    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for images, batch_targets in data_loader:
            device_images = [image.to(device, non_blocking=True) for image in images]
            batch_predictions = model(device_images)
            predictions.extend(_cpu_prediction(prediction) for prediction in batch_predictions)
            targets.extend(_cpu_target(target) for target in batch_targets)

    protocol = {
        "data_split": "val",
        "annotation_path": project_relative(annotation_path),
        "annotation_sha256": file_sha256(annotation_path),
        "images_evaluated": len(predictions),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }
    runtime = {
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    return predictions, targets, checkpoint_metadata, {"protocol": protocol, "runtime": runtime}


def metrics_at_threshold(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    class_names: Mapping[int, str],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    """Tính TP/FP/FN, Precision/Recall/F1 theo threshold đã cho."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions và targets phải có cùng số ảnh")
    if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("Confidence threshold và IoU threshold phải nằm trong [0, 1]")

    counts = {int(class_id): {"tp": 0, "fp": 0, "fn": 0} for class_id in class_names if int(class_id) != 0}
    for prediction, target in zip(predictions, targets):
        prediction_boxes = prediction["boxes"]
        prediction_labels = prediction["labels"]
        prediction_scores = prediction["scores"]
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        for class_id in counts:
            ground_truth_indices = torch.where(target_labels == class_id)[0]
            predicted_indices = torch.where(
                (prediction_labels == class_id) & (prediction_scores >= confidence_threshold)
            )[0]
            predicted_indices = predicted_indices[
                torch.argsort(prediction_scores[predicted_indices], descending=True)
            ]
            matched_ground_truth: set[int] = set()
            for predicted_index in predicted_indices.tolist():
                available_indices = [
                    index.item()
                    for index in ground_truth_indices
                    if index.item() not in matched_ground_truth
                ]
                if not available_indices:
                    counts[class_id]["fp"] += 1
                    continue
                overlaps = box_iou(
                    prediction_boxes[predicted_index].unsqueeze(0),
                    target_boxes[available_indices],
                ).squeeze(0)
                best_overlap, best_position = overlaps.max(dim=0)
                if best_overlap.item() >= iou_threshold:
                    matched_ground_truth.add(available_indices[best_position.item()])
                    counts[class_id]["tp"] += 1
                else:
                    counts[class_id]["fp"] += 1
            counts[class_id]["fn"] += len(ground_truth_indices) - len(matched_ground_truth)

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, class_name in class_names.items():
        if int(class_id) == 0:
            continue
        class_count = counts[int(class_id)]
        precision, recall = precision_recall(
            tp=class_count["tp"], fp=class_count["fp"], fn=class_count["fn"]
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(class_name)] = {**class_count, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {"per_class": per_class, "macro_f1": sum(f1_values) / len(f1_values)}


def select_best_threshold(candidates: Sequence[Mapping[str, Any]], target_class: str) -> Mapping[str, Any]:
    """Chọn threshold của một lớp theo F1, Recall, Precision và confidence."""
    if not candidates:
        raise ValueError("Cần ít nhất một threshold ứng viên")
    if any(target_class not in candidate["per_class"] for candidate in candidates):
        raise ValueError(f"Không có lớp target_class={target_class} trong kết quả")
    return max(
        candidates,
        key=lambda candidate: (
            candidate["per_class"][target_class]["f1"],
            candidate["per_class"][target_class]["recall"],
            candidate["per_class"][target_class]["precision"],
            candidate["confidence_threshold"],
        ),
    )


def select_thresholds_per_class(
    candidates: Sequence[Mapping[str, Any]], class_names: Mapping[int, str]
) -> dict[str, Mapping[str, Any]]:
    """Chọn độc lập threshold tối ưu trên validation cho từng lớp detector."""
    return {
        str(class_name): select_best_threshold(candidates, str(class_name))
        for class_id, class_name in class_names.items()
        if int(class_id) != 0
    }


def update_demo_config(path: Path, result: Mapping[str, Any]) -> None:
    """Ghi threshold theo từng model để demo nạp lại, không làm thay config đánh giá."""
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = yaml.safe_load(stream) or {}
        if not isinstance(existing, dict):
            raise ValueError("demo_config phải là YAML mapping")
    models = existing.setdefault("models", {})
    model_name = result["model"]["name"]
    selected_thresholds = result["selected_thresholds"]
    models[model_name] = {
        "confidence_thresholds": {
            class_name: selection["confidence_threshold"]
            for class_name, selection in selected_thresholds.items()
        },
        "selection_split": "val",
        "selection_target_class": "per_class",
        "selection_metric": "f1",
        "selection_iou_threshold": result["selection_policy"]["iou_threshold"],
        "checkpoint_sha256": result["checkpoint"]["sha256"],
    }
    existing["schema_version"] = "demo-thresholds-2.0"
    existing["note"] = (
        "Threshold riêng từng lớp được chọn trên validation; không dùng kết quả test."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_selection(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    target_class: str = "NoHelmet",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    iou_threshold: float = 0.50,
    device_name: str = "auto",
    batch_size: int = 1,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Chạy quét threshold trên validation và ghi JSON tái lập được."""
    class_names = {int(class_id): str(name) for class_id, name in config["classes"].items() if int(class_id) != 0}
    if target_class not in class_names.values():
        raise ValueError(f"target_class={target_class} không có trong config classes={class_names}")
    predictions, targets, checkpoint_metadata, context = collect_validation_predictions(
        config,
        checkpoint_path,
        device_name=device_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    candidates = []
    for threshold in thresholds:
        candidate = metrics_at_threshold(
            predictions,
            targets,
            class_names,
            confidence_threshold=float(threshold),
            iou_threshold=iou_threshold,
        )
        candidate["confidence_threshold"] = float(threshold)
        candidates.append(candidate)
    selected_thresholds = select_thresholds_per_class(candidates, class_names)
    selected = dict(selected_thresholds[target_class])
    result = {
        "schema_version": THRESHOLD_SELECTION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model"]["name"],
            "num_classes": int(config["model"]["num_classes"]),
        },
        "checkpoint": checkpoint_metadata,
        "selection_policy": {
            "data_split": "val",
            "target_class": target_class,
            "selection_metric": "f1",
            "iou_threshold": iou_threshold,
            "tie_breakers": ["recall", "precision", "higher_confidence_threshold"],
            "test_data_used": False,
        },
        **context,
        "candidates": candidates,
        "selected_threshold": selected,
        "selected_thresholds": selected_thresholds,
    }
    result_file = resolve_project_path(output_path)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    _configure_console_encoding()
    args = parse_args()
    config = load_config(args.config)
    result = run_selection(
        config,
        args.checkpoint or default_checkpoint(config),
        args.output,
        target_class=args.target_class,
        thresholds=args.thresholds,
        iou_threshold=args.iou_threshold,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    demo_config = resolve_project_path(args.demo_config)
    update_demo_config(demo_config, result)
    selected_thresholds = result["selected_thresholds"]
    threshold_text = ", ".join(
        f"{class_name}={selection['confidence_threshold']:.2f}"
        for class_name, selection in selected_thresholds.items()
    )
    target_metrics = result["selected_threshold"]["per_class"][args.target_class]
    print(
        f"Đã chọn threshold theo lớp cho {result['model']['name']} trên validation: "
        f"{threshold_text}. {args.target_class} F1={target_metrics['f1']:.4f}, "
        f"Precision={target_metrics['precision']:.4f}, Recall={target_metrics['recall']:.4f}."
    )


if __name__ == "__main__":
    main()
