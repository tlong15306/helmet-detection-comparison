"""Đánh giá TTA/ensemble trên validation mà không sửa checkpoint gốc."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CocoBoxDataset
from src.evaluate import (
    build_evaluation_model,
    checkpoint_state_dict,
    choose_device,
    class_names_from_config,
    default_checkpoint,
    file_sha256,
    project_relative,
)
from src.metrics import DetectionEvaluator
from src.prediction_fusion import fuse_predictions, horizontal_flip_prediction
from src.transforms import build_transforms
from src.threshold_selection import (
    DEFAULT_THRESHOLDS,
    metrics_at_threshold,
    select_thresholds_per_class,
)
from src.utils import load_config, resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector fusion/TTA on validation")
    parser.add_argument(
        "--mode",
        choices=("faster_tta", "retinanet_tta", "ensemble", "ensemble_tta"),
        required=True,
    )
    parser.add_argument("--faster-config", default="configs/faster_rcnn.yaml")
    parser.add_argument("--retina-config", default="configs/retinanet.yaml")
    parser.add_argument("--faster-checkpoint", default=None)
    parser.add_argument("--retina-checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "challenge"), default="val")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_model(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, str]]:
    checkpoint_file = resolve_project_path(checkpoint_path)
    model = build_evaluation_model(config, device)
    payload = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(payload), strict=True)
    model.eval()
    return model, {
        "path": project_relative(checkpoint_file),
        "sha256": file_sha256(checkpoint_file),
    }


def _predict_sources(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    use_tta: bool,
) -> list[dict[str, torch.Tensor]]:
    original = model([image])[0]
    sources = [original]
    if use_tta:
        flipped = model([torch.flip(image, dims=(-1,))])[0]
        sources.append(horizontal_flip_prediction(flipped, int(image.shape[-1])))
    return sources


def evaluate_fusion(args: argparse.Namespace) -> dict[str, Any]:
    faster_config = load_config(args.faster_config)
    retina_config = load_config(args.retina_config)
    if class_names_from_config(faster_config) != class_names_from_config(retina_config):
        raise ValueError("Hai config phải có cùng class mapping")
    if args.split == "challenge" and not args.annotations:
        raise ValueError("--annotations là bắt buộc với challenge")
    set_seed(int(faster_config.get("project", {}).get("seed", 42)))
    device = choose_device(args.device)
    annotation_path = resolve_project_path(
        args.annotations
        if args.annotations
        else faster_config["data"][f"{args.split}_annotations"]
    )
    image_root = resolve_project_path(faster_config["data"]["image_root"])
    dataset = CocoBoxDataset(
        image_root, annotation_path, transforms=build_transforms(train=False)
    )
    class_names = class_names_from_config(faster_config)
    evaluator = DetectionEvaluator(
        class_names,
        iou_threshold=float(
            faster_config["evaluation"]["precision_recall_iou_threshold"]
        ),
        confidence_threshold=faster_config["evaluation"].get("confidence_threshold"),
    )

    use_faster = args.mode in {"faster_tta", "ensemble", "ensemble_tta"}
    use_retina = args.mode in {"retinanet_tta", "ensemble", "ensemble_tta"}
    use_tta = args.mode in {"faster_tta", "retinanet_tta", "ensemble_tta"}
    models: list[tuple[str, torch.nn.Module]] = []
    checkpoints: dict[str, dict[str, str]] = {}
    if use_faster:
        checkpoint = args.faster_checkpoint or default_checkpoint(faster_config)
        model, metadata = _load_model(faster_config, checkpoint, device)
        models.append(("faster_rcnn", model))
        checkpoints["faster_rcnn"] = metadata
    if use_retina:
        checkpoint = args.retina_checkpoint or default_checkpoint(retina_config)
        model, metadata = _load_model(retina_config, checkpoint, device)
        models.append(("retinanet", model))
        checkpoints["retinanet"] = metadata

    validation_predictions: list[dict[str, torch.Tensor]] = []
    validation_targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            device_image = image.to(device, non_blocking=True)
            sources: list[dict[str, torch.Tensor]] = []
            for _name, model in models:
                sources.extend(_predict_sources(model, device_image, use_tta=use_tta))
            fused = fuse_predictions(
                sources,
                iou_threshold=args.iou,
                max_detections=args.max_detections,
            )
            evaluator.update([fused], [target])
            if args.split == "val":
                validation_predictions.append(
                    {key: value.detach().cpu() for key, value in fused.items()}
                )
                validation_targets.append(
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                    }
                )
            if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                print(f"Đã xử lý {index + 1}/{len(dataset)} ảnh", flush=True)

    threshold_selection = None
    if args.split == "val":
        candidates = []
        for threshold in DEFAULT_THRESHOLDS:
            candidate = metrics_at_threshold(
                validation_predictions,
                validation_targets,
                class_names,
                confidence_threshold=threshold,
                iou_threshold=0.5,
            )
            candidate["confidence_threshold"] = threshold
            candidates.append(candidate)
        selected = select_thresholds_per_class(candidates, class_names)
        threshold_selection = {
            "split": "val",
            "metric": "per_class_f1",
            "iou_threshold": 0.5,
            "test_data_used": False,
            "selected_thresholds": {
                class_name: float(selection["confidence_threshold"])
                for class_name, selection in selected.items()
            },
            "selected_metrics": {
                class_name: selection["per_class"][class_name]
                for class_name, selection in selected.items()
            },
            "candidates": candidates,
        }

    result = {
        "schema_version": "fusion-evaluation-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "split": args.split,
        "annotation_path": project_relative(annotation_path),
        "annotation_sha256": file_sha256(annotation_path),
        "fusion": {
            "method": "class_aware_weighted_box_fusion",
            "iou_threshold": args.iou,
            "max_detections": args.max_detections,
            "horizontal_flip_tta": use_tta,
            "sources_per_model": 2 if use_tta else 1,
        },
        "checkpoints": checkpoints,
        "metrics": evaluator.compute(),
        "threshold_selection": threshold_selection,
    }
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_fusion(args)
    print(
        f"{args.mode}: mAP@0.5:0.95={result['metrics']['map_50_95']}, "
        f"mAP@0.5={result['metrics']['map_50']}"
    )


if __name__ == "__main__":
    main()
