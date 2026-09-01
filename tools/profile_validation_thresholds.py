"""Profile confidence thresholds on an explicitly supplied, non-test COCO split.

The tool runs inference once, then computes precision/recall/F1 for a grid of
thresholds from the cached CPU predictions.  It is intended for an external
validation set such as ``vn_validation``; it refuses a path whose file name
contains ``test`` so that data collection decisions cannot be tuned on test.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluate import class_names_from_config
from src.threshold_selection import collect_validation_predictions, metrics_at_threshold, parse_thresholds
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile threshold on an explicit non-test COCO validation split"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", type=parse_thresholds, default=tuple(round(v * 0.05, 2) for v in range(1, 20)))
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def profile(
    config_path: str,
    annotations_path: str,
    checkpoint_path: str,
    output_path: str,
    *,
    thresholds: tuple[float, ...],
    iou_threshold: float,
    device: str,
    batch_size: int,
    num_workers: int | None,
) -> dict[str, Any]:
    annotations = resolve_project_path(annotations_path)
    if "test" in annotations.name.lower():
        raise ValueError("Không được profile threshold trên test split")
    if not annotations.is_file():
        raise FileNotFoundError(f"Không tìm thấy annotation: {annotations}")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold phải nằm trong [0, 1]")

    config = copy.deepcopy(load_config(config_path))
    config["data"]["val_annotations"] = annotations.as_posix()
    predictions, targets, checkpoint, context = collect_validation_predictions(
        config,
        checkpoint_path,
        device_name=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_names = class_names_from_config(config)
    grid = []
    for threshold in thresholds:
        grid.append(
            {
                "threshold": threshold,
                "metrics": metrics_at_threshold(
                    predictions,
                    targets,
                    class_names,
                    confidence_threshold=threshold,
                    iou_threshold=iou_threshold,
                ),
            }
        )
    result = {
        "schema_version": "external-validation-threshold-profile-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "data-collection diagnosis only; not a demo threshold update",
        "annotation_path": annotations.as_posix(),
        "checkpoint": checkpoint,
        "context": context,
        "iou_threshold": iou_threshold,
        "threshold_grid": grid,
    }
    output = resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    result = profile(
        args.config,
        args.annotations,
        args.checkpoint,
        args.output,
        thresholds=args.thresholds,
        iou_threshold=args.iou_threshold,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Đã profile {len(result['threshold_grid'])} threshold trên {result['context']['protocol']['images_evaluated']} ảnh")


if __name__ == "__main__":
    main()
