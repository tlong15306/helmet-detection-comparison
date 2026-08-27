"""Audit hình học box đầu–xe trong một COCO split, không dùng prediction model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.rider_association import analyze_rider_roles
from src.utils import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit association đầu–xe từ annotation COCO")
    parser.add_argument("--annotations", default="data/splits/val.json")
    parser.add_argument("--output", default="outputs/role_association/edgevision_val_geometry_audit.json")
    return parser.parse_args()


def _record(annotation: dict[str, Any], category_name: str) -> dict[str, Any]:
    x, y, width, height = annotation["bbox"]
    return {
        "detection_id": f"annotation_{annotation['id']}",
        "class_name": category_name,
        "confidence": 1.0,
        "box": [float(x), float(y), float(x + width), float(y + height)],
    }


def audit(annotations_path: str | Path) -> dict[str, Any]:
    path = resolve_project_path(annotations_path)
    with path.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)
    categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    group_statuses: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    per_image: list[dict[str, Any]] = []
    for image in coco["images"]:
        image_id = int(image["id"])
        records = [
            _record(annotation, categories[int(annotation["category_id"])])
            for annotation in annotations_by_image.get(image_id, [])
            if categories[int(annotation["category_id"])] in {"BikeWithRider", "Helmet", "NoHelmet"}
        ]
        result = analyze_rider_roles(records)
        summary = result["summary"]
        totals.update(summary)
        for group in result["rider_groups"]:
            group_statuses[str(group["association_status"])] += 1
        per_image.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name"),
                "summary": summary,
            }
        )

    return {
        "protocol": {
            "source": str(path),
            "uses_ground_truth": True,
            "role_inference": "candidate_only",
            "warning": "Audit này chỉ đánh giá khả năng ghép hình học, không đo độ đúng vai trò tài xế/người ngồi sau.",
        },
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "aggregate": dict(totals),
        "group_association_status": dict(group_statuses),
        "per_image": per_image,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.annotations)
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": result["aggregate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
