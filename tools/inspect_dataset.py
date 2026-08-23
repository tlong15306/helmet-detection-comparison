"""Kiểm tra nhanh cấu trúc và phân bố annotation COCO JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect COCO annotations")
    parser.add_argument("--annotations", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.annotations.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)

    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    counts = Counter(int(annotation["category_id"]) for annotation in coco.get("annotations", []))
    invalid_boxes = sum(
        1
        for annotation in coco.get("annotations", [])
        if len(annotation.get("bbox", [])) != 4
        or annotation["bbox"][2] <= 0
        or annotation["bbox"][3] <= 0
    )

    summary = {
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": category_names,
        "annotations_per_category": {
            category_names.get(category_id, str(category_id)): count
            for category_id, count in sorted(counts.items())
        },
        "invalid_boxes": invalid_boxes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
