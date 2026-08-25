"""Vẽ annotation COCO để kiểm tra trực quan dữ liệu."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize COCO annotations")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count-per-category", type=int, default=10)
    parser.add_argument("--problem-list", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_name(file_name: str, image_id: Any) -> str:
    stem = Path(file_name).stem.replace(" ", "_")
    return f"image_{image_id}_{stem}.jpg"


def select_annotation_ids(coco: dict[str, Any], count_per_category: int, seed: int) -> set[Any]:
    if count_per_category < 0:
        raise ValueError("count_per_category không được âm")
    rng = random.Random(seed)
    by_category: dict[Any, list[Any]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        by_category[annotation.get("category_id")].append(annotation.get("image_id"))
    result: set[Any] = set()
    for image_ids in by_category.values():
        unique_ids = sorted(set(image_ids), key=str)
        rng.shuffle(unique_ids)
        result.update(unique_ids[:count_per_category])
    return result


def render_annotations(coco: dict[str, Any], images_dir: Path, output_dir: Path, image_ids: set[Any]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {item["id"]: str(item.get("name", item["id"])) for item in coco.get("categories", [])}
    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[annotation.get("image_id")].append(annotation)

    rendered = 0
    for image_record in coco.get("images", []):
        image_id = image_record.get("id")
        if image_id not in image_ids:
            continue
        image_path = images_dir / image_record["file_name"]
        if not image_path.is_file():
            continue
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(image)
        for annotation in annotations_by_image.get(image_id, []):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = bbox
            if not all(isinstance(value, (int, float)) for value in bbox):
                continue
            color = COLORS[int(annotation.get("category_id", 0)) % len(COLORS)]
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
            label = categories.get(annotation.get("category_id"), "unknown")
            draw.text((max(0, x), max(0, y - 14)), label, fill=color, stroke_width=1, stroke_fill="black")
        image.save(output_dir / _safe_name(image_record["file_name"], image_id), quality=95)
        rendered += 1
    return rendered


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    image_ids = select_annotation_ids(coco, args.count_per_category, args.seed)
    if args.problem_list is not None:
        problems = json.loads(args.problem_list.read_text(encoding="utf-8"))
        image_ids.update(item.get("image_id") for item in problems if item.get("image_id") is not None)
    rendered = render_annotations(coco, args.images, args.output, image_ids)
    print(json.dumps({"rendered_images": rendered, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
