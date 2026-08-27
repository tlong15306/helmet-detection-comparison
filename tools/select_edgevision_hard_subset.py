"""Tạo tập con ảnh khó từ test EdgeVision bằng tiêu chí không dùng prediction.

Artifact này phục vụ phân tích độ khó nội bộ. Nó không phải external challenge
và không được dùng để điều chỉnh checkpoint hay threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps, ImageStat


HEAD_CATEGORY_IDS = {2, 3}
TAGS = ("crowded", "small_object", "overlapping_boxes", "low_light", "edge_truncation")


def clamp_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = map(float, box)
    return (
        max(0.0, min(x, width)),
        max(0.0, min(y, height)),
        max(0.0, min(x + box_width, width)),
        max(0.0, min(y + box_height, height)),
    )


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def image_brightness(path: Path) -> float:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((256, 256))
        return float(ImageStat.Stat(image).mean[0])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_image(image: dict[str, Any], annotations: list[dict[str, Any]], image_path: Path) -> dict[str, Any]:
    width, height = int(image["width"]), int(image["height"])
    image_area = width * height
    head_boxes = [
        clamp_box(annotation["bbox"], width, height)
        for annotation in annotations
        if int(annotation["category_id"]) in HEAD_CATEGORY_IDS
    ]
    small_heads = sum(
        ((box[2] - box[0]) * (box[3] - box[1])) / image_area <= 0.008
        for box in head_boxes
    )
    head_overlap = any(iou(first, second) >= 0.10 for index, first in enumerate(head_boxes) for second in head_boxes[index + 1 :])
    edge_boxes = sum(
        box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1
        for box in (clamp_box(annotation["bbox"], width, height) for annotation in annotations)
    )
    brightness = image_brightness(image_path)
    tags: list[str] = []
    if len(annotations) >= 5 or len(head_boxes) >= 3:
        tags.append("crowded")
    if small_heads:
        tags.append("small_object")
    if head_overlap:
        tags.append("overlapping_boxes")
    if brightness < 85:
        tags.append("low_light")
    if edge_boxes:
        tags.append("edge_truncation")
    score = (
        min(len(annotations), 8) * 0.8
        + small_heads * 1.5
        + int(head_overlap) * 2.0
        + int(brightness < 85) * 2.0
        + min(edge_boxes, 3) * 0.5
    )
    return {
        "image": image,
        "annotations": annotations,
        "tags": tags,
        "difficulty_score": round(score, 4),
        "brightness": round(brightness, 2),
        "small_head_count": small_heads,
        "head_count": len(head_boxes),
        "annotation_count": len(annotations),
        "edge_box_count": edge_boxes,
        "sha256": file_sha256(image_path),
    }


def choose_diverse(records: list[dict[str, Any]], limit: int, minimum_per_tag: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: record["difficulty_score"], reverse=True)
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    for tag in TAGS:
        matching = [record for record in ordered if tag in record["tags"]]
        for record in matching[:minimum_per_tag]:
            image_id = int(record["image"]["id"])
            if image_id not in chosen_ids and len(chosen) < limit:
                chosen.append(record)
                chosen_ids.add(image_id)
    for record in ordered:
        image_id = int(record["image"]["id"])
        if image_id not in chosen_ids and len(chosen) < limit:
            chosen.append(record)
            chosen_ids.add(image_id)
    return sorted(chosen, key=lambda record: int(record["image"]["id"]))


def coco_subset(coco: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {int(record["image"]["id"]) for record in records}
    return {
        "info": {
            "description": "EdgeVision Hard Subset v1; deterministic image-only selection without model predictions",
            "source_split": "data/splits/test.json",
        },
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [image for image in coco["images"] if int(image["id"]) in selected_ids],
        "annotations": [annotation for annotation in coco["annotations"] if int(annotation["image_id"]) in selected_ids],
    }


def write_outputs(records: list[dict[str, Any]], coco: dict[str, Any], args: argparse.Namespace) -> None:
    metadata_path = Path(args.metadata_output)
    annotations_path = Path(args.annotations_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "challenge_image_id", "edgevision_image_id", "local_filename", "source_group_id", "source_title",
        "license_name", "sha256", "difficulty_score", "difficulty_tags", "annotation_count", "head_count",
        "small_head_count", "brightness_mean", "edge_box_count", "intended_use", "review_status", "notes",
    )
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            image = record["image"]
            writer.writerow(
                {
                    "challenge_image_id": f"edgevision_hard_{index:03d}",
                    "edgevision_image_id": image["id"],
                    "local_filename": image["file_name"],
                    "source_group_id": f"edgevision_image_{image['id']}",
                    "source_title": "EdgeVision Dataset v1",
                    "license_name": "CC BY 4.0 (theo README dữ liệu dự án; cần đối chiếu trang DOI trước khi báo cáo)",
                    "sha256": record["sha256"],
                    "difficulty_score": record["difficulty_score"],
                    "difficulty_tags": ";".join(record["tags"]),
                    "annotation_count": record["annotation_count"],
                    "head_count": record["head_count"],
                    "small_head_count": record["small_head_count"],
                    "brightness_mean": record["brightness"],
                    "edge_box_count": record["edge_box_count"],
                    "intended_use": "edgevision_hard_subset_only",
                    "review_status": "candidate_review_required",
                    "notes": "Derived before challenge inference; not an external generalization benchmark.",
                }
            )
    annotations_path.write_text(json.dumps(coco_subset(coco, records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    annotation_path, image_root = Path(args.annotations), Path(args.images)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    records = []
    for image in coco["images"]:
        image_path = image_root / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh EdgeVision: {image_path}")
        records.append(score_image(image, annotations_by_image[int(image["id"])], image_path))
    selected = choose_diverse(records, args.limit, args.minimum_per_tag)
    if len(selected) != args.limit:
        raise ValueError(f"Chỉ chọn được {len(selected)} ảnh, nhỏ hơn limit={args.limit}")
    write_outputs(selected, coco, args)
    tag_counts = {tag: sum(tag in record["tags"] for record in selected) for tag in TAGS}
    print(json.dumps({"selected": len(selected), "tag_counts": tag_counts, "mean_score": round(sum(record["difficulty_score"] for record in selected) / len(selected), 3)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chọn EdgeVision Hard Subset theo đặc trưng ảnh/annotation")
    parser.add_argument("--annotations", default="data/splits/test.json")
    parser.add_argument("--images", default="data/raw/edgevision/images")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--minimum-per-tag", type=int, default=8)
    parser.add_argument("--metadata-output", default="data/challenge/metadata/edgevision_hard_subset_v1.csv")
    parser.add_argument("--annotations-output", default="data/challenge/annotations/edgevision_hard_subset_v1.coco.json")
    args = parser.parse_args()
    if args.limit < 1 or args.minimum_per_tag < 0:
        parser.error("limit phải lớn hơn 0 và minimum-per-tag không âm")
    return args


if __name__ == "__main__":
    run(parse_args())
