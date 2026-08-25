"""Tạo annotation COCO processed mà không sửa dữ liệu gốc.

Chính sách mặc định ưu tiên an toàn: chỉ clip bbox vượt biên nhỏ; bbox lỗi
nghiêm trọng được loại khỏi bản processed và luôn có bản ghi truy vết.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a reviewed COCO annotation copy")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", type=Path, help="Thư mục ảnh gốc để đọc EXIF.")
    parser.add_argument(
        "--apply-exif-orientation",
        action="store_true",
        help="Đồng bộ kích thước COCO với ảnh đã được xoay theo EXIF.",
    )
    parser.add_argument("--processed-output", required=True, type=Path)
    parser.add_argument("--changes-output", required=True, type=Path)
    parser.add_argument("--problems-output", required=True, type=Path)
    parser.add_argument(
        "--minor-overflow-pixels",
        type=float,
        default=20.0,
        help="Chỉ clip bbox nếu độ vượt biên lớn nhất không quá ngưỡng này.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _box_is_numeric(bbox: Any) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
    )


def _overflow(bbox: list[float], image_width: float, image_height: float) -> dict[str, float]:
    x, y, width, height = bbox
    return {
        "left": max(0.0, -x),
        "top": max(0.0, -y),
        "right": max(0.0, x + width - image_width),
        "bottom": max(0.0, y + height - image_height),
    }


def _clipped_box(bbox: list[float], image_width: float, image_height: float) -> list[float]:
    x, y, width, height = bbox
    x1 = min(max(x, 0.0), image_width)
    y1 = min(max(y, 0.0), image_height)
    x2 = min(max(x + width, 0.0), image_width)
    y2 = min(max(y + height, 0.0), image_height)
    return [x1, y1, x2 - x1, y2 - y1]


def _oriented_image_record(image: dict[str, Any], images_dir: Path | None, apply_exif_orientation: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return an image copy with EXIF-display dimensions when explicitly enabled."""
    updated = copy.deepcopy(image)
    if not apply_exif_orientation:
        return updated, None
    if images_dir is None:
        raise ValueError("--apply-exif-orientation yêu cầu --images")
    path = images_dir / image["file_name"]
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    with Image.open(path) as opened:
        orientation = opened.getexif().get(274)
        raw_size = opened.size
    oriented_size = raw_size[::-1] if orientation in {5, 6, 7, 8} else raw_size
    if tuple(oriented_size) == (image["width"], image["height"]):
        return updated, None
    updated["width"], updated["height"] = oriented_size
    return updated, {
        "entity": "image",
        "image_id": image["id"],
        "action": "update_display_dimensions",
        "reason": "exif_orientation",
        "exif_orientation": orientation,
        "original_size": [image["width"], image["height"]],
        "new_size": list(oriented_size),
    }


def prepare_coco(
    coco: dict[str, Any],
    minor_overflow_pixels: float,
    images_dir: Path | None = None,
    apply_exif_orientation: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return processed COCO, a change log and all problem records."""
    if minor_overflow_pixels < 0:
        raise ValueError("minor_overflow_pixels không được âm")
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            raise ValueError(f"COCO JSON thiếu danh sách {key}")

    categories = {item.get("id") for item in coco["categories"]}
    images_by_id = {item.get("id"): item for item in coco["images"] if item.get("id") is not None}
    valid_images: list[dict[str, Any]] = []
    valid_image_ids: set[Any] = set()
    changes: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    for image in coco["images"]:
        image_id = image.get("id")
        width, height = image.get("width"), image.get("height")
        valid = (
            image_id is not None
            and isinstance(image.get("file_name"), str)
            and bool(image.get("file_name"))
            and isinstance(width, (int, float))
            and isinstance(height, (int, float))
            and width > 0
            and height > 0
        )
        if valid:
            updated, orientation_change = _oriented_image_record(image, images_dir, apply_exif_orientation)
            valid_images.append(updated)
            valid_image_ids.add(image_id)
            images_by_id[image_id] = updated
            if orientation_change is not None:
                changes.append(orientation_change)
            continue
        record = {
            "entity": "image",
            "image_id": image_id,
            "action": "exclude_image",
            "reason": "malformed_image_metadata",
            "original": image,
        }
        changes.append(record)
        problems.append(record)

    processed_annotations: list[dict[str, Any]] = []
    for annotation in coco["annotations"]:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        image = images_by_id.get(image_id)
        base = {"entity": "annotation", "annotation_id": annotation_id, "image_id": image_id}

        if image_id not in valid_image_ids:
            record = {**base, "action": "exclude_annotation", "reason": "missing_or_invalid_image", "original_bbox": bbox}
            changes.append(record)
            problems.append(record)
            continue
        if category_id not in categories:
            record = {**base, "action": "exclude_annotation", "reason": "unknown_category", "original_bbox": bbox}
            changes.append(record)
            problems.append(record)
            continue
        if not _box_is_numeric(bbox):
            record = {**base, "action": "exclude_annotation", "reason": "malformed_bbox", "original_bbox": bbox}
            changes.append(record)
            problems.append(record)
            continue

        numeric_bbox = [float(value) for value in bbox]
        if numeric_bbox[2] <= 0 or numeric_bbox[3] <= 0:
            record = {**base, "action": "exclude_annotation", "reason": "non_positive_bbox_size", "original_bbox": bbox}
            changes.append(record)
            problems.append(record)
            continue

        image_width, image_height = float(image["width"]), float(image["height"])
        overflow = _overflow(numeric_bbox, image_width, image_height)
        max_overflow = max(overflow.values())
        if max_overflow == 0:
            processed_annotations.append(copy.deepcopy(annotation))
            continue

        problem = {
            **base,
            "category_id": category_id,
            "file_name": image["file_name"],
            "image_size": [image_width, image_height],
            "original_bbox": bbox,
            "overflow_pixels": overflow,
            "max_overflow_pixels": max_overflow,
        }
        if max_overflow <= minor_overflow_pixels:
            new_bbox = _clipped_box(numeric_bbox, image_width, image_height)
            if new_bbox[2] > 0 and new_bbox[3] > 0:
                updated = copy.deepcopy(annotation)
                updated["bbox"] = new_bbox
                updated["area"] = new_bbox[2] * new_bbox[3]
                processed_annotations.append(updated)
                record = {**problem, "action": "clip", "reason": "minor_bbox_overflow", "new_bbox": new_bbox}
                changes.append(record)
                problems.append(record)
                continue

        record = {**problem, "action": "exclude_annotation", "reason": "severe_bbox_overflow_requires_review"}
        changes.append(record)
        problems.append(record)

    processed = {
        key: copy.deepcopy(value)
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    processed["images"] = valid_images
    processed["annotations"] = processed_annotations
    summary = Counter(record["action"] for record in changes)
    processed["processing_metadata"] = {
        "policy": {
            "minor_overflow_pixels": minor_overflow_pixels,
            "severe_overflow": "excluded_pending_manual_review",
            "exif_orientation_applied": apply_exif_orientation,
            "raw_data_modified": False,
        },
        "changes": dict(sorted(summary.items())),
    }
    return processed, changes, problems


def main() -> None:
    args = parse_args()
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Không tìm thấy annotation: {args.annotations}")
    coco = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    processed, changes, problems = prepare_coco(
        coco,
        args.minor_overflow_pixels,
        images_dir=args.images,
        apply_exif_orientation=args.apply_exif_orientation,
    )
    _write_json(args.processed_output, processed)
    _write_json(args.changes_output, changes)
    _write_json(args.problems_output, problems)
    print(json.dumps({"processed_annotations": len(processed["annotations"]), "changes": len(changes), "problems": len(problems)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
