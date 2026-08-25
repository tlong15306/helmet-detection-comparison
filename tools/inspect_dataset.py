"""Kiểm định cấu trúc ảnh và annotation COCO của EdgeVision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra ảnh và annotation COCO của dataset"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument(
        "--images",
        type=Path,
        help="Thư mục ảnh. Nếu bỏ qua, chỉ kiểm tra nội dung JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Tệp JSON nhận báo cáo kiểm định.",
    )
    parser.add_argument(
        "--max-error-samples",
        type=int,
        default=20,
        help="Số ví dụ tối đa lưu cho mỗi loại lỗi.",
    )
    return parser.parse_args()


def _duplicates(values: list[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(
        (value for value, count in counts.items() if count > 1),
        key=str,
    )


def _sample(values: list[Any], limit: int) -> list[Any]:
    return values[: max(limit, 0)]


def _require_coco_lists(coco: dict[str, Any]) -> None:
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"COCO JSON thiếu trường bắt buộc: {key}")
        if not isinstance(coco[key], list):
            raise ValueError(f"COCO JSON: trường {key} phải là một danh sách")


def inspect_annotations(coco: dict[str, Any], sample_limit: int) -> dict[str, Any]:
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_ids = [image.get("id") for image in images]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    category_ids = [category.get("id") for category in categories]

    duplicate_image_ids = _duplicates(image_ids)
    duplicate_annotation_ids = _duplicates(annotation_ids)
    duplicate_category_ids = _duplicates(category_ids)

    image_by_id = {
        image.get("id"): image
        for image in images
        if image.get("id") is not None
    }
    category_names = {
        category.get("id"): str(category.get("name", ""))
        for category in categories
        if category.get("id") is not None
    }

    malformed_images: list[Any] = []
    for image in images:
        if (
            image.get("id") is None
            or not isinstance(image.get("file_name"), str)
            or not image.get("file_name")
            or not isinstance(image.get("width"), (int, float))
            or not isinstance(image.get("height"), (int, float))
            or image.get("width", 0) <= 0
            or image.get("height", 0) <= 0
        ):
            malformed_images.append(image.get("id"))

    invalid_boxes: list[Any] = []
    boxes_outside_image: list[Any] = []
    annotations_missing_images: list[Any] = []
    annotations_unknown_categories: list[Any] = []
    annotations_per_category: Counter[Any] = Counter()
    annotations_per_image: Counter[Any] = Counter()

    for annotation in annotations:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")

        annotations_per_category[category_id] += 1
        annotations_per_image[image_id] += 1

        image = image_by_id.get(image_id)
        if image is None:
            annotations_missing_images.append(annotation_id)
        if category_id not in category_names:
            annotations_unknown_categories.append(annotation_id)

        valid_bbox = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
            and bbox[2] > 0
            and bbox[3] > 0
        )
        if not valid_bbox:
            invalid_boxes.append(annotation_id)
            continue

        if image is not None:
            x, y, width, height = bbox
            image_width = image.get("width")
            image_height = image.get("height")
            dimensions_valid = (
                isinstance(image_width, (int, float))
                and isinstance(image_height, (int, float))
                and image_width > 0
                and image_height > 0
            )
            if dimensions_valid and (
                x < 0
                or y < 0
                or x + width > image_width
                or y + height > image_height
            ):
                boxes_outside_image.append(annotation_id)

    images_without_annotations = [
        image_id for image_id in image_ids if annotations_per_image[image_id] == 0
    ]

    issues = {
        "duplicate_image_ids": _sample(duplicate_image_ids, sample_limit),
        "duplicate_annotation_ids": _sample(duplicate_annotation_ids, sample_limit),
        "duplicate_category_ids": _sample(duplicate_category_ids, sample_limit),
        "malformed_images": _sample(malformed_images, sample_limit),
        "invalid_boxes": _sample(invalid_boxes, sample_limit),
        "boxes_outside_image": _sample(boxes_outside_image, sample_limit),
        "annotations_missing_images": _sample(
            annotations_missing_images, sample_limit
        ),
        "annotations_unknown_categories": _sample(
            annotations_unknown_categories, sample_limit
        ),
        "images_without_annotations": _sample(
            images_without_annotations, sample_limit
        ),
    }
    issue_counts = {
        "duplicate_image_ids": len(duplicate_image_ids),
        "duplicate_annotation_ids": len(duplicate_annotation_ids),
        "duplicate_category_ids": len(duplicate_category_ids),
        "malformed_images": len(malformed_images),
        "invalid_boxes": len(invalid_boxes),
        "boxes_outside_image": len(boxes_outside_image),
        "annotations_missing_images": len(annotations_missing_images),
        "annotations_unknown_categories": len(annotations_unknown_categories),
        "images_without_annotations": len(images_without_annotations),
    }

    return {
        "counts": {
            "images_in_json": len(images),
            "annotations": len(annotations),
            "categories": len(categories),
        },
        "categories": [
            {"id": category_id, "name": category_names[category_id]}
            for category_id in sorted(category_names, key=str)
        ],
        "annotations_per_category": {
            category_names.get(category_id, str(category_id)): count
            for category_id, count in sorted(
                annotations_per_category.items(), key=lambda item: str(item[0])
            )
        },
        "issue_counts": issue_counts,
        "issue_samples": issues,
    }


def inspect_image_files(
    images_dir: Path,
    coco_images: list[dict[str, Any]],
    sample_limit: int,
) -> dict[str, Any]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

    root = images_dir.resolve()
    disk_images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    disk_relative_names = {
        path.relative_to(images_dir).as_posix() for path in disk_images
    }

    referenced_names: set[str] = set()
    missing_files: list[str] = []
    unsafe_paths: list[str] = []
    unreadable_files: list[str] = []
    dimension_mismatches: list[dict[str, Any]] = []

    for image_record in coco_images:
        file_name = image_record.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue

        normalized_name = Path(file_name).as_posix()
        referenced_names.add(normalized_name)
        image_path = images_dir / Path(file_name)
        resolved_path = image_path.resolve()

        if not resolved_path.is_relative_to(root):
            unsafe_paths.append(file_name)
            continue
        if not image_path.is_file():
            missing_files.append(file_name)
            continue

        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image.verify()
        except (OSError, UnidentifiedImageError):
            unreadable_files.append(file_name)
            continue

        declared_size = (image_record.get("width"), image_record.get("height"))
        if declared_size != actual_size:
            dimension_mismatches.append(
                {
                    "file_name": file_name,
                    "declared": list(declared_size),
                    "actual": list(actual_size),
                }
            )

    unreferenced_files = sorted(disk_relative_names - referenced_names)
    return {
        "images_on_disk": len(disk_images),
        "referenced_image_names": len(referenced_names),
        "issue_counts": {
            "missing_image_files": len(missing_files),
            "unsafe_image_paths": len(unsafe_paths),
            "unreadable_image_files": len(unreadable_files),
            "dimension_mismatches": len(dimension_mismatches),
            "unreferenced_image_files": len(unreferenced_files),
        },
        "issue_samples": {
            "missing_image_files": _sample(missing_files, sample_limit),
            "unsafe_image_paths": _sample(unsafe_paths, sample_limit),
            "unreadable_image_files": _sample(unreadable_files, sample_limit),
            "dimension_mismatches": _sample(
                dimension_mismatches, sample_limit
            ),
            "unreferenced_image_files": _sample(
                unreferenced_files, sample_limit
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_error_samples < 0:
        raise ValueError("--max-error-samples không được âm")
    if not args.annotations.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tệp annotation: {args.annotations}"
        )

    with args.annotations.open("r", encoding="utf-8-sig") as stream:
        coco = json.load(stream)
    if not isinstance(coco, dict):
        raise ValueError("COCO JSON phải là một object ở cấp cao nhất")
    _require_coco_lists(coco)

    report: dict[str, Any] = {
        "annotations_path": str(args.annotations.resolve()),
        "annotation_checks": inspect_annotations(coco, args.max_error_samples),
    }
    if args.images is not None:
        report["images_path"] = str(args.images.resolve())
        report["image_checks"] = inspect_image_files(
            args.images,
            coco["images"],
            args.max_error_samples,
        )

    report["status"] = "completed"
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
