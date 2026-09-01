"""Kiểm tra và đóng băng fingerprint của annotation processed cùng COCO splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze validated COCO split inputs")
    parser.add_argument("--processed", type=Path, default=Path("data/processed/edgevision/annotations.json"))
    parser.add_argument("--groups", type=Path, default=Path("data/processed/edgevision/image_hashes.json"))
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.json"))
    parser.add_argument("--val", type=Path, default=Path("data/splits/val.json"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/splits/frozen_manifest.json"))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy COCO JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"COCO JSON không phải object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO JSON thiếu list {key}: {path}")
    return value


def validate_splits(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seen_images: set[Any] = set()
    expected_categories: set[Any] | None = None
    summary: dict[str, Any] = {}
    for name, coco in splits.items():
        images = coco["images"]
        annotations = coco["annotations"]
        image_ids = {image.get("id") for image in images}
        if None in image_ids or len(image_ids) != len(images):
            raise ValueError(f"{name}: image ID thiếu hoặc trùng")
        if seen_images & image_ids:
            raise ValueError(f"{name}: image ID giao nhau với split khác")
        seen_images.update(image_ids)
        categories = {category.get("id") for category in coco["categories"]}
        if expected_categories is None:
            expected_categories = categories
        elif categories != expected_categories:
            raise ValueError(f"{name}: category mapping khác split còn lại")
        per_category = {category_id: 0 for category_id in categories}
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if annotation.get("image_id") not in image_ids:
                raise ValueError(f"{name}: annotation tham chiếu ảnh ngoài split")
            if annotation.get("category_id") not in categories:
                raise ValueError(f"{name}: annotation có category không hợp lệ")
            if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"{name}: annotation có bbox không hợp lệ")
            per_category[annotation["category_id"]] += 1
        if any(count == 0 for count in per_category.values()):
            raise ValueError(f"{name}: thiếu ít nhất một lớp")
        summary[name] = {
            "images": len(images),
            "annotations": len(annotations),
            "annotations_per_category": {str(key): value for key, value in sorted(per_category.items(), key=lambda item: str(item[0]))},
        }
    return {"image_union": len(seen_images), "splits": summary}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    files = {
        "processed_annotations": args.processed,
        "group_manifest": args.groups,
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cần đóng băng: {path}")
    splits = {"train": read_coco(args.train), "val": read_coco(args.val), "test": read_coco(args.test)}
    summary = validate_splits(splits)
    manifest = {
        "schema_version": "frozen-split-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "files": {name: {"path": str(path.as_posix()), "sha256": file_sha256(path)} for name, path in files.items()},
        "validation": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
