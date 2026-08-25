"""Tạo train/validation/test COCO split xác định theo nhóm ảnh."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic grouped COCO splits")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def _ratios(train: float, val: float, test: float) -> dict[str, float]:
    ratios = {"train": train, "val": val, "test": test}
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Ba tỷ lệ split phải dương và có tổng bằng 1")
    return ratios


def _mapping_from_manifest(manifest: dict[str, Any], expected_image_ids: set[Any]) -> dict[Any, str]:
    mapping: dict[Any, str] = {}
    for entry in manifest.get("images", []):
        image_id, group_id = entry.get("image_id"), entry.get("group_id")
        if image_id in mapping:
            raise ValueError(f"image_id lặp trong group manifest: {image_id}")
        if not group_id:
            raise ValueError(f"Thiếu group_id cho image_id: {image_id}")
        mapping[image_id] = str(group_id)
    missing = expected_image_ids - mapping.keys()
    extra = mapping.keys() - expected_image_ids
    if missing or extra:
        raise ValueError(f"Group manifest không khớp annotation: thiếu={len(missing)}, thừa={len(extra)}")
    return mapping


def _build_units(coco: dict[str, Any], image_to_group: dict[Any, str]) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for image in coco["images"]:
        group_id = image_to_group[image["id"]]
        unit = units.setdefault(group_id, {"image_ids": [], "categories": Counter()})
        unit["image_ids"].append(image["id"])
    for annotation in coco["annotations"]:
        image_id = annotation.get("image_id")
        if image_id not in image_to_group:
            raise ValueError(f"Annotation tham chiếu ảnh không có group: {image_id}")
        units[image_to_group[image_id]]["categories"][annotation.get("category_id")] += 1
    return units


def _score_assignment(
    counts: dict[str, int],
    category_counts: dict[str, Counter[Any]],
    add_split: str,
    unit: dict[str, Any],
    ratios: dict[str, float],
    total_images: int,
    total_categories: Counter[Any],
) -> float:
    score = 0.0
    for split in SPLIT_NAMES:
        projected_images = counts[split] + (len(unit["image_ids"]) if split == add_split else 0)
        target_images = max(total_images * ratios[split], 1.0)
        score += 0.7 * ((projected_images - target_images) / target_images) ** 2
        for category, total in total_categories.items():
            projected_category = category_counts[split][category] + (unit["categories"][category] if split == add_split else 0)
            target_category = max(total * ratios[split], 1.0)
            score += 0.3 * ((projected_category - target_category) / target_category) ** 2
    return score


def assign_groups(coco: dict[str, Any], image_to_group: dict[Any, str], ratios: dict[str, float], seed: int) -> dict[str, set[Any]]:
    """Assign groups greedily while balancing image and bbox-category counts."""
    units = _build_units(coco, image_to_group)
    total_images = len(coco["images"])
    total_categories = Counter(annotation.get("category_id") for annotation in coco["annotations"])
    rng = random.Random(seed)
    ordered_groups = list(units.items())
    rng.shuffle(ordered_groups)
    ordered_groups.sort(key=lambda item: len(item[1]["image_ids"]), reverse=True)

    counts = {split: 0 for split in SPLIT_NAMES}
    category_counts = {split: Counter() for split in SPLIT_NAMES}
    result = {split: set() for split in SPLIT_NAMES}
    for _group_id, unit in ordered_groups:
        selected = min(
            SPLIT_NAMES,
            key=lambda split: (_score_assignment(counts, category_counts, split, unit, ratios, total_images, total_categories), SPLIT_NAMES.index(split)),
        )
        result[selected].update(unit["image_ids"])
        counts[selected] += len(unit["image_ids"])
        category_counts[selected].update(unit["categories"])
    return result


def validate_splits(splits: dict[str, dict[str, Any]], image_to_group: dict[Any, str]) -> None:
    seen_images: set[Any] = set()
    seen_groups: set[str] = set()
    for split in SPLIT_NAMES:
        image_ids = {image["id"] for image in splits[split]["images"]}
        groups = {image_to_group[image_id] for image_id in image_ids}
        if seen_images & image_ids:
            raise ValueError("Phát hiện image_id xuất hiện ở nhiều split")
        if seen_groups & groups:
            raise ValueError("Phát hiện group xuất hiện ở nhiều split")
        seen_images.update(image_ids)
        seen_groups.update(groups)
        if any(annotation["image_id"] not in image_ids for annotation in splits[split]["annotations"]):
            raise ValueError(f"Annotation không thuộc ảnh của split {split}")


def build_splits(coco: dict[str, Any], image_to_group: dict[Any, str], ratios: dict[str, float], seed: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    assignments = assign_groups(coco, image_to_group, ratios, seed)
    annotations_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    image_to_split = {image_id: split for split, ids in assignments.items() for image_id in ids}
    for annotation in coco["annotations"]:
        annotations_by_split[image_to_split[annotation["image_id"]]].append(copy.deepcopy(annotation))

    shared = {key: copy.deepcopy(value) for key, value in coco.items() if key not in {"images", "annotations"}}
    output: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {"seed": seed, "ratios": ratios, "splits": {}}
    for split in SPLIT_NAMES:
        image_ids = assignments[split]
        split_coco = copy.deepcopy(shared)
        split_coco["images"] = [copy.deepcopy(image) for image in coco["images"] if image["id"] in image_ids]
        split_coco["annotations"] = annotations_by_split[split]
        output[split] = split_coco
        summary["splits"][split] = {
            "images": len(split_coco["images"]),
            "annotations": len(split_coco["annotations"]),
            "annotations_per_category": dict(sorted(Counter(item["category_id"] for item in split_coco["annotations"]).items(), key=lambda item: str(item[0]))),
            "group_count": len({image_to_group[image_id] for image_id in image_ids}),
        }
    validate_splits(output, image_to_group)
    return output, summary


def main() -> None:
    args = parse_args()
    ratios = _ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    coco = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    manifest = json.loads(args.groups.read_text(encoding="utf-8"))
    image_ids = {image["id"] for image in coco.get("images", [])}
    image_to_group = _mapping_from_manifest(manifest, image_ids)
    splits, summary = build_splits(coco, image_to_group, ratios, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_coco in splits.items():
        (args.output_dir / f"{split}.json").write_text(json.dumps(split_coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_out = {"seed": args.seed, "ratios": ratios, "image_to_split": {str(image_id): split for split, data in splits.items() for image_id in (item["id"] for item in data["images"])}}
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
