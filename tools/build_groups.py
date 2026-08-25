"""Tạo manifest nhóm ảnh từ hash chính xác và các ứng viên gần trùng lặp."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an image-group manifest for COCO splits")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--near-duplicate-output", required=True, type=Path)
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        help="Ngưỡng Hamming 0..7. Bỏ qua để chỉ lập nhóm ảnh trùng chính xác.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8)).get_flattened_data())
    average = sum(pixels) / len(pixels)
    result = 0
    for value in pixels:
        result = (result << 1) | int(value >= average)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_manifest(coco: dict[str, Any], images_dir: Path, near_duplicate_distance: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if near_duplicate_distance is not None and not 0 <= near_duplicate_distance <= 7:
        raise ValueError("near_duplicate_distance phải nằm trong khoảng 0..7")
    entries: list[dict[str, Any]] = []
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in sorted(coco.get("images", []), key=lambda item: item["id"]):
        path = images_dir / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
        entry: dict[str, Any] = {
            "image_id": image["id"],
            "file_name": image["file_name"],
            "sha256": sha256_file(path),
        }
        if near_duplicate_distance is not None:
            entry["average_hash"] = f"{average_hash(path):016x}"
        by_hash[entry["sha256"]].append(entry)
        entries.append(entry)

    exact_groups: list[dict[str, Any]] = []
    for digest, members in sorted(by_hash.items()):
        group_id = f"hash:{digest}"
        for member in members:
            member["group_id"] = group_id
        if len(members) > 1:
            exact_groups.append({"group_id": group_id, "image_ids": [member["image_id"] for member in members]})

    if near_duplicate_distance is None:
        manifest = {
            "version": 1,
            "grouping_policy": "exact_hash_only; near-duplicate review was not requested",
            "images": entries,
            "exact_duplicate_groups": exact_groups,
        }
        return manifest, []

    # Với ngưỡng <= 7 bit, hai hash cách nhau trong ngưỡng luôn có ít nhất một
    # byte giống hệt nhau. Lập bucket theo 8 byte giúp tránh duyệt 5,7 triệu cặp.
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        value = int(entry["average_hash"], 16)
        for byte_index in range(8):
            buckets[(byte_index, (value >> (byte_index * 8)) & 0xFF)].append(entry)

    candidate_pairs: set[tuple[Any, Any]] = set()
    by_id = {entry["image_id"]: entry for entry in entries}
    for bucket in buckets.values():
        for left_index in range(len(bucket)):
            for right_index in range(left_index + 1, len(bucket)):
                left_id, right_id = sorted((bucket[left_index]["image_id"], bucket[right_index]["image_id"]), key=str)
                candidate_pairs.add((left_id, right_id))

    candidates: list[dict[str, Any]] = []
    for left_id, right_id in sorted(candidate_pairs, key=lambda pair: (str(pair[0]), str(pair[1]))):
        left, right = by_id[left_id], by_id[right_id]
        if left["sha256"] == right["sha256"]:
            continue
        distance = hamming_distance(int(left["average_hash"], 16), int(right["average_hash"], 16))
        if distance <= near_duplicate_distance:
            candidates.append({"left_image_id": left_id, "right_image_id": right_id, "distance": distance})

    manifest = {
        "version": 1,
        "grouping_policy": "exact_hash_only; near-duplicate candidates require manual scene review",
        "images": entries,
        "exact_duplicate_groups": exact_groups,
    }
    return manifest, candidates


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    manifest, candidates = build_manifest(coco, args.images, args.near_duplicate_distance)
    for path, content in ((args.output, manifest), (args.near_duplicate_output, candidates)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(manifest["images"]), "exact_duplicate_groups": len(manifest["exact_duplicate_groups"]), "near_duplicate_candidates": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
