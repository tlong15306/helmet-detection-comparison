"""Phát hiện file trùng và ảnh gần trùng trong một thư mục Challenge Set."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def average_hash(path: Path, size: int = 16) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((size, size))
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def inspect(input_dir: Path, threshold: int) -> dict[str, Any]:
    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    records = [{"filename": path.name, "sha256": sha256(path), "ahash": average_hash(path)} for path in paths]
    exact_groups: dict[str, list[str]] = {}
    for record in records:
        exact_groups.setdefault(record["sha256"], []).append(record["filename"])
    exact_duplicates = [names for names in exact_groups.values() if len(names) > 1]
    near_duplicates = []
    for first, second in combinations(records, 2):
        distance = hamming_distance(first["ahash"], second["ahash"])
        if distance <= threshold and first["sha256"] != second["sha256"]:
            near_duplicates.append(
                {"first": first["filename"], "second": second["filename"], "hamming_distance": distance}
            )
    return {
        "image_count": len(records),
        "hash_algorithm": "sha256 + 16x16 average hash",
        "near_duplicate_threshold": threshold,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra ảnh trùng Challenge Set")
    parser.add_argument("--input-dir", default="data/challenge/candidates")
    parser.add_argument("--output", default="data/challenge/metadata/duplicate_report.json")
    parser.add_argument("--threshold", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 256:
        parser.error("--threshold phải nằm trong [0, 256]")
    return args


def main() -> None:
    args = parse_args()
    report = inspect(Path(args.input_dir), args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"image_count": report["image_count"], "exact_groups": len(report["exact_duplicates"]), "near_pairs": len(report["near_duplicates"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
