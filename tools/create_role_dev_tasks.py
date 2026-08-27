"""Tạo task review vai trò từ ground truth validation, không tự gán role."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.rider_association import analyze_rider_roles
from src.role_annotations import validate_role_tasks
from src.utils import resolve_project_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _display_path(path: Path) -> str:
    """Ưu tiên đường dẫn tương đối dự án, vẫn hỗ trợ fixture ngoài workspace."""
    try:
        return str(path.relative_to(resolve_project_path(".")))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo role_dev pending từ COCO validation")
    parser.add_argument("--annotations", default="data/splits/val.json")
    parser.add_argument("--image-root", default="data/raw/edgevision/images")
    parser.add_argument("--max-tasks", type=int, default=80)
    parser.add_argument("--output", default="data/role_association/annotations/role_dev.pending.json")
    parser.add_argument("--summary-output", default="data/role_association/metadata/role_dev_selection_summary.json")
    args = parser.parse_args()
    if args.max_tasks < 1:
        parser.error("--max-tasks phải lớn hơn 0")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(annotation: Mapping[str, Any], category_name: str) -> dict[str, Any]:
    x, y, width, height = annotation["bbox"]
    return {
        "detection_id": f"annotation_{annotation['id']}",
        "class_name": category_name,
        "confidence": 1.0,
        "box": [float(x), float(y), float(x + width), float(y + height)],
    }


def _annotation_id(detection_id: str) -> int:
    prefix = "annotation_"
    if not detection_id.startswith(prefix):
        raise ValueError(f"Không đọc được annotation ID: {detection_id}")
    return int(detection_id[len(prefix) :])


def _tags(image: Mapping[str, Any], all_records: list[dict[str, Any]], heads: list[dict[str, Any]]) -> list[str]:
    tags = ["single_head" if len(heads) == 1 else "multiple_heads"]
    vehicle_count = sum(record["class_name"] == "BikeWithRider" for record in all_records)
    if vehicle_count >= 3:
        tags.append("crowded_scene")
    image_area = float(image["width"]) * float(image["height"])
    if any(
        ((head["box_xyxy"][2] - head["box_xyxy"][0]) * (head["box_xyxy"][3] - head["box_xyxy"][1])) / image_area <= 0.008
        for head in heads
    ):
        tags.append("small_head")
    if any(
        head["box_xyxy"][0] <= 1
        or head["box_xyxy"][1] <= 1
        or head["box_xyxy"][2] >= float(image["width"]) - 1
        or head["box_xyxy"][3] >= float(image["height"]) - 1
        for head in heads
    ):
        tags.append("edge_truncation")
    return tags


def _round_robin(candidates: list[dict[str, Any]], max_tasks: int) -> list[dict[str, Any]]:
    """Chọn đa dạng theo tag, ưu tiên mỗi ảnh chỉ có một task."""
    target_per_tag = {
        "multiple_heads": 25,
        "single_head": 25,
        "crowded_scene": 15,
        "small_head": 15,
        "edge_truncation": 8,
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_images: set[int] = set()
    for tag, quota in target_per_tag.items():
        selected_for_tag = 0
        for candidate in candidates:
            if len(selected) >= max_tasks or selected_for_tag >= quota:
                break
            if tag not in candidate["difficulty_tags"]:
                continue
            if candidate["task_id"] in selected_ids or candidate["image_id"] in selected_images:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["task_id"])
            selected_images.add(candidate["image_id"])
            selected_for_tag += 1
    for candidate in candidates:
        if len(selected) >= max_tasks:
            break
        if candidate["task_id"] in selected_ids:
            continue
        if candidate["image_id"] in selected_images:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["task_id"])
        selected_images.add(candidate["image_id"])
    for candidate in candidates:
        if len(selected) >= max_tasks:
            break
        if candidate["task_id"] in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["task_id"])
    return selected


def _task_candidates(coco: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    images = {int(item["id"]): item for item in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    raw_candidates: list[dict[str, Any]] = []
    for image_id in sorted(images):
        image = images[image_id]
        annotations = annotations_by_image.get(image_id, [])
        annotation_index = {int(annotation["id"]): annotation for annotation in annotations}
        records = [
            _record(annotation, categories[int(annotation["category_id"])])
            for annotation in annotations
            if categories[int(annotation["category_id"])] in {"BikeWithRider", "Helmet", "NoHelmet"}
        ]
        analysis = analyze_rider_roles(records)
        for group in analysis["rider_groups"]:
            if not group["heads"]:
                continue
            bike_annotation_id = _annotation_id(str(group["bike_detection_id"]))
            heads: list[dict[str, Any]] = []
            for head in group["heads"]:
                annotation_id = _annotation_id(str(head["head_detection_id"]))
                source = annotation_index[annotation_id]
                heads.append(
                    {
                        "annotation_id": annotation_id,
                        "helmet_status": head["helmet_status"],
                        "box_xyxy": head["head_box"],
                        "source_category_id": int(source["category_id"]),
                    }
                )
            raw_candidates.append(
                {
                    "image_id": image_id,
                    "file_name": str(image["file_name"]),
                    "image_width": int(image["width"]),
                    "image_height": int(image["height"]),
                    "bike_annotation_id": bike_annotation_id,
                    "bike_box_xyxy": group["bike_box"],
                    "heads": heads,
                    "difficulty_tags": _tags(image, records, heads),
                    "proposed_driver_head_annotation_id": (
                        _annotation_id(str(group["driver"]["head_detection_id"])) if group["driver"] else None
                    ),
                    "proposal_status": "candidate_only" if group["driver"] else "unknown",
                }
            )
    raw_candidates.sort(
        key=lambda item: (-len(item["difficulty_tags"]), item["image_id"], item["bike_annotation_id"])
    )
    for index, candidate in enumerate(raw_candidates, start=1):
        candidate["task_id"] = f"role_dev_{index:03d}"
    return raw_candidates


def create_tasks(annotations_path: str | Path, image_root: str, max_tasks: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source = resolve_project_path(annotations_path)
    with source.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)
    selected = _round_robin(list(_task_candidates(coco)), max_tasks)
    selected.sort(key=lambda item: (item["image_id"], item["bike_annotation_id"]))
    tasks: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        tasks.append(
            {
                **candidate,
                "task_id": f"role_dev_{index:03d}",
                "image_path": str(Path(image_root) / candidate["file_name"]),
                "review": {
                    "status": "pending",
                    "reviewer": None,
                    "reviewed_at": None,
                    "driver_head_annotation_id": None,
                    "head_roles": {str(head["annotation_id"]): None for head in candidate["heads"]},
                    "notes": None,
                },
            }
        )
    payload = {
        "schema_version": "role_association_tasks_v1",
        "source": {
            "split": "validation",
            "annotations": _display_path(source),
            "annotations_sha256": _sha256(source),
            "image_root": image_root,
            "selection": "ground_truth_geometry_only; no model predictions; no role labels inferred",
        },
        "tasks": tasks,
    }
    validation = validate_role_tasks(payload)
    tag_counts: Counter[str] = Counter(tag for task in tasks for tag in task["difficulty_tags"])
    summary = {
        "source": payload["source"],
        "selection": validation,
        "difficulty_tags": dict(sorted(tag_counts.items())),
        "proposals": dict(Counter(task["proposal_status"] for task in tasks)),
        "warning": "Mọi role vẫn pending. proposed_driver_head_annotation_id chỉ là gợi ý baseline, không phải ground truth.",
    }
    return payload, summary


def main() -> None:
    args = parse_args()
    payload, summary = create_tasks(args.annotations, args.image_root, args.max_tasks)
    output = resolve_project_path(args.output)
    summary_output = resolve_project_path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
