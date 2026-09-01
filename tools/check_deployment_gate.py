"""Chặn checkpoint mới nếu chất lượng validation thấp hơn baseline.

Gate chỉ đọc lịch sử validation và annotation validation. Test set không được
đọc ở bước này. Một candidate đạt gate tự động vẫn cần duyệt ảnh lỗi khó trước
khi thay checkpoint đang chạy trên demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "deployment-gate-1.0"


def _configure_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra checkpoint trước khi deploy")
    parser.add_argument("--baseline-history", required=True)
    parser.add_argument("--candidate-history", required=True)
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--candidate-val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protected-class",
        action="append",
        default=None,
        help="Lớp không được phép giảm AP; có thể truyền nhiều lần (mặc định: NoHelmet)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Mức giảm tuyệt đối tối đa được cho phép trên validation",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Trả exit code 1 nếu candidate bị chặn",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def best_validation_epoch(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Chọn epoch có mAP@0.5:0.95 validation cao nhất."""
    valid = [row for row in history if row.get("validation", {}).get("map_50_95") is not None]
    if not valid:
        raise ValueError("History không có metric validation map_50_95")
    row = max(valid, key=lambda item: float(item["validation"]["map_50_95"]))
    validation = row["validation"]
    return {
        "epoch": int(row["epoch"]),
        "map_50_95": float(validation["map_50_95"]),
        "map_50": (
            None if validation.get("map_50") is None else float(validation["map_50"])
        ),
        "per_class": {
            name: {
                "ap_50_95": float(metrics["ap_50_95"]),
                "ap_50": (
                    None if metrics.get("ap_50") is None else float(metrics["ap_50"])
                ),
            }
            for name, metrics in validation.get("per_class", {}).items()
        },
    }


def semantic_coco_fingerprint(annotation_path: str | Path) -> dict[str, Any]:
    """Băm nội dung COCO theo tên ảnh/lớp/hộp, bỏ qua ID và đường dẫn cha."""
    coco = _load_json(annotation_path)
    categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    images_by_id: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for image in coco["images"]:
        name = Path(str(image["file_name"])).name
        if name in names:
            raise ValueError(f"Tên ảnh bị trùng trong validation: {name}")
        names.add(name)
        images_by_id[int(image["id"])] = {
            "file_name": name,
            "width": int(image["width"]),
            "height": int(image["height"]),
        }

    normalized_images = sorted(images_by_id.values(), key=lambda item: item["file_name"])
    normalized_annotations = []
    for annotation in coco["annotations"]:
        image = images_by_id[int(annotation["image_id"])]
        category = categories[int(annotation["category_id"])]
        normalized_annotations.append(
            {
                "file_name": image["file_name"],
                "category": category,
                "bbox": [round(float(value), 6) for value in annotation["bbox"]],
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    normalized_annotations.sort(
        key=lambda item: (item["file_name"], item["category"], item["bbox"])
    )
    payload = {
        "images": normalized_images,
        "annotations": normalized_annotations,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "sha256": digest,
        "image_count": len(normalized_images),
        "annotation_count": len(normalized_annotations),
    }


def evaluate_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    same_validation: bool,
    protected_classes: Sequence[str],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    if tolerance < 0:
        raise ValueError("tolerance phải lớn hơn hoặc bằng 0")
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "same_validation_ground_truth",
            "passed": bool(same_validation),
            "baseline": same_validation,
            "candidate": same_validation,
        }
    )

    baseline_map = float(baseline["map_50_95"])
    candidate_map = float(candidate["map_50_95"])
    checks.append(
        {
            "name": "primary_map_50_95_not_lower",
            "passed": candidate_map + tolerance >= baseline_map,
            "baseline": baseline_map,
            "candidate": candidate_map,
            "delta": candidate_map - baseline_map,
        }
    )

    for class_name in protected_classes:
        try:
            baseline_ap = float(baseline["per_class"][class_name]["ap_50_95"])
            candidate_ap = float(candidate["per_class"][class_name]["ap_50_95"])
        except KeyError as exc:
            raise ValueError(f"Không tìm thấy metric lớp {class_name}") from exc
        checks.append(
            {
                "name": f"class_{class_name}_ap_50_95_not_lower",
                "passed": candidate_ap + tolerance >= baseline_ap,
                "baseline": baseline_ap,
                "candidate": candidate_ap,
                "delta": candidate_ap - baseline_ap,
            }
        )

    automatic_pass = all(check["passed"] for check in checks)
    return {
        "automatic_pass": automatic_pass,
        "status": "eligible_for_manual_review" if automatic_pass else "blocked",
        "checks": checks,
        "manual_review_required": True,
        "test_set_used": False,
    }


def main() -> int:
    _configure_console_encoding()
    args = parse_args()
    protected_classes = args.protected_class or ["NoHelmet"]
    baseline_best = best_validation_epoch(_load_json(args.baseline_history))
    candidate_best = best_validation_epoch(_load_json(args.candidate_history))
    baseline_validation = semantic_coco_fingerprint(args.baseline_val)
    candidate_validation = semantic_coco_fingerprint(args.candidate_val)
    gate = evaluate_gate(
        baseline_best,
        candidate_best,
        same_validation=baseline_validation["sha256"] == candidate_validation["sha256"],
        protected_classes=protected_classes,
        tolerance=args.tolerance,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "history": str(args.baseline_history),
            "best_validation": baseline_best,
            "validation_fingerprint": baseline_validation,
        },
        "candidate": {
            "history": str(args.candidate_history),
            "best_validation": candidate_best,
            "validation_fingerprint": candidate_validation,
        },
        "policy": {
            "protected_classes": protected_classes,
            "absolute_tolerance": args.tolerance,
            "test_set_must_remain_sealed": True,
        },
        "gate": gate,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Deployment gate: {gate['status']}")
    for check in gate["checks"]:
        print(f"- {'PASS' if check['passed'] else 'FAIL'}: {check['name']}")
    print(f"Đã lưu: {output_path}")
    return 1 if args.require_pass and not gate["automatic_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
