"""Kiểm tra giao thức và tổng hợp hai JSON đánh giá mà không tạo số liệu mới."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .utils import PROJECT_ROOT, resolve_project_path


REQUIRED_PROTOCOL_KEYS = (
    "version",
    "split",
    "annotation_sha256",
    "class_names",
    "map",
    "map_backend",
    "precision_recall",
)


def project_relative(path: Path) -> str:
    """Đưa đường dẫn về dạng ổn định tương đối với root dự án nếu có thể."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_result(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Đọc và kiểm tra schema tối thiểu của một tệp kết quả evaluation."""
    result_path = resolve_project_path(path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy JSON kết quả: {result_path}")
    with result_path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"JSON kết quả phải là object: {result_path}")
    if result.get("schema_version") != "evaluation-result-1.0":
        raise ValueError(
            f"Schema không hỗ trợ trong {result_path}. Cần evaluation-result-1.0."
        )
    for key in ("model", "evaluation_protocol", "metrics"):
        if not isinstance(result.get(key), Mapping):
            raise ValueError(f"Thiếu hoặc sai trường {key} trong {result_path}")
    return result_path, result


def protocol_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Liệt kê mọi điều kiện đánh giá khác nhau, thay vì âm thầm so sánh."""
    differences: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_PROTOCOL_KEYS:
        if key not in left or key not in right:
            differences[key] = {"left": left.get(key), "right": right.get(key)}
        elif left[key] != right[key]:
            differences[key] = {"left": left[key], "right": right[key]}
    return differences


def metric_value(result: Mapping[str, Any], key: str) -> float | None:
    """Đọc metric số nếu có; giữ null khi một metric không xác định."""
    value = result["metrics"].get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Metric {key} không phải số: {value!r}")
    return float(value)


def no_helmet_metric(result: Mapping[str, Any], key: str) -> float | None:
    """Đọc Precision/Recall riêng lớp NoHelmet từ cùng JSON gốc."""
    per_class = result["metrics"].get("per_class", {})
    if not isinstance(per_class, Mapping) or "NoHelmet" not in per_class:
        return None
    value = per_class["NoHelmet"].get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"NoHelmet.{key} không phải số: {value!r}")
    return float(value)


def value_pair(left: float | None, right: float | None) -> dict[str, float | None]:
    """Ghi các số đo gốc và chênh lệch RetinaNet - Faster R-CNN nếu xác định."""
    return {
        "faster_rcnn": left,
        "retinanet": right,
        "retinanet_minus_faster_rcnn": None if left is None or right is None else right - left,
    }


def sha256(path: Path) -> str:
    """Fingerprint tệp đầu vào để bảng so sánh truy vết được nguồn số liệu."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_results(
    faster_rcnn_path: str | Path,
    retinanet_path: str | Path,
    allow_protocol_mismatch: bool = False,
) -> dict[str, Any]:
    """Tạo đối tượng comparison có thể JSON hoá từ hai kết quả test/validation."""
    faster_path, faster = load_result(faster_rcnn_path)
    retina_path, retina = load_result(retinanet_path)
    differences = protocol_differences(
        faster["evaluation_protocol"], retina["evaluation_protocol"]
    )
    if differences and not allow_protocol_mismatch:
        difference_names = ", ".join(differences)
        raise ValueError(
            "Không thể so sánh công bằng vì giao thức đánh giá khác nhau ở: "
            f"{difference_names}. Sửa cấu hình/kết quả trước; chỉ dùng "
            "--allow-protocol-mismatch để xuất dữ liệu chẩn đoán."
        )

    return {
        "schema_version": "model-comparison-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_status": "comparable" if not differences else "protocol_mismatch",
        "evaluation_protocol": faster["evaluation_protocol"],
        "protocol_differences": differences,
        "inputs": {
            "faster_rcnn": {
                "path": project_relative(faster_path),
                "sha256": sha256(faster_path),
                "model": faster["model"],
            },
            "retinanet": {
                "path": project_relative(retina_path),
                "sha256": sha256(retina_path),
                "model": retina["model"],
            },
        },
        "metrics": {
            "map_50_95": value_pair(metric_value(faster, "map_50_95"), metric_value(retina, "map_50_95")),
            "map_50": value_pair(metric_value(faster, "map_50"), metric_value(retina, "map_50")),
            "map_75": value_pair(metric_value(faster, "map_75"), metric_value(retina, "map_75")),
            "mar_100": value_pair(metric_value(faster, "mar_100"), metric_value(retina, "mar_100")),
            "no_helmet_precision": value_pair(
                no_helmet_metric(faster, "precision"), no_helmet_metric(retina, "precision")
            ),
            "no_helmet_recall": value_pair(
                no_helmet_metric(faster, "recall"), no_helmet_metric(retina, "recall")
            ),
            "no_helmet_ap_50_95": value_pair(
                no_helmet_metric(faster, "ap_50_95"), no_helmet_metric(retina, "ap_50_95")
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two standard evaluator JSON files")
    parser.add_argument("--faster-rcnn", required=True, help="JSON do src.evaluate tạo")
    parser.add_argument("--retinanet", required=True, help="JSON do src.evaluate tạo")
    parser.add_argument("--output", required=True, help="Nơi lưu JSON bảng so sánh")
    parser.add_argument(
        "--allow-protocol-mismatch",
        action="store_true",
        help="Chỉ xuất chẩn đoán khi hai lần đánh giá không cùng giao thức.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_results(
        args.faster_rcnn,
        args.retinanet,
        allow_protocol_mismatch=args.allow_protocol_mismatch,
    )
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Đã ghi so sánh: {output_path} ({comparison['comparison_status']})")


if __name__ == "__main__":
    main()
