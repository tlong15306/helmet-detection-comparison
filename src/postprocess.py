"""Hậu xử lý an toàn cho demo phát hiện mũ bảo hiểm.

Module này chạy *sau* detector và tách biệt với ``src.evaluate``. Do đó nó
không thay đổi metric test chính thức hay checkpoint. Các quyết định mơ hồ
được giữ ở trạng thái ``unknown`` thay vì suy đoán thành vi phạm.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .infer import prediction_records
from .rider_association import (
    AssociationConfig,
    RoleDecisionConfig,
    analyze_rider_roles,
)


HEAD_CLASS_NAMES = {"Helmet", "NoHelmet"}
UNKNOWN_COLOR = (245, 158, 11)
ALERT_COLOR = (220, 38, 38)
CAUTION_COLOR = (245, 158, 11)


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = first_area + second_area - intersection
    return 0.0 if denominator <= 0 else intersection / denominator


def _components(records: Sequence[Mapping[str, Any]], iou_threshold: float) -> list[list[int]]:
    """Nhóm các box đầu đối nghịch có khả năng là cùng một đầu."""
    edges: dict[int, set[int]] = {index: set() for index in range(len(records))}
    for left_index, left in enumerate(records):
        if left["class_name"] not in HEAD_CLASS_NAMES:
            continue
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            if (
                right["class_name"] not in HEAD_CLASS_NAMES
                or right["class_name"] == left["class_name"]
                or _iou(left["box"], right["box"]) < iou_threshold
            ):
                continue
            edges[left_index].add(right_index)
            edges[right_index].add(left_index)

    visited: set[int] = set()
    result: list[list[int]] = []
    for index in range(len(records)):
        if index in visited:
            continue
        stack = [index]
        component: list[int] = []
        visited.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in edges[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        result.append(sorted(component))
    return result


def resolve_head_label_conflicts(
    records: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
    confidence_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loại xung đột Helmet–NoHelmet mà không gộp nhầm các đầu cạnh nhau.

    Chỉ các box *khác lớp* và chồng lấn vượt ngưỡng mới được xét. Khi điểm hai
    nhãn quá gần nhau, cả hai bị loại khỏi kết luận và tạo một ``unknown_case``.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold phải nằm trong [0, 1]")
    if confidence_margin < 0.0:
        raise ValueError("confidence_margin không được âm")

    kept: list[dict[str, Any]] = []
    unknown_cases: list[dict[str, Any]] = []
    for component in _components(records, iou_threshold):
        candidates = [dict(records[index]) for index in component]
        head_candidates = [item for item in candidates if item["class_name"] in HEAD_CLASS_NAMES]
        head_classes = {str(item["class_name"]) for item in head_candidates}
        if len(head_classes) < 2:
            kept.extend(candidates)
            continue

        best_by_class = {
            class_name: max(
                (item for item in head_candidates if item["class_name"] == class_name),
                key=lambda item: (float(item["confidence"]), str(item["detection_id"])),
            )
            for class_name in sorted(head_classes)
        }
        helmet = best_by_class["Helmet"]
        no_helmet = best_by_class["NoHelmet"]
        difference = abs(float(helmet["confidence"]) - float(no_helmet["confidence"]))
        non_heads = [item for item in candidates if item["class_name"] not in HEAD_CLASS_NAMES]
        kept.extend(non_heads)
        if difference <= confidence_margin:
            representative = max(head_candidates, key=lambda item: float(item["confidence"]))
            unknown_cases.append(
                {
                    "kind": "helmet_label_conflict",
                    "status": "unknown",
                    "reason": "helmet_no_helmet_scores_too_close",
                    "box": list(representative["box"]),
                    "candidates": [
                        {
                            "class_name": item["class_name"],
                            "confidence": item["confidence"],
                            "box": list(item["box"]),
                        }
                        for item in sorted(head_candidates, key=lambda item: str(item["detection_id"]))
                    ],
                }
            )
            continue
        winner = helmet if float(helmet["confidence"]) > float(no_helmet["confidence"]) else no_helmet
        kept.append(winner)
    return kept, unknown_cases


def records_to_prediction(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Đưa records đã chọn về schema tensor của detector để dùng các hàm sẵn có."""
    if not records:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,), dtype=torch.float32),
        }
    return {
        "boxes": torch.tensor([item["box"] for item in records], dtype=torch.float32),
        "labels": torch.tensor([item["class_id"] for item in records], dtype=torch.int64),
        "scores": torch.tensor([item["confidence"] for item in records], dtype=torch.float32),
    }


def build_alerts(rider_analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Chỉ tạo cảnh báo khi quy tắc vai trò đã xác nhận tài xế."""
    alerts: list[dict[str, Any]] = []
    for group in rider_analysis.get("rider_groups", []):
        driver = group.get("driver")
        if not isinstance(driver, Mapping) or driver.get("status") != "rule_based":
            continue
        if driver.get("helmet_status") == "no_helmet":
            alerts.append(
                {
                    "status": "driver_no_helmet",
                    "severity": "alert",
                    "group_id": group["group_id"],
                    "bike_detection_id": group["bike_detection_id"],
                    "head_detection_id": driver["head_detection_id"],
                    "message": "Người điều khiển có khả năng không đội mũ bảo hiểm",
                }
            )
    return alerts


def build_display_annotations(rider_analysis: Mapping[str, Any], alerts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Chỉ đánh dấu trực quan cảnh báo đã đủ điều kiện.

    Các detection không tạo cảnh báo vẫn giữ nguyên nhãn mũ gốc của detector.
    ``Không xác định`` được vẽ riêng cho xung đột nhãn mũ trong
    :func:`resolve_head_label_conflicts`.
    """
    annotations: dict[str, dict[str, Any]] = {}
    alert_ids = {str(item["head_detection_id"]) for item in alerts}
    for group in rider_analysis.get("rider_groups", []):
        for head in group.get("heads", []):
            head_id = str(head["head_detection_id"])
            if head_id in alert_ids:
                annotations[head_id] = {"label": "NoHelmet · cảnh báo", "color": ALERT_COLOR}
    return annotations


def postprocess_detections(
    prediction: Mapping[str, torch.Tensor],
    class_names: Mapping[int, str],
    association_config: AssociationConfig,
    role_config: RoleDecisionConfig,
    *,
    head_conflict_iou_threshold: float,
    head_confidence_margin: float,
) -> dict[str, Any]:
    """Giải quyết xung đột, ghép đầu–xe và tạo cảnh báo cho một ảnh/frame."""
    raw_records = prediction_records(prediction, class_names)
    selected_records, unknown_cases = resolve_head_label_conflicts(
        raw_records,
        iou_threshold=head_conflict_iou_threshold,
        confidence_margin=head_confidence_margin,
    )
    processed_prediction = records_to_prediction(selected_records)
    processed_records = prediction_records(processed_prediction, class_names)
    rider_analysis = analyze_rider_roles(processed_records, association_config, role_config)
    alerts = build_alerts(rider_analysis)
    return {
        "raw_detections": raw_records,
        "prediction": processed_prediction,
        "detections": processed_records,
        "unknown_cases": unknown_cases,
        "rider_analysis": rider_analysis,
        "alerts": alerts,
        "display_annotations": build_display_annotations(rider_analysis, alerts),
    }
