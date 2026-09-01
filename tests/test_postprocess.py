"""Kiểm thử hậu xử lý demo; không tác động metric đánh giá chính thức."""

from __future__ import annotations

import torch

from src.postprocess import postprocess_detections, resolve_head_label_conflicts
from src.rider_association import AssociationConfig, RoleDecisionConfig


CLASSES = {1: "BikeWithRider", 2: "NoHelmet", 3: "Helmet"}


def _record(class_id: int, class_name: str, score: float, box: list[float], index: int) -> dict:
    return {
        "detection_id": f"detection_{index}",
        "class_id": class_id,
        "class_name": class_name,
        "confidence": score,
        "box": box,
    }


def test_close_opposite_head_labels_become_unknown() -> None:
    records = [
        _record(2, "NoHelmet", 0.76, [10, 10, 30, 30], 1),
        _record(3, "Helmet", 0.73, [11, 10, 31, 30], 2),
    ]
    kept, unknown_cases = resolve_head_label_conflicts(
        records, iou_threshold=0.70, confidence_margin=0.10
    )
    assert kept == []
    assert unknown_cases[0]["status"] == "unknown"
    assert unknown_cases[0]["reason"] == "helmet_no_helmet_scores_too_close"


def test_clear_opposite_head_label_keeps_the_stronger_class() -> None:
    records = [
        _record(2, "NoHelmet", 0.91, [10, 10, 30, 30], 1),
        _record(3, "Helmet", 0.61, [11, 10, 31, 30], 2),
    ]
    kept, unknown_cases = resolve_head_label_conflicts(
        records, iou_threshold=0.70, confidence_margin=0.10
    )
    assert [item["class_name"] for item in kept] == ["NoHelmet"]
    assert unknown_cases == []


def test_alert_requires_validated_driver_association() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0], [40.0, 10.0, 60.0, 30.0]]),
        "labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.95, 0.95]),
    }
    role_config = RoleDecisionConfig(
        enabled=True,
        single_head_rule=True,
        observed_precision=0.96,
        observed_recall=0.65,
        observed_support=60,
    )
    processed = postprocess_detections(
        prediction,
        CLASSES,
        AssociationConfig(),
        role_config,
        head_conflict_iou_threshold=0.70,
        head_confidence_margin=0.10,
    )
    assert processed["alerts"][0]["status"] == "driver_no_helmet"
    assert processed["alerts"][0]["head_detection_id"] == "detection_2"


def test_multihead_group_stays_without_alert() -> None:
    prediction = {
        "boxes": torch.tensor(
            [[0.0, 0.0, 100.0, 100.0], [20.0, 10.0, 40.0, 30.0], [60.0, 10.0, 80.0, 30.0]]
        ),
        "labels": torch.tensor([1, 2, 3]),
        "scores": torch.tensor([0.95, 0.95, 0.95]),
    }
    role_config = RoleDecisionConfig(
        enabled=True,
        single_head_rule=True,
        observed_precision=0.96,
        observed_recall=0.65,
        observed_support=60,
    )
    processed = postprocess_detections(
        prediction,
        CLASSES,
        AssociationConfig(),
        role_config,
        head_conflict_iou_threshold=0.70,
        head_confidence_margin=0.10,
    )
    assert processed["alerts"] == []
    assert processed["rider_analysis"]["summary"]["unknown_role_groups"] == 1
    assert processed["display_annotations"] == {}
