"""Kiểm thử tầng liên kết đầu–xe trước khi có nhãn vai trò."""

from __future__ import annotations

import pytest

from src.rider_association import AssociationConfig, RoleDecisionConfig, analyze_rider_roles


def detection(detection_id: str, class_name: str, box: list[float]) -> dict[str, object]:
    return {"detection_id": detection_id, "class_name": class_name, "box": box, "confidence": 0.9}


def test_one_clear_head_creates_candidate_only_not_confirmed_alert() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("head_1", "NoHelmet", [40, 10, 60, 30]),
        ]
    )
    group = result["rider_groups"][0]
    assert group["heads"][0]["head_detection_id"] == "head_1"
    assert group["driver"]["role"] == "driver_candidate"
    assert group["driver"]["helmet_status"] == "no_helmet"
    assert result["summary"]["driver_candidate_no_helmet"] == 1
    assert result["summary"]["confirmed_driver_no_helmet"] == 0


def test_two_heads_are_not_forced_into_driver_and_passenger_roles() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("head_1", "Helmet", [15, 10, 35, 30]),
            detection("head_2", "NoHelmet", [65, 10, 85, 30]),
        ]
    )
    group = result["rider_groups"][0]
    assert len(group["heads"]) == 2
    assert group["driver"] is None
    assert result["summary"]["driver_candidates"] == 0


def test_head_matching_two_near_equal_vehicles_is_ambiguous() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("bike_2", "BikeWithRider", [5, 0, 105, 100]),
            detection("head_1", "Helmet", [42, 10, 62, 30]),
        ]
    )
    assert result["summary"]["associated_heads"] == 0
    assert result["summary"]["ambiguous_heads"] == 1
    assert result["ambiguous_heads"][0]["head_detection_id"] == "head_1"


def test_head_outside_vehicle_is_kept_unassigned() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("head_1", "NoHelmet", [120, 10, 140, 30]),
        ]
    )
    assert result["summary"]["unassigned_heads"] == 1
    assert result["unassigned_heads"][0]["reason"] == "no_vehicle_match"


def test_duplicate_detection_id_and_invalid_config_are_rejected() -> None:
    with pytest.raises(ValueError, match="bị trùng"):
        analyze_rider_roles(
            [
                detection("same", "BikeWithRider", [0, 0, 100, 100]),
                detection("same", "Helmet", [10, 10, 30, 30]),
            ]
        )
    with pytest.raises(ValueError, match="min_head_coverage"):
        AssociationConfig(min_head_coverage=1.1)


def validated_role_config(**overrides: object) -> RoleDecisionConfig:
    values = {
        "enabled": True,
        "single_head_rule": True,
        "minimum_precision": 0.95,
        "minimum_support": 50,
        "observed_precision": 0.962264,
        "observed_recall": 0.653846,
        "observed_support": 53,
        "source_split": "validation",
        "source_tasks": "role_dev.json",
        "source_sha256": "abc",
    }
    values.update(overrides)
    return RoleDecisionConfig(**values)  # type: ignore[arg-type]


def test_validated_single_head_rule_creates_rule_based_alert() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("head_1", "NoHelmet", [40, 10, 60, 30]),
        ],
        role_config=validated_role_config(),
    )
    driver = result["rider_groups"][0]["driver"]
    assert driver["role"] == "driver"
    assert driver["status"] == "rule_based"
    assert driver["validation_evidence"]["precision"] == pytest.approx(0.962264)
    assert result["summary"]["rule_based_drivers"] == 1
    assert result["summary"]["driver_no_helmet_alerts"] == 1
    assert result["summary"]["confirmed_driver_no_helmet"] == 0


def test_role_rule_falls_back_when_precision_is_below_requirement() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("head_1", "Helmet", [40, 10, 60, 30]),
        ],
        role_config=validated_role_config(observed_precision=0.94),
    )
    assert result["rider_groups"][0]["driver"]["status"] == "candidate_only"
    assert result["summary"]["rule_based_drivers"] == 0


def test_validated_rule_still_abstains_for_multihead_group() -> None:
    result = analyze_rider_roles(
        [
            detection("bike_1", "BikeWithRider", [0, 0, 100, 100]),
            detection("head_1", "Helmet", [15, 10, 35, 30]),
            detection("head_2", "NoHelmet", [65, 10, 85, 30]),
        ],
        role_config=validated_role_config(),
    )
    assert result["rider_groups"][0]["driver"] is None
    assert result["summary"]["unknown_role_groups"] == 1
    assert result["summary"]["driver_no_helmet_alerts"] == 0
