"""Kiểm thử tầng liên kết đầu–xe trước khi có nhãn vai trò."""

from __future__ import annotations

import pytest

from src.rider_association import AssociationConfig, analyze_rider_roles


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
