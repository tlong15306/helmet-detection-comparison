"""Kiểm thử schema role_dev để ngăn nhãn heuristic trở thành ground truth."""

from __future__ import annotations

import pytest

from src.role_annotations import validate_role_tasks


def task(status: str = "pending") -> dict[str, object]:
    return {
        "schema_version": "role_association_tasks_v1",
        "tasks": [
            {
                "task_id": "role_dev_001",
                "image_id": 1,
                "heads": [{"annotation_id": 11, "helmet_status": "no_helmet"}],
                "review": {
                    "status": status,
                    "reviewer": None,
                    "driver_head_annotation_id": None,
                    "head_roles": {"11": None},
                },
            }
        ],
    }


def test_pending_role_task_has_no_invented_labels() -> None:
    assert validate_role_tasks(task()) == {
        "tasks": 1,
        "pending": 1,
        "reviewed": 0,
        "needs_second_review": 0,
        "images": 1,
    }


def test_pending_role_task_rejects_pre_filled_driver_label() -> None:
    payload = task()
    review = payload["tasks"][0]["review"]  # type: ignore[index]
    review["driver_head_annotation_id"] = 11  # type: ignore[index]
    review["head_roles"] = {"11": "driver"}  # type: ignore[index]
    with pytest.raises(ValueError, match="pending"):
        validate_role_tasks(payload)


def test_reviewed_task_requires_one_consistent_driver() -> None:
    payload = task("reviewed")
    review = payload["tasks"][0]["review"]  # type: ignore[index]
    review["reviewer"] = "Long"  # type: ignore[index]
    review["driver_head_annotation_id"] = 11  # type: ignore[index]
    review["head_roles"] = {"11": "driver"}  # type: ignore[index]
    assert validate_role_tasks(payload)["reviewed"] == 1
