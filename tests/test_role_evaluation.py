"""Kiểm thử báo cáo chẩn đoán role_dev."""

from __future__ import annotations

import pytest

from src.role_evaluation import evaluate_role_candidate_baseline


def _task(
    task_id: str,
    *,
    status: str,
    proposal: int | None,
    driver: int | None,
    roles: dict[str, str | None],
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "image_id": int(task_id[-1]),
        "heads": [
            {"annotation_id": int(annotation_id), "helmet_status": "no_helmet"}
            for annotation_id in roles
        ],
        "proposed_driver_head_annotation_id": proposal,
        "review": {
            "status": status,
            "reviewer": "Long" if status != "pending" else None,
            "driver_head_annotation_id": driver,
            "head_roles": roles,
        },
    }


def _payload(tasks: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": "role_association_tasks_v1", "tasks": tasks}


def test_role_candidate_report_excludes_pending_and_reports_abstention() -> None:
    report = evaluate_role_candidate_baseline(
        _payload(
            [
                _task("role_1", status="needs_second_review", proposal=11, driver=11, roles={"11": "driver"}),
                _task("role_2", status="needs_second_review", proposal=21, driver=22, roles={"21": "passenger", "22": "driver"}),
                _task("role_3", status="needs_second_review", proposal=None, driver=31, roles={"31": "driver", "32": "passenger"}),
                _task("role_4", status="pending", proposal=41, driver=None, roles={"41": None}),
            ]
        )
    )
    diagnostic = report["candidate_diagnostic"]
    assert report["ground_truth_status"] == "provisional_first_review"
    assert report["task_counts"] == {"available": 4, "used": 3, "excluded_by_status": {"pending": 1}}
    assert diagnostic["candidate_proposals"] == 2
    assert diagnostic["correct_candidates"] == 1
    assert diagnostic["abstentions"] == 1
    assert diagnostic["candidate_precision"] == 0.5
    assert diagnostic["candidate_recall"] == pytest.approx(1 / 3)


def test_role_candidate_report_rejects_pending_as_ground_truth() -> None:
    payload = _payload([_task("role_1", status="pending", proposal=None, driver=None, roles={"11": None})])
    with pytest.raises(ValueError, match="pending"):
        evaluate_role_candidate_baseline(payload, included_statuses={"pending"})


def test_role_candidate_report_recognizes_team_confirmation() -> None:
    payload = _payload(
        [_task("role_1", status="reviewed", proposal=11, driver=11, roles={"11": "driver"})]
    )
    payload["team_confirmation"] = {
        "status": "confirmed_by_team",
        "reported_by": "Long",
        "note": "Không có log vòng hai theo từng task.",
    }
    report = evaluate_role_candidate_baseline(payload)
    assert report["ground_truth_status"] == "team_confirmed_role_dev"
    assert report["team_confirmation"] == payload["team_confirmation"]
    assert any("không có log review vòng hai" in item for item in report["limitations"])
