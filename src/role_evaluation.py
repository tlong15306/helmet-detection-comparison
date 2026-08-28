"""Đánh giá chẩn đoán cho gợi ý vai trò trên tập role_dev.

Module này chỉ đối chiếu gợi ý ``driver_candidate`` của baseline hình học với
nhãn do người review nhập. Đây là công cụ phát triển quy tắc: không phải thước
đo chất lượng detector, không thay thế kiểm tra chéo và không được dùng để kết
luận cuối cùng trên role_test.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from src.role_annotations import VALID_REVIEW_STATUSES, validate_role_tasks


DEFAULT_INCLUDED_STATUSES = frozenset({"reviewed", "needs_second_review"})


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_role_candidate_baseline(
    payload: Mapping[str, Any],
    *,
    included_statuses: Iterable[str] = DEFAULT_INCLUDED_STATUSES,
) -> dict[str, Any]:
    """Đối chiếu candidate một-đầu với nhãn review hiện có.

    ``proposed_driver_head_annotation_id`` là dự đoán dương duy nhất của
    baseline. Giá trị ``None`` là abstention có chủ đích, không phải dự đoán
    rằng ảnh không có tài xế. Các task pending luôn bị loại để không nhầm
    heuristic với ground truth.
    """
    validation_counts = validate_role_tasks(payload)
    statuses = set(included_statuses)
    if not statuses:
        raise ValueError("Cần chọn ít nhất một review status để đánh giá")
    invalid_statuses = statuses - VALID_REVIEW_STATUSES
    if invalid_statuses:
        raise ValueError(f"review status không hợp lệ: {sorted(invalid_statuses)}")
    if "pending" in statuses:
        raise ValueError("Không được dùng task pending làm ground truth")

    used: list[Mapping[str, Any]] = []
    excluded_by_status: Counter[str] = Counter()
    for task in payload["tasks"]:
        status = str(task["review"]["status"])
        if status in statuses:
            used.append(task)
        else:
            excluded_by_status[status] += 1

    proposals = 0
    correct_proposals = 0
    actual_driver_tasks = 0
    actual_driver_with_proposal = 0
    abstentions = 0
    task_outcomes: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    multihead_rule_counts: Counter[str] = Counter()
    multihead_tasks_with_driver = 0

    for task in used:
        review = task["review"]
        actual_driver = review["driver_head_annotation_id"]
        proposal = task.get("proposed_driver_head_annotation_id")
        head_roles = review["head_roles"]
        role_counts.update(str(role) for role in head_roles.values())

        if actual_driver is not None:
            actual_driver_tasks += 1
        if actual_driver is not None and len(task["heads"]) > 1:
            multihead_tasks_with_driver += 1
            heads = task["heads"]
            strategies = {
                "leftmost": min(heads, key=lambda head: (head["box_xyxy"][0] + head["box_xyxy"][2], head["annotation_id"])),
                "rightmost": max(heads, key=lambda head: (head["box_xyxy"][0] + head["box_xyxy"][2], -head["annotation_id"])),
                "highest": min(heads, key=lambda head: (head["box_xyxy"][1] + head["box_xyxy"][3], head["annotation_id"])),
                "lowest": max(heads, key=lambda head: (head["box_xyxy"][1] + head["box_xyxy"][3], -head["annotation_id"])),
                "largest_box": max(
                    heads,
                    key=lambda head: (
                        (head["box_xyxy"][2] - head["box_xyxy"][0])
                        * (head["box_xyxy"][3] - head["box_xyxy"][1]),
                        -head["annotation_id"],
                    ),
                ),
            }
            for name, selected_head in strategies.items():
                if selected_head["annotation_id"] == actual_driver:
                    multihead_rule_counts[name] += 1
        if proposal is None:
            abstentions += 1
            task_outcomes["abstained"] += 1
            continue

        if not isinstance(proposal, int):
            raise ValueError(f"{task['task_id']}: proposed_driver_head_annotation_id không hợp lệ")
        proposals += 1
        if proposal == actual_driver:
            correct_proposals += 1
            actual_driver_with_proposal += 1
            task_outcomes["correct_candidate"] += 1
        elif actual_driver is None:
            task_outcomes["candidate_when_role_unknown"] += 1
        else:
            task_outcomes["wrong_candidate"] += 1

    # Với task không có proposal, không thể có candidate đúng. Đây là chủ đích
    # an toàn của baseline khi có nhiều đầu hoặc ghép hình học chưa rõ ràng.
    team_confirmation = payload.get("team_confirmation")
    has_team_confirmation = (
        isinstance(team_confirmation, Mapping)
        and team_confirmation.get("status") == "confirmed_by_team"
        and validation_counts["reviewed"] == validation_counts["tasks"]
    )
    has_provisional_reviews = validation_counts["needs_second_review"] > 0
    limitations = [
        "Candidate precision/recall chỉ đo gợi ý vai trò trên role_dev; không phải mAP hay metric của Faster R-CNN/RetinaNet.",
        "Không dùng kết quả này để huấn luyện, chọn cấu hình cuối hoặc kết luận trên role_test.",
        "Abstention là hành vi an toàn của baseline, cần được diễn giải cùng độ bao phủ thay vì xem như nhãn âm.",
    ]
    if has_provisional_reviews:
        limitations.insert(1, "Task needs_second_review mới là nhãn vòng một, chưa phải ground truth đã thống nhất.")
    if has_team_confirmation:
        limitations.insert(1, "Nhóm đã xác nhận tập role_dev nhưng không có log review vòng hai riêng cho từng task.")

    result = {
        "schema_version": "role_candidate_evaluation_v1",
        "purpose": "development_diagnostic_only",
        "ground_truth_status": "team_confirmed_role_dev" if has_team_confirmation else (
            "provisional_first_review" if has_provisional_reviews else "reviewed"
        ),
        "included_statuses": sorted(statuses),
        "task_counts": {
            "available": validation_counts["tasks"],
            "used": len(used),
            "excluded_by_status": dict(sorted(excluded_by_status.items())),
        },
        "human_role_counts": dict(sorted(role_counts.items())),
        "candidate_diagnostic": {
            "candidate_proposals": proposals,
            "correct_candidates": correct_proposals,
            "abstentions": abstentions,
            "tasks_with_human_driver": actual_driver_tasks,
            "candidate_precision": _rate(correct_proposals, proposals),
            "candidate_recall": _rate(actual_driver_with_proposal, actual_driver_tasks),
            "task_outcomes": dict(sorted(task_outcomes.items())),
        },
        "multihead_simple_rules": {
            "tasks_with_human_driver": multihead_tasks_with_driver,
            "strategies": {
                name: {
                    "correct": multihead_rule_counts[name],
                    "precision_if_forced": _rate(multihead_rule_counts[name], multihead_tasks_with_driver),
                }
                for name in ("leftmost", "rightmost", "highest", "lowest", "largest_box")
            },
            "decision": "abstain_multihead",
        },
        "team_confirmation": team_confirmation if has_team_confirmation else None,
        "limitations": limitations,
    }
    return result
