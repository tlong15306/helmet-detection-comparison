"""Kiểm tra schema nhãn vai trò thủ công cho tầng liên kết người–xe."""

from __future__ import annotations

from typing import Any, Mapping


VALID_ROLES = {"driver", "passenger", "unknown"}
VALID_REVIEW_STATUSES = {"pending", "reviewed", "needs_second_review"}


def validate_role_tasks(payload: Mapping[str, Any]) -> dict[str, int]:
    """Kiểm tra task role_dev mà không gán nhãn thay cho người review.

    Một task ``reviewed`` phải chỉ định chính xác một đầu tài xế hoặc ghi rõ
    rằng vai trò không thể xác định. Task ``pending`` phải giữ mọi nhãn là
    ``None`` để tránh vô tình xem heuristic là ground truth.
    """
    if payload.get("schema_version") != "role_association_tasks_v1":
        raise ValueError("schema_version không phải role_association_tasks_v1")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("tasks phải là danh sách")

    task_ids: set[str] = set()
    image_ids: set[int] = set()
    counts = {"tasks": 0, "pending": 0, "reviewed": 0, "needs_second_review": 0}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("Mỗi task phải là mapping")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id phải là chuỗi không rỗng")
        if task_id in task_ids:
            raise ValueError(f"task_id bị trùng: {task_id}")
        task_ids.add(task_id)
        image_id = task.get("image_id")
        if not isinstance(image_id, int):
            raise ValueError(f"{task_id}: image_id phải là số nguyên")
        image_ids.add(image_id)
        heads = task.get("heads")
        if not isinstance(heads, list) or not heads:
            raise ValueError(f"{task_id}: cần ít nhất một head để review")
        head_ids: set[int] = set()
        for head in heads:
            if not isinstance(head, Mapping) or not isinstance(head.get("annotation_id"), int):
                raise ValueError(f"{task_id}: head thiếu annotation_id")
            annotation_id = int(head["annotation_id"])
            if annotation_id in head_ids:
                raise ValueError(f"{task_id}: annotation đầu bị trùng")
            head_ids.add(annotation_id)
            if head.get("helmet_status") not in {"helmet", "no_helmet"}:
                raise ValueError(f"{task_id}: helmet_status không hợp lệ")

        review = task.get("review")
        if not isinstance(review, Mapping):
            raise ValueError(f"{task_id}: thiếu review")
        status = review.get("status")
        if status not in VALID_REVIEW_STATUSES:
            raise ValueError(f"{task_id}: review.status không hợp lệ")
        roles = review.get("head_roles")
        if not isinstance(roles, Mapping):
            raise ValueError(f"{task_id}: review.head_roles phải là mapping")
        normalized_roles = {int(key): value for key, value in roles.items()}
        if set(normalized_roles) - head_ids:
            raise ValueError(f"{task_id}: có role cho head không thuộc task")
        if any(role not in VALID_ROLES and role is not None for role in normalized_roles.values()):
            raise ValueError(f"{task_id}: role không hợp lệ")

        driver_head_id = review.get("driver_head_annotation_id")
        if status == "pending":
            if driver_head_id is not None or any(role is not None for role in normalized_roles.values()):
                raise ValueError(f"{task_id}: task pending không được có nhãn vai trò")
        else:
            if not isinstance(review.get("reviewer"), str) or not str(review["reviewer"]).strip():
                raise ValueError(f"{task_id}: task reviewed cần reviewer")
            if set(normalized_roles) != head_ids or any(role is None for role in normalized_roles.values()):
                raise ValueError(f"{task_id}: task reviewed cần role cho mọi head")
            if driver_head_id is not None:
                if not isinstance(driver_head_id, int) or driver_head_id not in head_ids:
                    raise ValueError(f"{task_id}: driver_head_annotation_id không thuộc task")
                if normalized_roles.get(driver_head_id) != "driver":
                    raise ValueError(f"{task_id}: head tài xế phải có role=driver")
                if sum(role == "driver" for role in normalized_roles.values()) != 1:
                    raise ValueError(f"{task_id}: chỉ được có một head tài xế")
            elif any(role == "driver" for role in normalized_roles.values()):
                raise ValueError(f"{task_id}: role=driver cần driver_head_annotation_id")

        counts["tasks"] += 1
        counts[str(status)] += 1
    counts["images"] = len(image_ids)
    return counts
