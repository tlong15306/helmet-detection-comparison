"""Kiểm thử tạo task pending từ annotation COCO tối thiểu."""

from __future__ import annotations

import json

from tools.create_role_dev_tasks import create_tasks


def test_create_tasks_keeps_roles_pending_and_tags_small_head(tmp_path) -> None:
    annotations = {
        "categories": [
            {"id": 1, "name": "BikeWithRider"},
            {"id": 2, "name": "NoHelmet"},
            {"id": 3, "name": "Helmet"},
        ],
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100]},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [40, 10, 8, 8]},
        ],
    }
    path = tmp_path / "val.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    payload, summary = create_tasks(path, "data/raw/edgevision/images", max_tasks=1)
    task = payload["tasks"][0]
    assert task["review"]["status"] == "pending"
    assert task["review"]["driver_head_annotation_id"] is None
    assert task["review"]["head_roles"] == {"11": None}
    assert "small_head" in task["difficulty_tags"]
    assert summary["selection"]["pending"] == 1
