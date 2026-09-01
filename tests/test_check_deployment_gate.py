import json

from tools.check_deployment_gate import (
    best_validation_epoch,
    evaluate_gate,
    semantic_coco_fingerprint,
)


def _metrics(epoch, score, nohelmet):
    return {
        "epoch": epoch,
        "validation": {
            "map_50_95": score,
            "map_50": score + 0.1,
            "per_class": {"NoHelmet": {"ap_50_95": nohelmet, "ap_50": None}},
        },
    }


def test_best_validation_epoch_uses_primary_metric():
    assert best_validation_epoch([_metrics(1, 0.5, 0.4), _metrics(2, 0.6, 0.3)])["epoch"] == 2


def test_gate_blocks_lower_protected_class_even_when_map_improves():
    baseline = best_validation_epoch([_metrics(1, 0.5, 0.5)])
    candidate = best_validation_epoch([_metrics(1, 0.6, 0.4)])
    result = evaluate_gate(
        baseline,
        candidate,
        same_validation=True,
        protected_classes=["NoHelmet"],
    )
    assert result["status"] == "blocked"
    assert result["automatic_pass"] is False


def test_semantic_fingerprint_ignores_ids_and_parent_paths(tmp_path):
    first = {
        "images": [{"id": 1, "file_name": "one/a.jpg", "width": 10, "height": 20}],
        "categories": [{"id": 2, "name": "NoHelmet"}],
        "annotations": [{"id": 3, "image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4]}],
    }
    second = {
        "images": [{"id": 9, "file_name": "other/a.jpg", "width": 10, "height": 20}],
        "categories": [{"id": 7, "name": "NoHelmet"}],
        "annotations": [{"id": 8, "image_id": 9, "category_id": 7, "bbox": [1, 2, 3, 4]}],
    }
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert semantic_coco_fingerprint(first_path)["sha256"] == semantic_coco_fingerprint(second_path)["sha256"]
