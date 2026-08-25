"""Kiểm thử việc từ chối so sánh hai JSON đánh giá không cùng giao thức."""

from __future__ import annotations

import json

import pytest

from src.compare_models import compare_results


def make_result(model_name: str, annotation_sha256: str, map_50: float) -> dict:
    return {
        "schema_version": "evaluation-result-1.0",
        "model": {"name": model_name},
        "evaluation_protocol": {
            "version": "1.0",
            "split": "test",
            "annotation_sha256": annotation_sha256,
            "class_names": {"1": "BikeWithRider", "2": "NoHelmet", "3": "Helmet"},
            "map": "COCO mAP@[IoU=0.50:0.95, step=0.05] via TorchMetrics/pycocotools",
            "map_backend": "pycocotools",
            "precision_recall": {
                "matching": "greedy one-to-one, score descending, same class",
                "iou_threshold": 0.5,
                "confidence_threshold": None,
            },
        },
        "metrics": {
            "map_50_95": 0.4,
            "map_50": map_50,
            "map_75": 0.3,
            "mar_100": 0.5,
            "per_class": {
                "NoHelmet": {"precision": 0.7, "recall": 0.6, "ap_50_95": 0.35}
            },
        },
    }


def write_json(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")


def test_compare_results_derives_deltas_only_from_matching_protocol(tmp_path):
    faster_path = tmp_path / "faster.json"
    retina_path = tmp_path / "retina.json"
    write_json(faster_path, make_result("fasterrcnn_resnet50_fpn_v2", "same", 0.5))
    write_json(retina_path, make_result("retinanet_resnet50_fpn_v2", "same", 0.7))

    comparison = compare_results(faster_path, retina_path)

    assert comparison["comparison_status"] == "comparable"
    assert comparison["metrics"]["map_50"]["retinanet_minus_faster_rcnn"] == pytest.approx(0.2)
    assert comparison["metrics"]["no_helmet_precision"]["faster_rcnn"] == 0.7


def test_compare_results_refuses_different_test_annotation(tmp_path):
    faster_path = tmp_path / "faster.json"
    retina_path = tmp_path / "retina.json"
    write_json(faster_path, make_result("fasterrcnn_resnet50_fpn_v2", "first", 0.5))
    write_json(retina_path, make_result("retinanet_resnet50_fpn_v2", "second", 0.7))

    with pytest.raises(ValueError, match="giao thức đánh giá khác nhau"):
        compare_results(faster_path, retina_path)
