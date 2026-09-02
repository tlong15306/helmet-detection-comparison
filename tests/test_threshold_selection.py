import torch

from src.threshold_selection import (
    metrics_at_threshold,
    parse_thresholds,
    select_best_threshold,
    select_thresholds_per_class,
    update_demo_config,
)


def test_parse_thresholds_sorts_and_deduplicates():
    assert parse_thresholds("0.5,0.1,0.5") == (0.1, 0.5)


def test_threshold_selection_uses_nohelmet_f1_then_recall():
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([2, 2]),
            "scores": torch.tensor([0.8, 0.4]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([2])}]
    class_names = {2: "NoHelmet"}
    candidates = []
    for threshold in (0.3, 0.7):
        candidate = metrics_at_threshold(
            predictions,
            targets,
            class_names,
            confidence_threshold=threshold,
            iou_threshold=0.5,
        )
        candidate["confidence_threshold"] = threshold
        candidates.append(candidate)
    selected = select_best_threshold(candidates, "NoHelmet")
    assert selected["confidence_threshold"] == 0.7
    assert selected["per_class"]["NoHelmet"]["f1"] == 1.0


def test_per_class_selection_keeps_a_threshold_for_every_detector_class():
    candidates = [
        {
            "confidence_threshold": 0.4,
            "per_class": {
                "BikeWithRider": {"f1": 0.8, "recall": 0.8, "precision": 0.8},
                "NoHelmet": {"f1": 0.6, "recall": 0.7, "precision": 0.5},
                "Helmet": {"f1": 0.7, "recall": 0.7, "precision": 0.7},
            },
        },
        {
            "confidence_threshold": 0.7,
            "per_class": {
                "BikeWithRider": {"f1": 0.7, "recall": 0.8, "precision": 0.6},
                "NoHelmet": {"f1": 0.8, "recall": 0.8, "precision": 0.8},
                "Helmet": {"f1": 0.7, "recall": 0.8, "precision": 0.6},
            },
        },
    ]
    selected = select_thresholds_per_class(
        candidates, {1: "BikeWithRider", 2: "NoHelmet", 3: "Helmet"}
    )
    assert selected["BikeWithRider"]["confidence_threshold"] == 0.4
    assert selected["NoHelmet"]["confidence_threshold"] == 0.7
    assert selected["Helmet"]["confidence_threshold"] == 0.7


def test_demo_config_can_store_thresholds_under_an_explicit_checkpoint_key(tmp_path):
    path = tmp_path / "thresholds.yaml"
    result = {
        "model": {"name": "fasterrcnn_resnet50_fpn_v2"},
        "selected_thresholds": {
            "BikeWithRider": {"confidence_threshold": 0.8},
            "NoHelmet": {"confidence_threshold": 0.7},
            "Helmet": {"confidence_threshold": 0.65},
        },
        "selection_policy": {"iou_threshold": 0.5},
        "checkpoint": {"sha256": "a" * 64},
    }
    update_demo_config(path, result, model_key="fasterrcnn_resnet50_fpn_v2_final_combined")
    content = path.read_text(encoding="utf-8")
    assert "fasterrcnn_resnet50_fpn_v2_final_combined" in content
