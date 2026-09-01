import torch

from src.threshold_selection import (
    metrics_at_threshold,
    parse_thresholds,
    select_best_threshold,
    select_thresholds_per_class,
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
