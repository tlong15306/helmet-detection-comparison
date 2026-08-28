import torch

from src.threshold_selection import metrics_at_threshold, parse_thresholds, select_best_threshold


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
