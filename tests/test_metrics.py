"""Kiểm thử công thức và giao thức metric chung."""

import torch
import pytest

from src.metrics import DetectionEvaluator, precision_recall


def test_precision_recall():
    precision, recall = precision_recall(tp=8, fp=2, fn=4)
    assert precision == 0.8
    assert recall == 8 / 12


def test_detection_evaluator_matches_each_ground_truth_once():
    evaluator = DetectionEvaluator({1: "BikeWithRider", 2: "NoHelmet"})
    predictions = [
        {
            "boxes": torch.tensor(
                [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
            ),
            "labels": torch.tensor([2, 2]),
            "scores": torch.tensor([0.9, 0.8]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([2]),
        }
    ]

    evaluator.update(predictions, targets)
    result = evaluator.compute()

    no_helmet = result["per_class"]["NoHelmet"]
    assert no_helmet["tp"] == 1
    assert no_helmet["fp"] == 1
    assert no_helmet["fn"] == 0
    assert no_helmet["precision"] == 0.5
    assert no_helmet["recall"] == 1.0
    assert result["map_50"] == 1.0


def test_detection_evaluator_respects_confidence_threshold():
    evaluator = DetectionEvaluator(
        {2: "NoHelmet"},
        iou_threshold=0.5,
        confidence_threshold=0.75,
    )
    evaluator.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "labels": torch.tensor([2]),
                "scores": torch.tensor([0.5]),
            }
        ],
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "labels": torch.tensor([2]),
            }
        ],
    )

    no_helmet = evaluator.compute()["per_class"]["NoHelmet"]
    assert no_helmet["tp"] == 0
    assert no_helmet["fp"] == 0
    assert no_helmet["fn"] == 1


def test_detection_evaluator_rejects_annotation_label_outside_protocol():
    evaluator = DetectionEvaluator({2: "NoHelmet"})
    with pytest.raises(ValueError, match="ngoài class_names"):
        evaluator.update(
            [{"boxes": [], "labels": [], "scores": []}],
            [{"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [3]}],
        )
