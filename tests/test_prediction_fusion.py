"""Kiểm thử quy đổi TTA và hợp nhất prediction."""

import pytest
import torch

from src.prediction_fusion import (
    HorizontalFlipTTADetector,
    fuse_predictions,
    horizontal_flip_prediction,
)


def _prediction(box, score=0.8, label=2):
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([label], dtype=torch.int64),
        "scores": torch.tensor([score], dtype=torch.float32),
    }


def test_horizontal_flip_prediction_returns_original_coordinates() -> None:
    flipped = _prediction([60, 10, 90, 30])
    restored = horizontal_flip_prediction(flipped, image_width=100)
    assert restored["boxes"].tolist() == [[10.0, 10.0, 40.0, 30.0]]


def test_fusion_rewards_cross_source_agreement() -> None:
    first = _prediction([10, 10, 30, 30], score=0.8)
    second = _prediction([11, 10, 31, 30], score=0.6)
    fused = fuse_predictions([first, second], iou_threshold=0.5)
    assert fused["boxes"].shape == (1, 4)
    assert fused["scores"].item() == pytest.approx(0.7)
    assert fused["labels"].item() == 2


def test_single_source_detection_is_retained_but_downweighted() -> None:
    present = _prediction([10, 10, 30, 30], score=0.8)
    absent = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
    }
    fused = fuse_predictions([present, absent], iou_threshold=0.5)
    assert fused["scores"].item() == pytest.approx(0.4)


def test_different_classes_are_never_merged() -> None:
    helmet = _prediction([10, 10, 30, 30], score=0.9, label=3)
    no_helmet = _prediction([10, 10, 30, 30], score=0.9, label=2)
    fused = fuse_predictions([helmet, no_helmet], iou_threshold=0.5)
    assert sorted(fused["labels"].tolist()) == [2, 3]


def test_tta_detector_restores_flipped_boxes_before_fusion() -> None:
    class Detector(torch.nn.Module):
        def forward(self, images):
            return [
                _prediction([10, 10, 30, 30], score=0.8)
                if float(image[0, 0, 0]) == 0.0
                else _prediction([70, 10, 90, 30], score=0.6)
                for image in images
            ]

    image = torch.zeros((3, 50, 100))
    image[:, :, -1] = 1.0
    output = HorizontalFlipTTADetector(Detector())([image])[0]

    assert output["boxes"].shape == (1, 4)
    assert output["scores"].item() == pytest.approx(0.7)
