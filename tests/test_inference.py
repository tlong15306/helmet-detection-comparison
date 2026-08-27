"""Kiểm thử pipeline suy luận không cần checkpoint hoặc GPU."""

from __future__ import annotations

import torch
import pytest
from PIL import Image

from src.infer import (
    draw_detections,
    filter_predictions,
    image_to_tensor,
    prediction_records,
    summarize_detections,
)


def sample_prediction() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[1.0, 2.0, 20.0, 30.0], [5.0, 6.0, 25.0, 35.0]]),
        "labels": torch.tensor([2, 3]),
        "scores": torch.tensor([0.85, 0.59]),
    }


def test_filter_predictions_is_inclusive_and_keeps_alignment() -> None:
    filtered = filter_predictions(sample_prediction(), 0.85)
    assert filtered["boxes"].tolist() == [[1.0, 2.0, 20.0, 30.0]]
    assert filtered["labels"].tolist() == [2]
    assert filtered["scores"].tolist() == pytest.approx([0.85])


def test_filter_predictions_handles_empty_result() -> None:
    filtered = filter_predictions(sample_prediction(), 0.9)
    assert filtered["boxes"].shape == (0, 4)
    assert filtered["labels"].shape == (0,)
    assert filtered["scores"].shape == (0,)


def test_filter_predictions_rejects_invalid_threshold() -> None:
    for threshold in (-0.01, 1.01):
        try:
            filter_predictions(sample_prediction(), threshold)
        except ValueError as error:
            assert "[0, 1]" in str(error)
        else:
            raise AssertionError("Threshold ngoài [0, 1] phải bị từ chối")


def test_image_to_tensor_normalizes_grayscale_and_rgba_to_rgb() -> None:
    for mode in ("L", "RGBA"):
        image = Image.new(mode, (12, 8))
        tensor = image_to_tensor(image)
        assert tensor.shape == (3, 8, 12)
        assert tensor.dtype == torch.float32
        assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_summary_records_and_drawing_use_class_mapping() -> None:
    classes = {1: "BikeWithRider", 2: "NoHelmet", 3: "Helmet"}
    prediction = filter_predictions(sample_prediction(), 0.5)
    assert summarize_detections(prediction, classes) == {
        "BikeWithRider": 0,
        "NoHelmet": 1,
        "Helmet": 1,
    }
    records = prediction_records(prediction, classes)
    assert records[0]["class_name"] == "NoHelmet"
    assert records[1]["class_name"] == "Helmet"
    rendered = draw_detections(Image.new("RGB", (40, 40), "white"), prediction, classes)
    assert rendered.mode == "RGB"
    assert rendered.size == (40, 40)
