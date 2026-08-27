"""Kiểm thử các điều kiện an toàn khi evaluate một challenge COCO tùy chọn."""

import pytest

from src.evaluate import evaluate


def test_challenge_split_requires_explicit_annotation_path() -> None:
    with pytest.raises(ValueError, match="annotations_override"):
        evaluate(
            config={},
            split="challenge",
            checkpoint_path="unused.pth",
            output_path="unused.json",
        )


def test_custom_precision_recall_threshold_is_validated_before_loading_files() -> None:
    with pytest.raises(ValueError, match="confidence_threshold_override"):
        evaluate(
            config={},
            split="challenge",
            annotations_override="challenge.json",
            checkpoint_path="unused.pth",
            output_path="unused.json",
            confidence_threshold_override=1.01,
        )
