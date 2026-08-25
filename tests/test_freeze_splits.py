"""Kiểm thử kiểm tra toàn vẹn split trước smoke test."""

from __future__ import annotations

import pytest

from tools.freeze_splits import validate_splits


def _split(image_id: int) -> dict:
    return {
        "images": [{"id": image_id, "file_name": f"{image_id}.jpg", "width": 32, "height": 24}],
        "categories": [{"id": 1, "name": "BikeWithRider"}],
        "annotations": [{"id": image_id, "image_id": image_id, "category_id": 1, "bbox": [1, 1, 10, 10]}],
    }


def test_freeze_validation_accepts_disjoint_valid_splits() -> None:
    result = validate_splits({"train": _split(1), "val": _split(2), "test": _split(3)})
    assert result["image_union"] == 3


def test_freeze_validation_rejects_image_leakage() -> None:
    with pytest.raises(ValueError, match="giao nhau"):
        validate_splits({"train": _split(1), "val": _split(1), "test": _split(3)})
