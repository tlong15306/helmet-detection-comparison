"""Tests for deterministic group-aware COCO splits."""

from __future__ import annotations

from tools.create_splits import _ratios, build_splits


def _coco() -> dict:
    return {
        "images": [
            {"id": index, "file_name": f"{index}.jpg", "width": 32, "height": 24}
            for index in range(1, 11)
        ],
        "categories": [{"id": 1, "name": "BikeWithRider"}, {"id": 2, "name": "NoHelmet"}],
        "annotations": [
            {"id": index, "image_id": index, "category_id": 1 if index % 2 else 2, "bbox": [1, 1, 10, 10]}
            for index in range(1, 11)
        ],
    }


def test_grouped_split_is_deterministic_and_has_no_group_leakage() -> None:
    coco = _coco()
    groups = {index: "scene_a" if index in {1, 2} else f"scene_{index}" for index in range(1, 11)}
    ratios = _ratios(0.7, 0.15, 0.15)

    first, first_summary = build_splits(coco, groups, ratios, seed=42)
    second, second_summary = build_splits(coco, groups, ratios, seed=42)

    assert first_summary == second_summary
    assert {item["id"] for item in first["train"]["images"]} == {item["id"] for item in second["train"]["images"]}
    image_sets = [{item["id"] for item in first[split]["images"]} for split in ("train", "val", "test")]
    assert len(set.union(*image_sets)) == 10
    assert not (image_sets[0] & image_sets[1])
    assert not (image_sets[0] & image_sets[2])
    assert not (image_sets[1] & image_sets[2])
    assert any({1, 2}.issubset(image_ids) for image_ids in image_sets)


def test_ratios_must_sum_to_one() -> None:
    try:
        _ratios(0.7, 0.2, 0.2)
    except ValueError as error:
        assert "tổng bằng 1" in str(error)
    else:
        raise AssertionError("Expected invalid ratios to fail")
