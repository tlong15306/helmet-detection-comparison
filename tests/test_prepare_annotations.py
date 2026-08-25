"""Regression tests for safe COCO annotation processing."""

from __future__ import annotations

from PIL import Image

from tools.prepare_annotations import prepare_coco


def _coco() -> dict:
    return {
        "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 80}],
        "categories": [{"id": 1, "name": "BikeWithRider"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [95, 10, 10, 20]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 0]},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [10, 10, 200, 20]},
        ],
    }


def test_prepare_keeps_valid_clips_minor_and_excludes_unsafe_boxes() -> None:
    processed, changes, problems = prepare_coco(_coco(), minor_overflow_pixels=10)

    assert [item["id"] for item in processed["annotations"]] == [1, 2]
    clipped = processed["annotations"][1]
    assert clipped["bbox"] == [95.0, 10.0, 5.0, 20.0]
    assert clipped["area"] == 100.0
    assert [item["action"] for item in changes] == ["clip", "exclude_annotation", "exclude_annotation"]
    assert {item["reason"] for item in problems} == {
        "minor_bbox_overflow",
        "non_positive_bbox_size",
        "severe_bbox_overflow_requires_review",
    }
    assert processed["processing_metadata"]["policy"]["raw_data_modified"] is False


def test_prepare_does_not_mutate_input() -> None:
    source = _coco()
    prepare_coco(source, minor_overflow_pixels=10)
    assert source["annotations"][1]["bbox"] == [95, 10, 10, 20]


def test_prepare_uses_exif_display_dimensions_when_requested(tmp_path) -> None:
    image_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 10), "white").save(image_path, exif=exif)
    coco = {
        "images": [{"id": 1, "file_name": "rotated.jpg", "width": 20, "height": 10}],
        "categories": [{"id": 1, "name": "BikeWithRider"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 15]}],
    }

    processed, changes, problems = prepare_coco(
        coco,
        minor_overflow_pixels=0,
        images_dir=tmp_path,
        apply_exif_orientation=True,
    )

    assert processed["images"][0]["width"] == 10
    assert processed["images"][0]["height"] == 20
    assert [item["id"] for item in processed["annotations"]] == [1]
    assert changes[0]["reason"] == "exif_orientation"
    assert problems == []
