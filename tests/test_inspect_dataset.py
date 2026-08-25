"""Kiểm thử công cụ kiểm định dataset bằng một mẫu COCO tối thiểu."""

from __future__ import annotations

from PIL import Image

from tools.inspect_dataset import inspect_annotations, inspect_image_files


def test_inspect_valid_coco_dataset(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (32, 24), color="white").save(images_dir / "sample.jpg")

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "sample.jpg",
                "width": 32,
                "height": 24,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [1, 2, 10, 8],
            }
        ],
        "categories": [{"id": 2, "name": "NoHelmet"}],
    }

    annotation_report = inspect_annotations(coco, sample_limit=20)
    image_report = inspect_image_files(images_dir, coco["images"], sample_limit=20)

    assert annotation_report["counts"] == {
        "images_in_json": 1,
        "annotations": 1,
        "categories": 1,
    }
    assert all(count == 0 for count in annotation_report["issue_counts"].values())
    assert image_report["images_on_disk"] == 1
    assert all(count == 0 for count in image_report["issue_counts"].values())


def test_inspect_reports_missing_image_and_invalid_box(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "missing.jpg",
                "width": 32,
                "height": 24,
            }
        ],
        "annotations": [
            {
                "id": 7,
                "image_id": 1,
                "category_id": 2,
                "bbox": [1, 2, 0, 8],
            }
        ],
        "categories": [{"id": 2, "name": "NoHelmet"}],
    }

    annotation_report = inspect_annotations(coco, sample_limit=20)
    image_report = inspect_image_files(images_dir, coco["images"], sample_limit=20)

    assert annotation_report["issue_counts"]["invalid_boxes"] == 1
    assert annotation_report["issue_samples"]["invalid_boxes"] == [7]
    assert image_report["issue_counts"]["missing_image_files"] == 1
    assert image_report["issue_samples"]["missing_image_files"] == [
        "missing.jpg"
    ]
