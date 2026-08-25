"""Dataset COCO tối giản dành cho Faster R-CNN và RetinaNet."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class CocoBoxDataset(Dataset):
    """Đọc ảnh và bounding box từ annotation COCO JSON.

    Lớp này chỉ xử lý object detection bằng bounding box. Việc kiểm tra chất
    lượng category mapping phải được thực hiện trước khi huấn luyện.
    """

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        transforms: Any | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms

        with self.annotation_file.open("r", encoding="utf-8") as stream:
            coco = json.load(stream)

        self.images = sorted(coco.get("images", []), key=lambda item: item["id"])
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)

        self.categories = {
            int(category["id"]): str(category["name"])
            for category in coco.get("categories", [])
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.image_root / image_info["file_name"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        crowds: list[int] = []

        for annotation in self.annotations_by_image.get(image_id, []):
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(int(annotation["category_id"]))
            areas.append(float(annotation.get("area", width * height)))
            crowds.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    """Giữ ảnh có kích thước khác nhau dưới dạng danh sách."""
    return tuple(zip(*batch))
