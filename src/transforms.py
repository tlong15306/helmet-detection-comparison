"""Biến đổi ảnh và bounding box cho bài toán object detection."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch
from torchvision.transforms import functional as F


class Compose:
    """Áp dụng tuần tự các phép biến đổi cho cả ảnh và target."""

    def __init__(self, transforms: list[Callable[..., Any]]) -> None:
        self.transforms = transforms

    def __call__(self, image: Any, target: dict[str, torch.Tensor]):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToFloatTensor:
    """Chuyển ảnh PIL thành tensor float trong khoảng [0, 1]."""

    def __call__(self, image: Any, target: dict[str, torch.Tensor]):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image, torch.float32)
        return image, target


class RandomHorizontalFlip:
    """Lật ngang ảnh và cập nhật tọa độ bounding box tương ứng."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Any, target: dict[str, torch.Tensor]):
        if random.random() >= self.probability:
            return image, target

        image = F.hflip(image)
        width = image.width if hasattr(image, "width") else image.shape[-1]
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target = dict(target)
        target["boxes"] = boxes
        return image, target


def build_transforms(train: bool, horizontal_flip_probability: float = 0.5) -> Compose:
    """Tạo pipeline augmentation tối thiểu và an toàn cho baseline."""
    operations: list[Callable[..., Any]] = []
    if train:
        operations.append(RandomHorizontalFlip(horizontal_flip_probability))
    operations.append(ToFloatTensor())
    return Compose(operations)
