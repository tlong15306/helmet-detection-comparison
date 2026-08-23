"""Các hàm metric cơ bản và giao diện mở rộng cho COCO mAP."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou


def pairwise_iou(predicted_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Tính ma trận IoU giữa các hộp dự đoán và hộp ground truth."""
    return box_iou(predicted_boxes, target_boxes)


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Tính Precision và Recall từ TP, FP, FN."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def build_map_metric(class_metrics: bool = True):
    """Khởi tạo COCO-style mAP bằng TorchMetrics."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
        backend="pycocotools",
    )
