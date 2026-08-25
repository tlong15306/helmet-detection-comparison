"""Metric chung để đánh giá công bằng Faster R-CNN và RetinaNet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    """Khởi tạo COCO-style mAP bằng TorchMetrics và pycocotools."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
        backend="pycocotools",
    )


def _finite_metric(value: torch.Tensor | float | int) -> float | None:
    """Chuyển giá trị metric sang JSON và biểu diễn giá trị không xác định bằng null."""
    number = float(value.item() if isinstance(value, torch.Tensor) else value)
    return number if torch.isfinite(torch.tensor(number)) and number >= 0 else None


@dataclass
class DetectionCounts:
    """Số prediction đúng/sai và ground truth bị bỏ sót của một lớp."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def as_dict(self) -> dict[str, int | float]:
        precision, recall = precision_recall(
            self.true_positive,
            self.false_positive,
            self.false_negative,
        )
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": precision,
            "recall": recall,
        }


class DetectionEvaluator:
    """Tích lũy COCO mAP và Precision/Recall theo cùng một giao thức.

    Precision/Recall dùng ghép một-một: prediction được sắp theo confidence
    giảm dần, chỉ là TP nếu cùng lớp và ghép được với một ground truth chưa
    ghép có IoU không nhỏ hơn ngưỡng. Các ground truth còn lại là FN.
    """

    def __init__(
        self,
        class_names: Mapping[int, str],
        iou_threshold: float = 0.50,
        confidence_threshold: float | None = None,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold phải nằm trong [0, 1]")
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold phải nằm trong [0, 1]")

        self.class_names = {
            int(class_id): str(name)
            for class_id, name in class_names.items()
            if int(class_id) != 0
        }
        if not self.class_names:
            raise ValueError("Cần ít nhất một lớp detection khác background")

        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.map_metric = build_map_metric(class_metrics=True)
        self.counts = {
            class_id: DetectionCounts() for class_id in self.class_names
        }
        self.images_evaluated = 0

    @staticmethod
    def _tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype).detach().cpu()

    def _prepare_prediction(self, prediction: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        boxes = self._tensor(prediction.get("boxes", []), torch.float32).reshape(-1, 4)
        labels = self._tensor(prediction.get("labels", []), torch.int64).reshape(-1)
        scores = self._tensor(prediction.get("scores", []), torch.float32).reshape(-1)
        if len(boxes) != len(labels) or len(labels) != len(scores):
            raise ValueError("Prediction boxes, labels và scores phải cùng độ dài")
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("Prediction boxes và scores phải là số hữu hạn")
        if torch.any(boxes[:, 2] < boxes[:, 0]) or torch.any(boxes[:, 3] < boxes[:, 1]):
            raise ValueError("Prediction boxes phải theo định dạng xyxy hợp lệ")

        known_labels = torch.tensor(sorted(self.class_names), dtype=torch.int64)
        unknown_mask = ~torch.isin(labels, known_labels)
        if unknown_mask.any():
            unknown = sorted(set(labels[unknown_mask].tolist()))
            raise ValueError(f"Prediction có nhãn ngoài class_names: {unknown}")
        return {"boxes": boxes, "labels": labels, "scores": scores}

    def _prepare_target(self, target: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        boxes = self._tensor(target.get("boxes", []), torch.float32).reshape(-1, 4)
        labels = self._tensor(target.get("labels", []), torch.int64).reshape(-1)
        if len(boxes) != len(labels):
            raise ValueError("Target boxes và labels phải cùng độ dài")
        if not torch.isfinite(boxes).all():
            raise ValueError("Target boxes phải là số hữu hạn")
        if torch.any(boxes[:, 2] < boxes[:, 0]) or torch.any(boxes[:, 3] < boxes[:, 1]):
            raise ValueError("Target boxes phải theo định dạng xyxy hợp lệ")

        known_labels = torch.tensor(sorted(self.class_names), dtype=torch.int64)
        unknown_mask = ~torch.isin(labels, known_labels)
        if unknown_mask.any():
            unknown = sorted(set(labels[unknown_mask].tolist()))
            raise ValueError(f"Target có nhãn ngoài class_names: {unknown}")

        prepared = {"boxes": boxes, "labels": labels}
        for field, dtype in (("area", torch.float32), ("iscrowd", torch.int64)):
            if field not in target:
                continue
            values = self._tensor(target[field], dtype).reshape(-1)
            if len(values) != len(labels):
                raise ValueError(f"Target {field} phải có cùng độ dài với labels")
            prepared[field] = values
        return prepared

    def update(
        self,
        predictions: Sequence[Mapping[str, Any]],
        targets: Sequence[Mapping[str, Any]],
    ) -> None:
        """Bổ sung một batch prediction/ground truth vào evaluator."""
        if len(predictions) != len(targets):
            raise ValueError("Số prediction phải bằng số target trong cùng batch")

        prepared_predictions = [
            self._prepare_prediction(prediction) for prediction in predictions
        ]
        prepared_targets = [self._prepare_target(target) for target in targets]
        self.map_metric.update(prepared_predictions, prepared_targets)

        for prediction, target in zip(prepared_predictions, prepared_targets):
            self._update_precision_recall(prediction, target)
            self.images_evaluated += 1

    def _update_precision_recall(
        self,
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
    ) -> None:
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        prediction_boxes = prediction["boxes"]
        prediction_labels = prediction["labels"]
        prediction_scores = prediction["scores"]

        for class_id, class_counts in self.counts.items():
            ground_truth_indices = torch.where(target_labels == class_id)[0]
            predicted_indices = torch.where(prediction_labels == class_id)[0]
            predicted_indices = predicted_indices[
                torch.argsort(prediction_scores[predicted_indices], descending=True)
            ]
            matched_ground_truth: set[int] = set()

            for predicted_index in predicted_indices.tolist():
                if (
                    self.confidence_threshold is not None
                    and prediction_scores[predicted_index].item()
                    < self.confidence_threshold
                ):
                    continue

                available_indices = [
                    index.item()
                    for index in ground_truth_indices
                    if index.item() not in matched_ground_truth
                ]
                if not available_indices:
                    class_counts.false_positive += 1
                    continue

                candidate_boxes = target_boxes[available_indices]
                overlaps = box_iou(
                    prediction_boxes[predicted_index].unsqueeze(0),
                    candidate_boxes,
                ).squeeze(0)
                best_overlap, best_position = overlaps.max(dim=0)
                if best_overlap.item() >= self.iou_threshold:
                    matched_ground_truth.add(available_indices[best_position.item()])
                    class_counts.true_positive += 1
                else:
                    class_counts.false_positive += 1

            class_counts.false_negative += len(ground_truth_indices) - len(
                matched_ground_truth
            )

    def compute(self) -> dict[str, Any]:
        """Trả metric dưới dạng JSON-serializable để ghi kết quả thực nghiệm."""
        raw_map = self.map_metric.compute()
        summary = {
            "images_evaluated": self.images_evaluated,
            "iou_threshold_for_precision_recall": self.iou_threshold,
            "confidence_threshold_for_precision_recall": self.confidence_threshold,
            "map_50_95": _finite_metric(raw_map["map"]),
            "map_50": _finite_metric(raw_map["map_50"]),
            "map_75": _finite_metric(raw_map["map_75"]),
            "mar_100": _finite_metric(raw_map["mar_100"]),
            "per_class": {},
        }

        map_classes = raw_map.get("classes")
        map_by_class: dict[int, dict[str, float | None]] = {}
        if map_classes is not None:
            class_ids = torch.as_tensor(map_classes).reshape(-1).tolist()
            map_per_class = torch.as_tensor(raw_map["map_per_class"]).reshape(-1)
            map_50_per_class = raw_map.get("map_50_per_class")
            if map_50_per_class is not None:
                map_50_per_class = torch.as_tensor(map_50_per_class).reshape(-1)
            for position, class_id in enumerate(class_ids):
                map_by_class[int(class_id)] = {
                    "ap_50_95": _finite_metric(map_per_class[position]),
                    "ap_50": (
                        _finite_metric(map_50_per_class[position])
                        if map_50_per_class is not None
                        else None
                    ),
                }

        for class_id, class_name in self.class_names.items():
            per_class_metrics: dict[str, Any] = {
                "class_id": class_id,
                **self.counts[class_id].as_dict(),
            }
            per_class_metrics.update(
                map_by_class.get(class_id, {"ap_50_95": None, "ap_50": None})
            )
            summary["per_class"][class_name] = per_class_metrics
        return summary
