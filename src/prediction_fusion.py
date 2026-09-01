"""Hợp nhất prediction từ nhiều detector hoặc nhiều phép TTA.

Các hàm trong module này không thay đổi checkpoint. Hộp chỉ được gộp khi
cùng lớp và có IoU đủ lớn; điểm đồng thuận giúp các detection được nhiều
nguồn cùng xác nhận đứng trước detection chỉ xuất hiện ở một nguồn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torchvision.ops import box_iou


class EnsembleDetector(torch.nn.Module):
    """Bọc nhiều Torchvision detector thành một detector hợp nhất.

    Giao diện ``forward(images)`` giống detector gốc nên có thể dùng lại toàn
    bộ pipeline ảnh/video hiện có. Các mô hình chạy tuần tự để giảm đỉnh bộ nhớ
    kích hoạt trên GPU.
    """

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        *,
        iou_threshold: float = 0.55,
        max_detections: int = 100,
    ) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("EnsembleDetector cần ít nhất hai mô hình")
        self.models = torch.nn.ModuleList(models)
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        outputs_by_model = [model(images) for model in self.models]
        return [
            fuse_predictions(
                [outputs[image_index] for outputs in outputs_by_model],
                iou_threshold=self.iou_threshold,
                max_detections=self.max_detections,
            )
            for image_index in range(len(images))
        ]


class HorizontalFlipTTADetector(torch.nn.Module):
    """Tăng độ chính xác một detector bằng ảnh gốc và ảnh lật ngang."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        iou_threshold: float = 0.55,
        max_detections: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        original_outputs = self.model(images)
        flipped_images = [torch.flip(image, dims=(-1,)) for image in images]
        flipped_outputs = self.model(flipped_images)
        restored_outputs = [
            horizontal_flip_prediction(output, int(image.shape[-1]))
            for output, image in zip(flipped_outputs, images, strict=True)
        ]
        return [
            fuse_predictions(
                [original, restored],
                iou_threshold=self.iou_threshold,
                max_detections=self.max_detections,
            )
            for original, restored in zip(
                original_outputs, restored_outputs, strict=True
            )
        ]


def horizontal_flip_prediction(
    prediction: Mapping[str, torch.Tensor], image_width: int
) -> dict[str, torch.Tensor]:
    """Đưa prediction của ảnh lật ngang về hệ tọa độ ảnh gốc."""
    if image_width <= 0:
        raise ValueError("image_width phải lớn hơn 0")
    boxes = prediction["boxes"].detach().clone()
    boxes[:, [0, 2]] = float(image_width) - boxes[:, [2, 0]]
    return {
        "boxes": boxes,
        "labels": prediction["labels"].detach().clone(),
        "scores": prediction["scores"].detach().clone(),
    }


def _empty_prediction(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "labels": torch.empty((0,), dtype=torch.int64, device=device),
        "scores": torch.empty((0,), dtype=torch.float32, device=device),
    }


def fuse_predictions(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    *,
    source_weights: Sequence[float] | None = None,
    iou_threshold: float = 0.55,
    max_detections: int = 100,
) -> dict[str, torch.Tensor]:
    """Weighted box fusion nhỏ gọn cho detection cùng lớp.

    Điểm của một cụm bằng tổng điểm tốt nhất từ mỗi nguồn, có trọng số, chia
    cho tổng trọng số của mọi nguồn. Vì vậy detection được nhiều nguồn xác
    nhận sẽ được ưu tiên, còn detection đơn lẻ vẫn được giữ nhưng có điểm thấp
    hơn. Tọa độ hộp là trung bình theo ``score * source_weight``.
    """
    if not predictions:
        return _empty_prediction()
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold phải nằm trong (0, 1]")
    if max_detections < 1:
        raise ValueError("max_detections phải lớn hơn hoặc bằng 1")
    if source_weights is None:
        source_weights = [1.0] * len(predictions)
    if len(source_weights) != len(predictions):
        raise ValueError("source_weights phải có cùng độ dài với predictions")
    if any(float(weight) <= 0.0 for weight in source_weights):
        raise ValueError("Mọi source_weight phải lớn hơn 0")

    device = predictions[0]["boxes"].device
    labels_present = sorted(
        {
            int(label)
            for prediction in predictions
            for label in prediction["labels"].detach().cpu().tolist()
        }
    )
    if not labels_present:
        return _empty_prediction(device)

    total_source_weight = float(sum(float(weight) for weight in source_weights))
    fused_boxes: list[torch.Tensor] = []
    fused_labels: list[int] = []
    fused_scores: list[float] = []

    for class_id in labels_present:
        candidates: list[dict[str, object]] = []
        for source_index, (prediction, source_weight) in enumerate(
            zip(predictions, source_weights)
        ):
            class_indices = torch.where(prediction["labels"] == class_id)[0]
            for index in class_indices.tolist():
                candidates.append(
                    {
                        "box": prediction["boxes"][index].detach(),
                        "score": float(prediction["scores"][index].item()),
                        "source": source_index,
                        "source_weight": float(source_weight),
                    }
                )
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)

        clusters: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_box = candidate["box"]
            assert isinstance(candidate_box, torch.Tensor)
            best_cluster: dict[str, object] | None = None
            best_iou = -1.0
            for cluster in clusters:
                cluster_box = cluster["box"]
                assert isinstance(cluster_box, torch.Tensor)
                overlap = float(
                    box_iou(candidate_box.unsqueeze(0), cluster_box.unsqueeze(0))[0, 0]
                )
                if overlap >= iou_threshold and overlap > best_iou:
                    best_cluster = cluster
                    best_iou = overlap

            if best_cluster is None:
                weighted_score = float(candidate["score"]) * float(
                    candidate["source_weight"]
                )
                clusters.append(
                    {
                        "box": candidate_box.clone(),
                        "members": [candidate],
                        "box_weight_sum": weighted_score,
                    }
                )
                continue

            members = best_cluster["members"]
            assert isinstance(members, list)
            members.append(candidate)
            weighted_boxes = []
            box_weights = []
            for member in members:
                member_box = member["box"]
                assert isinstance(member_box, torch.Tensor)
                weight = float(member["score"]) * float(member["source_weight"])
                weighted_boxes.append(member_box * weight)
                box_weights.append(weight)
            weight_sum = max(sum(box_weights), 1e-12)
            best_cluster["box"] = torch.stack(weighted_boxes).sum(dim=0) / weight_sum
            best_cluster["box_weight_sum"] = weight_sum

        for cluster in clusters:
            members = cluster["members"]
            cluster_box = cluster["box"]
            assert isinstance(members, list)
            assert isinstance(cluster_box, torch.Tensor)
            best_by_source: dict[int, float] = {}
            for member in members:
                source_index = int(member["source"])
                weighted_score = float(member["score"]) * float(
                    member["source_weight"]
                )
                best_by_source[source_index] = max(
                    best_by_source.get(source_index, 0.0), weighted_score
                )
            fused_boxes.append(cluster_box)
            fused_labels.append(class_id)
            fused_scores.append(
                min(1.0, sum(best_by_source.values()) / total_source_weight)
            )

    if not fused_boxes:
        return _empty_prediction(device)
    boxes_tensor = torch.stack(fused_boxes)
    labels_tensor = torch.tensor(fused_labels, dtype=torch.int64, device=device)
    scores_tensor = torch.tensor(fused_scores, dtype=torch.float32, device=device)
    order = torch.argsort(scores_tensor, descending=True)[:max_detections]
    return {
        "boxes": boxes_tensor[order],
        "labels": labels_tensor[order],
        "scores": scores_tensor[order],
    }
