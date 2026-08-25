"""Khởi tạo Faster R-CNN và RetinaNet từ Torchvision."""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


SUPPORTED_MODELS = {
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
}

WeightsOption = Literal["DEFAULT", "NONE"] | None


def _resolve_retinanet_weights(
    weights: WeightsOption | RetinaNet_ResNet50_FPN_V2_Weights,
) -> RetinaNet_ResNet50_FPN_V2_Weights | None:
    """Chuyển giá trị cấu hình RetinaNet thành weights enum của Torchvision.

    ``NONE`` phù hợp với smoke test vì cả COCO weights lẫn ImageNet backbone
    weights đều bị tắt, do đó không phát sinh tải trọng số từ mạng.
    """
    if weights is None or weights == "NONE":
        return None
    if weights == "DEFAULT":
        return RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    if isinstance(weights, RetinaNet_ResNet50_FPN_V2_Weights):
        return weights
    raise ValueError("weights của RetinaNet chỉ nhận DEFAULT hoặc NONE.")


def build_model(
    name: str,
    num_classes: int,
    min_size: int = 512,
    max_size: int = 768,
    trainable_backbone_layers: int = 3,
    weights: WeightsOption | RetinaNet_ResNet50_FPN_V2_Weights = "DEFAULT",
):
    """Tạo detector và thay classification head theo ``num_classes``.

    Theo quy ước detector của Torchvision, ``num_classes`` bao gồm lớp
    background (chỉ số 0); hai kiến trúc phải nhận cùng giá trị từ config chung.
    """
    if name == "fasterrcnn_resnet50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        input_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
        return model

    if name == "retinanet_resnet50_fpn_v2":
        resolved_weights = _resolve_retinanet_weights(weights)
        model = retinanet_resnet50_fpn_v2(
            weights=resolved_weights,
            # Khi weights=None, Torchvision mặc định có thể tải ImageNet
            # backbone weights. Đặt rõ None để smoke test chạy ngoại tuyến.
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        anchors_per_location = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=anchors_per_location,
            num_classes=num_classes,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        return model

    choices = ", ".join(sorted(SUPPORTED_MODELS))
    raise ValueError(f"Mô hình không được hỗ trợ: {name}. Các lựa chọn: {choices}")
