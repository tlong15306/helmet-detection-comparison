"""Khởi tạo Faster R-CNN và RetinaNet từ Torchvision."""

from __future__ import annotations

from functools import partial

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


def build_model(
    name: str,
    num_classes: int,
    min_size: int = 512,
    max_size: int = 768,
    trainable_backbone_layers: int = 3,
):
    """Tạo detector pretrained COCO và thay classification head."""
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
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
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
