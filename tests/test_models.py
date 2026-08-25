"""Kiểm thử model factory mà không tải trọng số từ mạng."""

from types import SimpleNamespace

import pytest

from src import models
from src.models import SUPPORTED_MODELS


def test_expected_models_are_supported():
    assert "fasterrcnn_resnet50_fpn_v2" in SUPPORTED_MODELS
    assert "retinanet_resnet50_fpn_v2" in SUPPORTED_MODELS


def test_retinanet_none_weights_disables_all_downloads(monkeypatch):
    calls = {}
    fake_model = SimpleNamespace(
        anchor_generator=SimpleNamespace(num_anchors_per_location=lambda: [9]),
        backbone=SimpleNamespace(out_channels=256),
        head=SimpleNamespace(classification_head=None),
    )

    def fake_retinanet(**kwargs):
        calls.update(kwargs)
        return fake_model

    monkeypatch.setattr(models, "retinanet_resnet50_fpn_v2", fake_retinanet)
    monkeypatch.setattr(
        models,
        "RetinaNetClassificationHead",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    model = models.build_model(
        "retinanet_resnet50_fpn_v2",
        num_classes=4,
        weights="NONE",
    )

    assert model is fake_model
    assert calls["weights"] is None
    assert calls["weights_backbone"] is None
    assert model.head.classification_head.num_classes == 4
    assert model.head.classification_head.num_anchors == 9


@pytest.mark.parametrize("value", [None, "NONE"])
def test_retinanet_none_weight_options_are_offline(value):
    assert models._resolve_retinanet_weights(value) is None


def test_retinanet_rejects_unknown_weight_option():
    with pytest.raises(ValueError, match="DEFAULT hoặc NONE"):
        models._resolve_retinanet_weights("IMAGENET")


@pytest.mark.parametrize("value", [None, "NONE"])
def test_fasterrcnn_none_weight_options_are_offline(value):
    assert models._resolve_fasterrcnn_weights(value) is None


def test_fasterrcnn_rejects_unknown_weight_option():
    with pytest.raises(ValueError, match="DEFAULT hoặc NONE"):
        models._resolve_fasterrcnn_weights("IMAGENET")
