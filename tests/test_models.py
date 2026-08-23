"""Kiểm thử danh sách kiến trúc được dự án hỗ trợ."""

from src.models import SUPPORTED_MODELS


def test_expected_models_are_supported():
    assert "fasterrcnn_resnet50_fpn_v2" in SUPPORTED_MODELS
    assert "retinanet_resnet50_fpn_v2" in SUPPORTED_MODELS
