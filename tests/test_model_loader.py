"""Kiểm thử catalog demo mà không nạp checkpoint vào RAM/GPU."""

from app.model_loader import list_model_metadata, model_metadata


def test_model_catalog_uses_validation_thresholds() -> None:
    models = {item["id"]: item for item in list_model_metadata()}
    assert set(models) == {"faster_rcnn", "retinanet"}
    assert models["faster_rcnn"]["default_threshold"] == 0.85
    assert models["retinanet"]["default_threshold"] == 0.6
    assert models["faster_rcnn"]["threshold_source"] == "val"
    assert models["retinanet"]["threshold_source"] == "val"
    assert all(item["checkpoint_available"] for item in models.values())


def test_model_catalog_rejects_unknown_model() -> None:
    try:
        model_metadata("unknown")
    except ValueError as error:
        assert "không được hỗ trợ" in str(error)
    else:
        raise AssertionError("Model không xác định phải bị từ chối")
