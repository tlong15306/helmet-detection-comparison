"""Kiểm thử catalog demo mà không nạp checkpoint vào RAM/GPU."""

from app.model_loader import list_model_metadata, model_metadata


def test_model_catalog_uses_validation_thresholds() -> None:
    models = {item["id"]: item for item in list_model_metadata()}
    assert set(models) == {
        "faster_rcnn",
        "retinanet",
        "high_accuracy",
        "faster_rcnn_vietnam_v6",
        "retinanet_vietnam_v6",
    }
    assert models["faster_rcnn"]["default_threshold"] == 0.65
    assert models["retinanet"]["default_threshold"] == 0.4
    assert models["high_accuracy"]["default_threshold"] == 0.45
    assert models["faster_rcnn"]["default_thresholds"] == {
        "BikeWithRider": 0.95,
        "NoHelmet": 0.65,
        "Helmet": 0.70,
    }
    assert models["retinanet"]["default_thresholds"] == {
        "BikeWithRider": 0.65,
        "NoHelmet": 0.40,
        "Helmet": 0.40,
    }
    assert models["faster_rcnn"]["threshold_source"] == "val"
    assert models["retinanet"]["threshold_source"] == "val"
    assert models["faster_rcnn"]["inference_mode"] == "horizontal_flip_tta"
    assert models["retinanet"]["inference_mode"] == "horizontal_flip_tta"
    assert models["high_accuracy"]["default_thresholds"] == {
        "BikeWithRider": 0.55,
        "NoHelmet": 0.45,
        "Helmet": 0.45,
    }
    assert models["high_accuracy"]["threshold_source"] == "val"
    assert models["faster_rcnn_vietnam_v6"]["threshold_source"] == "exploratory"
    assert models["retinanet_vietnam_v6"]["threshold_source"] == "exploratory"
    assert all(item["checkpoint_available"] for item in models.values())
    assert all("finetune_roboflow_consensus" not in item["checkpoint"] for item in models.values())


def test_model_catalog_rejects_unknown_model() -> None:
    try:
        model_metadata("unknown")
    except ValueError as error:
        assert "không được hỗ trợ" in str(error)
    else:
        raise AssertionError("Model không xác định phải bị từ chối")
