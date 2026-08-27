"""Nạp và quản lý một detector trên GPU tại một thời điểm cho demo."""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from src.evaluate import (
    build_evaluation_model,
    class_names_from_config,
    default_checkpoint,
    file_sha256,
    load_checkpoint_into_model,
)
from src.infer import predict_image
from src.utils import load_config, load_yaml, resolve_project_path


MODEL_SPECS: dict[str, dict[str, str]] = {
    "faster_rcnn": {
        "display_name": "Faster R-CNN",
        "config": "configs/faster_rcnn.yaml",
    },
    "retinanet": {
        "display_name": "RetinaNet",
        "config": "configs/retinanet.yaml",
    },
}


@dataclass
class LoadedDetector:
    model_id: str
    display_name: str
    model: torch.nn.Module
    device: torch.device
    config: dict[str, Any]
    class_names: dict[int, str]
    default_threshold: float
    checkpoint_metadata: dict[str, Any]


def _threshold_entry(model_name: str) -> dict[str, Any]:
    threshold_config = load_yaml("configs/demo_thresholds.yaml")
    if threshold_config.get("schema_version") != "demo-thresholds-1.0":
        raise ValueError("Schema configs/demo_thresholds.yaml không được hỗ trợ")
    models = threshold_config.get("models", {})
    entry = models.get(model_name)
    if not isinstance(entry, dict):
        raise ValueError(f"Không có threshold validation cho model {model_name}")
    threshold = float(entry.get("confidence_threshold", -1))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold không hợp lệ cho model {model_name}: {threshold}")
    if entry.get("selection_split") != "val":
        raise ValueError("Threshold demo bắt buộc phải được chọn trên validation")
    expected_hash = str(entry.get("checkpoint_sha256", "")).strip().lower()
    if len(expected_hash) != 64:
        raise ValueError(f"Thiếu SHA-256 checkpoint cho model {model_name}")
    return dict(entry)


def model_metadata(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Model không được hỗ trợ: {model_id}")
    spec = MODEL_SPECS[model_id]
    config = load_config(spec["config"])
    model_name = str(config["model"]["name"])
    threshold = _threshold_entry(model_name)
    checkpoint = resolve_project_path(default_checkpoint(config))
    return {
        "id": model_id,
        "name": spec["display_name"],
        "architecture": model_name,
        "description": (
            "Mô hình phát hiện hai giai đoạn"
            if model_id == "faster_rcnn"
            else "Mô hình phát hiện một giai đoạn"
        ),
        "checkpoint": checkpoint.relative_to(resolve_project_path(".")).as_posix(),
        "checkpoint_available": checkpoint.is_file(),
        "default_threshold": float(threshold["confidence_threshold"]),
        "threshold_source": str(threshold["selection_split"]),
        "classes": class_names_from_config(config),
    }


def list_model_metadata() -> list[dict[str, Any]]:
    return [model_metadata(model_id) for model_id in MODEL_SPECS]


def load_detector(model_id: str, requested_device: str = "auto") -> LoadedDetector:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Model không được hỗ trợ: {model_id}")
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("requested_device chỉ nhận auto, cpu hoặc cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng PyTorch không nhận diện được GPU")

    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    spec = MODEL_SPECS[model_id]
    config = load_config(spec["config"])
    model_name = str(config["model"]["name"])
    threshold_entry = _threshold_entry(model_name)
    checkpoint = resolve_project_path(default_checkpoint(config))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")

    expected_hash = str(threshold_entry["checkpoint_sha256"]).lower()
    actual_hash = file_sha256(checkpoint)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Checkpoint không khớp checkpoint đã dùng chọn threshold validation: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    model = build_evaluation_model(config, device)
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint, device)
    model.eval()
    return LoadedDetector(
        model_id=model_id,
        display_name=spec["display_name"],
        model=model,
        device=device,
        config=config,
        class_names=class_names_from_config(config),
        default_threshold=float(threshold_entry["confidence_threshold"]),
        checkpoint_metadata=checkpoint_metadata,
    )


class DetectorManager:
    """Tuần tự hóa inference và chỉ giữ một checkpoint trên GPU."""

    def __init__(self, requested_device: str = "auto") -> None:
        self.requested_device = requested_device
        self._lock = threading.RLock()
        self._loaded: LoadedDetector | None = None

    @property
    def current_model_id(self) -> str | None:
        with self._lock:
            return None if self._loaded is None else self._loaded.model_id

    def _release_locked(self) -> None:
        if self._loaded is None:
            return
        del self._loaded.model
        self._loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_locked(self, model_id: str) -> LoadedDetector:
        if self._loaded is not None and self._loaded.model_id == model_id:
            return self._loaded
        self._release_locked()
        self._loaded = load_detector(model_id, self.requested_device)
        return self._loaded

    def predict(
        self,
        model_id: str,
        image: Image.Image,
        confidence_threshold: float | None = None,
    ) -> tuple[LoadedDetector, dict[str, torch.Tensor], float]:
        with self._lock:
            detector = self._get_locked(model_id)
            threshold = (
                detector.default_threshold
                if confidence_threshold is None
                else float(confidence_threshold)
            )
            prediction, latency_ms = predict_image(
                detector.model, image, detector.device, threshold
            )
            return detector, prediction, latency_ms

    def release(self) -> None:
        with self._lock:
            self._release_locked()
