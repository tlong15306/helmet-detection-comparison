"""Nạp và quản lý một detector trên GPU tại một thời điểm cho demo."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
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
from src.prediction_fusion import EnsembleDetector, HorizontalFlipTTADetector
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
    "high_accuracy": {
        "display_name": "Độ chính xác cao",
        "config": "configs/faster_rcnn.yaml",
    },
    "faster_rcnn_vietnam_v6": {
        "display_name": "Faster R-CNN · thử nghiệm VN",
        "config": "configs/vietnam_v6_wikimedia_faster_rcnn.yaml",
        "threshold_key": "fasterrcnn_resnet50_fpn_v2_vietnam_v6",
    },
    "retinanet_vietnam_v6": {
        "display_name": "RetinaNet · thử nghiệm VN",
        "config": "configs/vietnam_v6_wikimedia_retinanet.yaml",
        "threshold_key": "retinanet_resnet50_fpn_v2_vietnam_v6",
    },
    "faster_rcnn_final_combined": {
        "display_name": "Faster R-CNN · baseline gộp",
        "config": "configs/final_combined_faster_rcnn.yaml",
        "threshold_key": "fasterrcnn_resnet50_fpn_v2_final_combined",
    },
    "retinanet_final_combined": {
        "display_name": "RetinaNet · baseline gộp",
        "config": "configs/final_combined_retinanet.yaml",
        "threshold_key": "retinanet_resnet50_fpn_v2_final_combined",
    },
}

ENSEMBLE_ARCHITECTURE = "fasterrcnn_retinanet_ensemble_v1"
ENSEMBLE_MEMBERS = ("configs/faster_rcnn.yaml", "configs/retinanet.yaml")


@dataclass
class LoadedDetector:
    model_id: str
    display_name: str
    model: torch.nn.Module
    device: torch.device
    config: dict[str, Any]
    class_names: dict[int, str]
    default_threshold: float
    default_thresholds: dict[int, float]
    postprocess_config: dict[str, float]
    checkpoint_metadata: dict[str, Any]


def _threshold_entry(model_name: str) -> dict[str, Any]:
    threshold_config = load_yaml("configs/demo_thresholds.yaml")
    if threshold_config.get("schema_version") not in {"demo-thresholds-1.0", "demo-thresholds-2.0"}:
        raise ValueError("Schema configs/demo_thresholds.yaml không được hỗ trợ")
    models = threshold_config.get("models", {})
    entry = models.get(model_name)
    if not isinstance(entry, dict):
        raise ValueError(f"Không có threshold validation cho model {model_name}")
    if "confidence_threshold" in entry:
        threshold = float(entry["confidence_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold không hợp lệ cho model {model_name}: {threshold}")
    elif not isinstance(entry.get("confidence_thresholds"), Mapping):
        raise ValueError(f"Thiếu confidence_threshold hoặc confidence_thresholds cho {model_name}")
    if entry.get("selection_split") != "val" and not entry.get("experimental", False):
        raise ValueError("Threshold demo bắt buộc phải được chọn trên validation")
    expected_hash = str(entry.get("checkpoint_sha256", "")).strip().lower()
    expected_hashes = entry.get("checkpoint_sha256s")
    valid_hash_mapping = isinstance(expected_hashes, Mapping) and bool(expected_hashes)
    if len(expected_hash) != 64 and not valid_hash_mapping:
        raise ValueError(f"Thiếu SHA-256 checkpoint cho model {model_name}")
    result = dict(entry)
    postprocess = threshold_config.get("postprocess", {})
    if not isinstance(postprocess, Mapping):
        raise ValueError("postprocess trong demo_thresholds.yaml phải là mapping")
    result["_postprocess"] = dict(postprocess)
    return result


def _class_thresholds(
    entry: Mapping[str, Any], class_names: Mapping[int, str]
) -> dict[int, float]:
    """Đọc ngưỡng theo tên lớp; cấu hình v1 được nâng cấp thành cùng một ngưỡng."""
    named = entry.get("confidence_thresholds")
    if isinstance(named, Mapping):
        thresholds: dict[int, float] = {}
        for class_id, class_name in class_names.items():
            if int(class_id) == 0:
                continue
            if class_name not in named:
                raise ValueError(f"Thiếu threshold cho lớp {class_name}")
            value = float(named[class_name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Threshold lớp {class_name} không hợp lệ: {value}")
            thresholds[int(class_id)] = value
        return thresholds
    threshold = float(entry["confidence_threshold"])
    return {int(class_id): threshold for class_id in class_names if int(class_id) != 0}


def _postprocess_config(entry: Mapping[str, Any]) -> dict[str, float]:
    config = entry.get("_postprocess", {})
    iou = float(config.get("head_conflict_iou_threshold", 0.70))
    margin = float(config.get("head_confidence_margin", 0.10))
    if not 0.0 <= iou <= 1.0 or margin < 0.0:
        raise ValueError("Cấu hình xử lý xung đột đầu không hợp lệ")
    return {"head_conflict_iou_threshold": iou, "head_confidence_margin": margin}


def _threshold_key(model_id: str, config: Mapping[str, Any]) -> str:
    """Khóa ngưỡng tách riêng checkpoint thử nghiệm khỏi checkpoint baseline."""
    if model_id == "high_accuracy":
        return ENSEMBLE_ARCHITECTURE
    return str(MODEL_SPECS[model_id].get("threshold_key", config["model"]["name"]))


def model_metadata(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Model không được hỗ trợ: {model_id}")
    spec = MODEL_SPECS[model_id]
    config = load_config(spec["config"])
    model_name = _threshold_key(model_id, config)
    threshold = _threshold_entry(model_name)
    class_names = class_names_from_config(config)
    thresholds = _class_thresholds(threshold, class_names)
    if model_id == "high_accuracy":
        member_configs = [load_config(path) for path in ENSEMBLE_MEMBERS]
        checkpoints = [
            resolve_project_path(default_checkpoint(member_config))
            for member_config in member_configs
        ]
        checkpoint_label = " + ".join(
            checkpoint.relative_to(resolve_project_path(".")).as_posix()
            for checkpoint in checkpoints
        )
        checkpoint_available = all(checkpoint.is_file() for checkpoint in checkpoints)
    else:
        checkpoint = resolve_project_path(default_checkpoint(config))
        checkpoint_label = checkpoint.relative_to(resolve_project_path(".")).as_posix()
        checkpoint_available = checkpoint.is_file()
    return {
        "id": model_id,
        "name": spec["display_name"],
        "architecture": model_name,
        "description": (
            "Hợp nhất Faster R-CNN và RetinaNet"
            if model_id == "high_accuracy"
            else "Fine-tune bổ sung dữ liệu Việt Nam v6; chỉ để kiểm tra"
            if model_id.endswith("_vietnam_v6")
            else "Baseline hoàn chỉnh trên dữ liệu EdgeVision gộp với dữ liệu Việt Nam đã duyệt"
            if model_id.endswith("_final_combined")
            else "Mô hình phát hiện hai giai đoạn"
            if model_id == "faster_rcnn"
            else "Mô hình phát hiện một giai đoạn"
        ),
        "checkpoint": checkpoint_label,
        "checkpoint_available": checkpoint_available,
        "default_threshold": thresholds.get(2, next(iter(thresholds.values()))),
        "default_thresholds": {class_names[class_id]: value for class_id, value in thresholds.items()},
        "threshold_source": str(threshold["selection_split"]),
        "inference_mode": str(threshold.get("inference_mode", "standard")),
        "classes": class_names,
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
    if model_id == "high_accuracy":
        threshold_entry = _threshold_entry(ENSEMBLE_ARCHITECTURE)
        expected_hashes = threshold_entry.get("checkpoint_sha256s")
        if not isinstance(expected_hashes, Mapping):
            raise ValueError("Thiếu fingerprint checkpoint cho ensemble")
        member_models: list[torch.nn.Module] = []
        member_metadata: dict[str, Any] = {}
        member_configs = [load_config(path) for path in ENSEMBLE_MEMBERS]
        class_names = class_names_from_config(member_configs[0])
        for member_config in member_configs:
            if class_names_from_config(member_config) != class_names:
                raise ValueError("Các thành viên ensemble phải có cùng class mapping")
            member_name = str(member_config["model"]["name"])
            checkpoint = resolve_project_path(default_checkpoint(member_config))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
            expected_hash = str(expected_hashes.get(member_name, "")).lower()
            actual_hash = file_sha256(checkpoint)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Checkpoint ensemble không khớp fingerprint validation: "
                    f"model={member_name}, expected={expected_hash}, actual={actual_hash}"
                )
            member_model = build_evaluation_model(member_config, device)
            member_metadata[member_name] = load_checkpoint_into_model(
                member_model, checkpoint, device
            )
            member_model.eval()
            member_models.append(member_model)
        thresholds = _class_thresholds(threshold_entry, class_names)
        ensemble = EnsembleDetector(
            member_models,
            iou_threshold=float(threshold_entry.get("fusion_iou_threshold", 0.55)),
            max_detections=int(threshold_entry.get("max_detections", 100)),
        ).to(device)
        ensemble.eval()
        combined_config = dict(member_configs[0])
        combined_config["model"] = {
            **member_configs[0]["model"],
            "name": ENSEMBLE_ARCHITECTURE,
        }
        return LoadedDetector(
            model_id=model_id,
            display_name=spec["display_name"],
            model=ensemble,
            device=device,
            config=combined_config,
            class_names=class_names,
            default_threshold=thresholds.get(2, next(iter(thresholds.values()))),
            default_thresholds=thresholds,
            postprocess_config=_postprocess_config(threshold_entry),
            checkpoint_metadata=member_metadata,
        )

    config = load_config(spec["config"])
    threshold_entry = _threshold_entry(_threshold_key(model_id, config))
    class_names = class_names_from_config(config)
    thresholds = _class_thresholds(threshold_entry, class_names)
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
    if threshold_entry.get("inference_mode") == "horizontal_flip_tta":
        model = HorizontalFlipTTADetector(
            model,
            iou_threshold=float(threshold_entry.get("fusion_iou_threshold", 0.55)),
            max_detections=int(threshold_entry.get("max_detections", 100)),
        ).to(device)
        model.eval()
    return LoadedDetector(
        model_id=model_id,
        display_name=spec["display_name"],
        model=model,
        device=device,
        config=config,
        class_names=class_names,
        default_threshold=thresholds.get(2, next(iter(thresholds.values()))),
        default_thresholds=thresholds,
        postprocess_config=_postprocess_config(threshold_entry),
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
        confidence_threshold: float | Mapping[int, float] | None = None,
    ) -> tuple[LoadedDetector, dict[str, torch.Tensor], float]:
        with self._lock:
            detector = self._get_locked(model_id)
            threshold = (
                detector.default_thresholds
                if confidence_threshold is None
                else confidence_threshold
            )
            prediction, latency_ms = predict_image(
                detector.model, image, detector.device, threshold
            )
            return detector, prediction, latency_ms

    @contextmanager
    def hold_detector(self, model_id: str):
        """Giữ model cố định trong một phiên xử lý dài, ví dụ một video.

        Khóa được giữ trong toàn bộ phiên để một yêu cầu khác không thể thay
        checkpoint giữa các frame. Nhờ vậy GPU luôn chỉ chứa một model và video
        nhất quán với model người dùng đã chọn.
        """
        with self._lock:
            yield self._get_locked(model_id)

    def predict_loaded(
        self,
        detector: LoadedDetector,
        image: Image.Image,
        confidence_threshold: float | Mapping[int, float] | None = None,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Suy luận bằng detector đang được giữ bởi :meth:`hold_detector`."""
        threshold = (
            detector.default_thresholds
            if confidence_threshold is None
            else confidence_threshold
        )
        return predict_image(detector.model, image, detector.device, threshold)

    def release(self) -> None:
        with self._lock:
            self._release_locked()
