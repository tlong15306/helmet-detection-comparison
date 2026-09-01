"""Fine-tune detector theo cùng pipeline cho Faster R-CNN và RetinaNet.

Tập ``test`` không xuất hiện trong mô-đun này. Validation chỉ dùng để theo dõi
quá trình fine-tune và chọn checkpoint; đánh giá cuối cùng phải chạy qua
``src.evaluate`` sau khi cấu hình được chốt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from .dataset import CocoBoxDataset, collate_fn
from .metrics import DetectionEvaluator
from .models import build_model
from .transforms import build_transforms
from .utils import load_config, resolve_project_path, set_seed


CHECKPOINT_FORMAT_VERSION = 1
SUPPORTED_RUN_TYPES = {"baseline", "pilot", "finetune", "smoke"}


def _configure_console_encoding() -> None:
    """Cho phép log tiếng Việt khi chạy từ PowerShell Windows dùng CP1252."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh mà không ghi đè cấu hình thí nghiệm."""
    parser = argparse.ArgumentParser(description="Fine-tune a Torchvision detector")
    parser.add_argument("--config", required=True, help="Tệp YAML cấu hình thí nghiệm")
    parser.add_argument(
        "--resume", default=None, help="Checkpoint last.pth để tiếp tục thí nghiệm"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy tối đa một batch train và validation để kiểm tra pipeline",
    )
    parser.add_argument(
        "--smoke-output-dir",
        default=None,
        help="Thư mục artifact riêng cho smoke test; không dùng checkpoint official.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Ghi đè lựa chọn thiết bị trong config (mặc định: auto)",
    )
    return parser.parse_args()


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Cấu hình thiếu mapping bắt buộc: {key}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Kiểm tra thông tin cần có trước khi tạo model hoặc DataLoader."""
    model = _require_mapping(config, "model")
    data = _require_mapping(config, "data")
    training = _require_mapping(config, "training")
    runtime = _require_mapping(config, "runtime")
    output = _require_mapping(config, "output")
    classes = _require_mapping(config, "classes")

    required_paths = ("image_root", "train_annotations", "val_annotations")
    missing_paths = [key for key in required_paths if not data.get(key)]
    if missing_paths:
        raise ValueError(f"Thiếu đường dẫn dữ liệu: {', '.join(missing_paths)}")
    if not model.get("name") or int(model.get("num_classes", 0)) < 2:
        raise ValueError("model.name và model.num_classes (>= 2) là bắt buộc")
    if int(model["num_classes"]) != len(classes):
        raise ValueError(
            "model.num_classes phải bằng số nhãn trong classes, bao gồm background"
        )
    if int(training.get("epochs", 0)) < 1 or int(training.get("batch_size", 0)) < 1:
        raise ValueError("training.epochs và training.batch_size phải lớn hơn 0")
    if int(training.get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError("gradient_accumulation_steps phải lớn hơn 0")
    initial_checkpoint = training.get("initial_checkpoint")
    if initial_checkpoint is not None and not isinstance(initial_checkpoint, str):
        raise ValueError("training.initial_checkpoint phải là đường dẫn chuỗi nếu được khai báo")
    if not isinstance(training.get("freeze_backbone_body", False), bool):
        raise ValueError("training.freeze_backbone_body phải là true hoặc false")
    early_stop = training.get("early_stop")
    if early_stop is not None:
        if not isinstance(early_stop, dict):
            raise ValueError("training.early_stop phải là mapping nếu được khai báo")
        if int(early_stop.get("patience", 2)) < 1:
            raise ValueError("training.early_stop.patience phải lớn hơn 0")
        if float(early_stop.get("tolerance", 0.0)) < 0:
            raise ValueError("training.early_stop.tolerance không được âm")
        protected = early_stop.get("protected_classes", ["NoHelmet"])
        if not isinstance(protected, list) or not all(isinstance(name, str) for name in protected):
            raise ValueError("training.early_stop.protected_classes phải là list chuỗi")
    source_mix = config.get("source_mix")
    if source_mix is not None:
        if not isinstance(source_mix, dict):
            raise ValueError("source_mix phải là mapping nếu được cấu hình")
        if bool(source_mix.get("enabled", False)):
            ratio = float(source_mix.get("vietnam_ratio", 0.0))
            if not 0.0 < ratio < 1.0:
                raise ValueError("source_mix.vietnam_ratio phải nằm trong (0, 1)")
            if not isinstance(source_mix.get("vietnam_source_dataset"), str):
                raise ValueError("source_mix.vietnam_source_dataset phải là chuỗi")
            if bool(config.get("sampling", {}).get("enabled", False)):
                raise ValueError("Không thể bật đồng thời source_mix và weighted sampling")
    targeted_source_mix = config.get("targeted_source_mix")
    if targeted_source_mix is not None:
        if not isinstance(targeted_source_mix, dict):
            raise ValueError("targeted_source_mix phải là mapping nếu được cấu hình")
        if bool(targeted_source_mix.get("enabled", False)):
            ratio = float(targeted_source_mix.get("vietnam_ratio", 0.0))
            if not 0.0 < ratio < 1.0:
                raise ValueError("targeted_source_mix.vietnam_ratio phải nằm trong (0, 1)")
            if not isinstance(targeted_source_mix.get("vietnam_source_dataset"), str):
                raise ValueError(
                    "targeted_source_mix.vietnam_source_dataset phải là chuỗi"
                )
            if int(targeted_source_mix.get("focus_category_id", 0)) < 1:
                raise ValueError(
                    "targeted_source_mix.focus_category_id phải là số nguyên dương"
                )
            if bool(source_mix and source_mix.get("enabled", False)):
                raise ValueError("Không thể bật đồng thời source_mix và targeted_source_mix")
            if bool(config.get("sampling", {}).get("enabled", False)):
                raise ValueError(
                    "Không thể bật đồng thời targeted_source_mix và weighted sampling"
                )
    if int(runtime.get("num_workers", 0)) < 0:
        raise ValueError("runtime.num_workers không được âm")
    if not output.get("best_checkpoint") or not output.get("last_checkpoint"):
        raise ValueError("Cấu hình phải chỉ định best_checkpoint và last_checkpoint")
    run = config.get("run", {})
    if run and not isinstance(run, dict):
        raise ValueError("run phải là mapping nếu được khai báo")
    run_type = str(run.get("type", "baseline")).lower()
    if run_type not in {"baseline", "pilot", "finetune"}:
        raise ValueError("run.type chỉ có thể là baseline, pilot hoặc finetune")
    if int(run.get("progress_every_batches", 0)) < 0:
        raise ValueError("run.progress_every_batches không được âm")


def choose_device(requested: str, configured: str = "cuda") -> torch.device:
    """Chọn CUDA khi khả dụng, nhưng vẫn cho phép toàn bộ pipeline chạy CPU."""
    preference = configured if requested == "auto" else requested
    if preference not in {"auto", "cuda", "cpu"}:
        raise ValueError("Thiết bị chỉ có thể là auto, cuda hoặc cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        print("CUDA không khả dụng; chuyển sang CPU.")
        return torch.device("cpu")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def run_type_from_config(config: dict[str, Any], smoke_test: bool = False) -> str:
    """Xác định loại lần chạy; smoke luôn ghi đè cấu hình YAML."""
    if smoke_test:
        return "smoke"
    return str(config.get("run", {}).get("type", "baseline")).lower()


def build_detector_from_config(config: dict[str, Any]) -> nn.Module:
    """Tạo detector, bảo đảm tham số weights trong YAML được sử dụng."""
    model = config["model"]
    image = config["image"]
    return build_model(
        model["name"],
        num_classes=int(model["num_classes"]),
        min_size=int(image.get("min_size", 512)),
        max_size=int(image.get("max_size", 768)),
        trainable_backbone_layers=int(model.get("trainable_backbone_layers", 3)),
        weights=model.get("weights", "DEFAULT"),
    )


def image_sampling_weights(
    dataset: CocoBoxDataset,
    *,
    class_weights: dict[int, float],
    small_object_ratio: float = 0.005,
    small_object_boost: float = 0.4,
    max_weight: float = 2.5,
) -> torch.Tensor:
    """Tạo trọng số ảnh để ưu tiên lớp hiếm và vùng đầu nhỏ trong train.

    Trọng số chỉ quyết định tần suất lấy ảnh, không sửa annotation. Mỗi ảnh
    nhận trọng số lớp lớn nhất trong ảnh và thêm boost một lần nếu chứa
    Helmet/NoHelmet có diện tích tương đối nhỏ.
    """
    if small_object_ratio < 0 or small_object_boost < 0 or max_weight <= 0:
        raise ValueError("Thông số sampling phải không âm và max_weight > 0")
    if any(float(weight) <= 0 for weight in class_weights.values()):
        raise ValueError("class_weights phải lớn hơn 0")

    weights: list[float] = []
    for image_info in dataset.images:
        image_id = int(image_info["id"])
        annotations = dataset.annotations_by_image.get(image_id, [])
        weight = 1.0
        has_small_head = False
        image_area = float(image_info.get("width", 0)) * float(
            image_info.get("height", 0)
        )
        for annotation in annotations:
            class_id = int(annotation["category_id"])
            weight = max(weight, float(class_weights.get(class_id, 1.0)))
            if class_id not in {2, 3} or image_area <= 0:
                continue
            x, y, width, height = map(float, annotation["bbox"])
            del x, y
            area = float(annotation.get("area", width * height))
            if area / image_area < small_object_ratio:
                has_small_head = True
        if has_small_head:
            weight += small_object_boost
        weights.append(min(weight, max_weight))
    return torch.tensor(weights, dtype=torch.double)


class VietnamPrioritySampler(Sampler[int]):
    """Giữ toàn bộ ảnh Việt Nam, luân phiên một phần ảnh gốc mỗi epoch.

    Sampler không lặp ảnh Việt Nam trong một epoch và không dùng weighted
    sampling. Ảnh EdgeVision được chọn lại theo seed ở mỗi epoch để mô hình vẫn
    nhìn thấy dữ liệu gốc đa dạng, trong khi tỷ lệ Việt Nam được ưu tiên.
    """

    def __init__(
        self,
        images: Sequence[Mapping[str, Any]],
        *,
        vietnam_source_dataset: str,
        vietnam_ratio: float,
        seed: int,
    ) -> None:
        if not 0.0 < vietnam_ratio < 1.0:
            raise ValueError("vietnam_ratio phải nằm trong (0, 1)")
        self.vietnam_indices = [
            index
            for index, image in enumerate(images)
            if str(image.get("source_dataset")) == vietnam_source_dataset
        ]
        self.original_indices = [
            index
            for index, image in enumerate(images)
            if str(image.get("source_dataset")) != vietnam_source_dataset
        ]
        if not self.vietnam_indices or not self.original_indices:
            raise ValueError(
                "VietnamPrioritySampler cần có cả ảnh Việt Nam và ảnh gốc"
            )
        self.vietnam_count = len(self.vietnam_indices)
        self.original_count = max(
            1, round(self.vietnam_count * (1.0 - vietnam_ratio) / vietnam_ratio)
        )
        if self.original_count > len(self.original_indices):
            raise ValueError(
                "Không đủ ảnh gốc để đạt vietnam_ratio mà không lặp ảnh gốc"
            )
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.vietnam_count + self.original_count

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        vietnam = list(self.vietnam_indices)
        original = rng.sample(self.original_indices, self.original_count)
        rng.shuffle(vietnam)
        rng.shuffle(original)

        vietnam_used = original_used = 0
        for position in range(1, len(self) + 1):
            expected_vietnam = round(position * self.vietnam_count / len(self))
            if vietnam_used < expected_vietnam:
                yield vietnam[vietnam_used]
                vietnam_used += 1
            else:
                yield original[original_used]
                original_used += 1


class FocusedVietnamSampler(Sampler[int]):
    """Chỉ lấy ảnh Việt Nam có một lớp đích, không lặp lại ảnh trong epoch.

    Dùng khi dữ liệu Việt Nam bị lệch mạnh về lớp Helmet: toàn bộ ảnh Việt Nam
    chứa lớp đích được đưa vào một epoch, còn ảnh EdgeVision được lấy ngẫu nhiên
    không hoàn lại để giữ tỷ lệ dữ liệu gốc. Đây không phải weighted sampling.
    """

    def __init__(
        self,
        images: Sequence[Mapping[str, Any]],
        annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
        *,
        vietnam_source_dataset: str,
        focus_category_id: int,
        vietnam_ratio: float,
        seed: int,
    ) -> None:
        if not 0.0 < vietnam_ratio < 1.0:
            raise ValueError("vietnam_ratio phải nằm trong (0, 1)")
        self.vietnam_indices = [
            index
            for index, image in enumerate(images)
            if str(image.get("source_dataset")) == vietnam_source_dataset
            and any(
                int(annotation["category_id"]) == focus_category_id
                for annotation in annotations_by_image.get(int(image["id"]), [])
            )
        ]
        self.original_indices = [
            index
            for index, image in enumerate(images)
            if str(image.get("source_dataset")) != vietnam_source_dataset
        ]
        if not self.vietnam_indices or not self.original_indices:
            raise ValueError(
                "FocusedVietnamSampler cần có ảnh Việt Nam lớp đích và ảnh gốc"
            )
        self.vietnam_count = len(self.vietnam_indices)
        self.original_count = max(
            1, round(self.vietnam_count * (1.0 - vietnam_ratio) / vietnam_ratio)
        )
        if self.original_count > len(self.original_indices):
            raise ValueError(
                "Không đủ ảnh gốc để đạt vietnam_ratio mà không lặp ảnh gốc"
            )
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.vietnam_count + self.original_count

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        vietnam = list(self.vietnam_indices)
        original = rng.sample(self.original_indices, self.original_count)
        rng.shuffle(vietnam)
        rng.shuffle(original)

        vietnam_used = original_used = 0
        for position in range(1, len(self) + 1):
            expected_vietnam = round(position * self.vietnam_count / len(self))
            if vietnam_used < expected_vietnam:
                yield vietnam[vietnam_used]
                vietnam_used += 1
            else:
                yield original[original_used]
                original_used += 1


def build_loaders(config: dict[str, Any], smoke_test: bool = False) -> tuple[DataLoader, DataLoader]:
    """Tạo train/validation loader cùng image root và không đọc tập test."""
    data = config["data"]
    augmentation = config.get("augmentation", {})
    runtime = config["runtime"]
    image_root = resolve_project_path(data["image_root"])
    train_file = resolve_project_path(data["train_annotations"])
    val_file = resolve_project_path(data["val_annotations"])

    for required in (image_root, train_file, val_file):
        if not required.exists():
            raise FileNotFoundError(
                f"Chưa tìm thấy {required}. Hãy kiểm tra dataset và split trước khi train."
            )

    train_dataset = CocoBoxDataset(
        image_root,
        train_file,
        transforms=build_transforms(
            train=True,
            horizontal_flip_probability=float(
                augmentation.get("horizontal_flip_probability", 0.5)
            ),
            color_jitter_probability=float(
                augmentation.get("color_jitter_probability", 0.0)
            ),
            brightness=float(augmentation.get("brightness", 0.2)),
            contrast=float(augmentation.get("contrast", 0.2)),
            saturation=float(augmentation.get("saturation", 0.15)),
            hue=float(augmentation.get("hue", 0.02)),
        ),
    )
    val_dataset = CocoBoxDataset(
        image_root, val_file, transforms=build_transforms(train=False)
    )
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Train và validation split đều phải có ít nhất một ảnh")

    workers = 0 if smoke_test else int(runtime.get("num_workers", 0))
    common = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    sampling = config.get("sampling", {})
    source_mix = config.get("source_mix", {})
    targeted_source_mix = config.get("targeted_source_mix", {})
    sampler = None
    if bool(targeted_source_mix.get("enabled", False)):
        sampler = FocusedVietnamSampler(
            train_dataset.images,
            train_dataset.annotations_by_image,
            vietnam_source_dataset=str(targeted_source_mix["vietnam_source_dataset"]),
            focus_category_id=int(targeted_source_mix["focus_category_id"]),
            vietnam_ratio=float(targeted_source_mix["vietnam_ratio"]),
            seed=int(config.get("project", {}).get("seed", 42)),
        )
    elif bool(source_mix.get("enabled", False)):
        sampler = VietnamPrioritySampler(
            train_dataset.images,
            vietnam_source_dataset=str(source_mix["vietnam_source_dataset"]),
            vietnam_ratio=float(source_mix["vietnam_ratio"]),
            seed=int(config.get("project", {}).get("seed", 42)),
        )
    elif bool(sampling.get("enabled", False)) and not smoke_test:
        configured_class_weights = {
            int(class_id): float(weight)
            for class_id, weight in sampling.get("class_weights", {}).items()
        }
        weights = image_sampling_weights(
            train_dataset,
            class_weights=configured_class_weights,
            small_object_ratio=float(sampling.get("small_object_ratio", 0.005)),
            small_object_boost=float(sampling.get("small_object_boost", 0.4)),
            max_weight=float(sampling.get("max_weight", 2.5)),
        )
        generator = torch.Generator().manual_seed(
            int(config.get("project", {}).get("seed", 42))
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    return (
        DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def build_optimizer(model: nn.Module, training: dict[str, Any]) -> Optimizer:
    """Khởi tạo optimizer đã công bố trong file cấu hình."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = str(training.get("optimizer", "sgd")).lower()
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(training.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Optimizer không được hỗ trợ: {name}")


def configure_trainable_parameters(model: nn.Module, training: dict[str, Any]) -> dict[str, int]:
    """Đóng băng phần thân backbone cho pilot, vẫn học FPN và detection heads."""
    if bool(training.get("freeze_backbone_body", False)):
        backbone = getattr(model, "backbone", None)
        body = getattr(backbone, "body", None)
        if body is None:
            raise ValueError("Model không có backbone.body để đóng băng")
        for parameter in body.parameters():
            parameter.requires_grad = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    if trainable == 0:
        raise ValueError("Không còn tham số trainable sau khi áp dụng cấu hình đóng băng")
    return {"trainable": trainable, "frozen": frozen}


def build_scheduler(
    optimizer: Optimizer, training: dict[str, Any]
) -> LRScheduler | None:
    """Tạo scheduler theo config; None nghĩa là không đổi learning rate."""
    name = str(training.get("scheduler", "none")).lower()
    if name in {"none", "null", ""}:
        return None
    if name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training["scheduler_step_size"]),
            gamma=float(training["scheduler_gamma"]),
        )
    raise ValueError(f"Scheduler không được hỗ trợ: {name}")


def validation_regressed(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    primary_metric: str,
    protected_classes: Sequence[str],
    tolerance: float = 0.0,
) -> bool:
    """True khi mAP hoặc AP lớp bảo vệ giảm so với lần validation trước.

    Dùng riêng cho early-stop của fine-tune. Checkpoint tốt nhất vẫn được chọn
    độc lập theo primary metric; hàm này không đọc hay động tới test split.
    """
    if previous is None or current is None:
        return False
    if float(current[primary_metric]) + tolerance < float(previous[primary_metric]):
        return True
    for class_name in protected_classes:
        try:
            before = float(previous["per_class"][class_name]["ap_50_95"])
            after = float(current["per_class"][class_name]["ap_50_95"])
        except KeyError as error:
            raise ValueError(f"early_stop không tìm thấy metric lớp {class_name}") from error
        if after + tolerance < before:
            return True
    return False


def _move_targets(targets: Iterable[dict[str, torch.Tensor]], device: torch.device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    accumulation_steps: int = 1,
    max_batches: int | None = None,
    progress_every_batches: int = 0,
) -> dict[str, float]:
    """Cập nhật trọng số trên train split và trả average loss có thể ghi log."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    batches = 0
    epoch_started = time.perf_counter()
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = _move_targets(targets, device)
        with _autocast_context(device, use_amp):
            losses = model(images, targets)
            if not isinstance(losses, dict):
                raise TypeError("Model ở chế độ train phải trả về dictionary loss")
            loss = sum(value for value in losses.values())

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss không hữu hạn ở batch {batch_index}: {loss.item()}")
        scaler.scale(loss / accumulation_steps).backward()
        if batch_index % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_total += float(loss.detach().item())
        batches += 1
        if progress_every_batches and batches % progress_every_batches == 0:
            elapsed_seconds = time.perf_counter() - epoch_started
            average_loss = loss_total / batches
            memory_text = ""
            if device.type == "cuda":
                peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
                memory_text = f", peak_vram={peak_mib:.0f} MiB"
            total_text = str(total_batches) if total_batches is not None else "?"
            print(
                f"Train batch {batches}/{total_text}: loss_avg={average_loss:.5f}, "
                f"elapsed={elapsed_seconds:.1f}s{memory_text}"
            )
        if max_batches is not None and batches >= max_batches:
            break

    if batches and batches % accumulation_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if not batches:
        raise ValueError("Train loader không có batch nào")
    return {"train_loss": loss_total / batches, "train_batches": float(batches)}


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: dict[int, str],
    iou_threshold: float,
    confidence_threshold: float | None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Đánh giá trên validation split; không cập nhật trọng số hoặc scheduler."""
    model.eval()
    evaluator = DetectionEvaluator(
        class_names,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )
    batches = 0
    for images, targets in loader:
        inputs = [image.to(device, non_blocking=True) for image in images]
        predictions = model(inputs)
        evaluator.update(predictions, targets)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if not batches:
        raise ValueError("Validation loader không có batch nào")
    result = evaluator.compute()
    result["validation_batches"] = batches
    return result


def _checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    config: dict[str, Any],
    validation: dict[str, Any] | None,
    *,
    smoke_test: bool = False,
    run_type: str = "baseline",
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "model_name": config["model"]["name"],
        "num_classes": config["model"]["num_classes"],
        "config": copy.deepcopy(config),
        "validation": validation,
        "smoke_test": smoke_test,
        "run_type": run_type,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Ghi checkpoint kèm metadata để tái lập cấu hình đã dùng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    device: torch.device,
    expected_model_name: str,
    expected_run_type: str = "baseline",
) -> int:
    """Khôi phục checkpoint tương thích và trả epoch kế tiếp cần chạy."""
    checkpoint_path = resolve_project_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint resume: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint không đúng định dạng của pipeline hiện tại")
    if payload.get("model_name") != expected_model_name:
        raise ValueError("Checkpoint thuộc kiến trúc khác với config hiện tại")
    if payload.get("smoke_test"):
        raise ValueError("Không được resume train từ checkpoint smoke test")
    checkpoint_run_type = str(payload.get("run_type", "baseline")).lower()
    if checkpoint_run_type != expected_run_type:
        raise ValueError(
            "Checkpoint thuộc loại lần chạy khác: "
            f"checkpoint={checkpoint_run_type}, expected={expected_run_type}"
        )
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload["epoch"]) + 1


def load_initial_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
    expected_model_name: str,
) -> dict[str, Any]:
    """Nạp riêng trọng số model cho fine-tune, không khôi phục optimizer/epoch."""
    checkpoint_path = resolve_project_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint khởi tạo: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint khởi tạo không đúng định dạng pipeline")
    if payload.get("model_name") != expected_model_name:
        raise ValueError("Checkpoint khởi tạo thuộc kiến trúc khác với config hiện tại")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return {
        "path": str(checkpoint_path),
        "sha256": _file_sha256(checkpoint_path),
        "source_epoch": payload.get("epoch"),
        "source_run_type": payload.get("run_type"),
    }


def _read_checkpoint_metric(path: Path, metric_name: str, device: torch.device) -> float:
    """Lấy metric tốt nhất đã lưu để resume không ghi đè checkpoint tốt hơn."""
    if not path.exists():
        return -math.inf
    payload = torch.load(path, map_location=device, weights_only=False)
    validation = payload.get("validation") or {}
    value = validation.get(metric_name)
    return float(value) if value is not None else -math.inf


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)


def _read_history(path: Path) -> list[dict[str, Any]]:
    """Đọc lịch sử trước đó khi resume, lỗi định dạng sẽ không bị che giấu."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        history = json.load(stream)
    if not isinstance(history, list):
        raise ValueError(f"History không có định dạng list: {path}")
    return history


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _smoke_directory(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return resolve_project_path(override)
    output = config["output"]
    configured = output.get("smoke_directory")
    if configured:
        return resolve_project_path(configured)
    model_short_name = str(config["model"]["name"]).replace("_resnet50_fpn_v2", "")
    return resolve_project_path(Path("outputs") / "smoke" / model_short_name)


def _write_json(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _training_data_hashes(config: dict[str, Any]) -> dict[str, str | None]:
    """Fingerprint các đầu vào pipeline train/validation, tuyệt đối không đọc test."""
    data = config["data"]
    tracked_files = {
        "processed_annotations": data.get("processed_annotations"),
        "train_annotations": data["train_annotations"],
        "val_annotations": data["val_annotations"],
        "frozen_split_manifest": data.get("frozen_manifest", "data/splits/frozen_manifest.json"),
        "initial_checkpoint": config["training"].get("initial_checkpoint"),
    }
    hashes: dict[str, str | None] = {}
    for key, relative_path in tracked_files.items():
        if not relative_path:
            hashes[key] = None
            continue
        path = resolve_project_path(relative_path)
        hashes[key] = _file_sha256(path) if path.is_file() else None
    return hashes


def _validate_frozen_training_inputs(config: dict[str, Any]) -> None:
    """Chặn train nếu train/validation/processed khác frozen manifest.

    Hàm này cố ý không đọc test split; frozen manifest đã được tạo bởi công cụ
    kiểm tra đầy đủ ba split trước thời điểm huấn luyện.
    """
    manifest_path = resolve_project_path(
        config["data"].get("frozen_manifest", "data/splits/frozen_manifest.json")
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Thiếu frozen split manifest. Chạy tools/freeze_splits.py trước khi train."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Frozen split manifest không phải JSON hợp lệ: {manifest_path}") from error
    if manifest.get("schema_version") != "frozen-split-1.0":
        raise ValueError("Frozen split manifest có schema không được hỗ trợ")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Frozen split manifest thiếu files")
    expected = {
        "processed_annotations": config["data"].get("processed_annotations"),
        "train": config["data"]["train_annotations"],
        "val": config["data"]["val_annotations"],
    }
    for key, configured_path in expected.items():
        entry = files.get(key)
        if not configured_path or not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"Frozen split manifest thiếu fingerprint cho {key}")
        actual_path = resolve_project_path(configured_path)
        if not actual_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy đầu vào train đã đóng băng: {actual_path}")
        if _file_sha256(actual_path) != entry["sha256"]:
            raise ValueError(
                f"{key} đã thay đổi sau khi đóng băng split. "
                "Tạo lại manifest và hủy artifact train cũ trước khi chạy tiếp."
            )


def _cuda_memory_bytes(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _checkpoint_metadata(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": _file_sha256(path)}


def _run_manifest(
    config: dict[str, Any],
    *,
    run_type: str,
    status: str,
    device: torch.device,
    started_at_utc: str,
    duration_seconds: float,
    history: list[dict[str, Any]],
    best_checkpoint: Path,
    last_checkpoint: Path,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Tạo manifest nhẹ cho một lần train pilot hoặc baseline."""
    return {
        "schema_version": "training-run-1.0",
        "run_type": run_type,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "git_commit": _git_commit(),
        "model": copy.deepcopy(config["model"]),
        "image": copy.deepcopy(config["image"]),
        "training": copy.deepcopy(config["training"]),
        "runtime": {
            "device": str(device),
            "amp": bool(config["runtime"].get("mixed_precision", False)) and device.type == "cuda",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "data_hashes": _training_data_hashes(config),
        "checkpoints": {
            "best": _checkpoint_metadata(best_checkpoint),
            "last": _checkpoint_metadata(last_checkpoint),
        },
        "history": history,
        "cuda_memory_bytes": _cuda_memory_bytes(device),
        "error": error,
    }


def _smoke_manifest(
    config: dict[str, Any],
    device: torch.device,
    checkpoint: Path,
    history: list[dict[str, Any]],
    memory: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "smoke-run-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_test": True,
        "git_commit": _git_commit(),
        "model": copy.deepcopy(config["model"]),
        "image": copy.deepcopy(config["image"]),
        "training": copy.deepcopy(config["training"]),
        "runtime": {
            "device": str(device),
            "amp": bool(config["runtime"].get("mixed_precision", False)) and device.type == "cuda",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "data_hashes": _training_data_hashes(config),
        "checkpoint": {"path": str(checkpoint), "sha256": _file_sha256(checkpoint)},
        "history": history,
        "cuda_memory_bytes": memory,
    }


@torch.inference_mode()
def _smoke_checkpoint_inference(
    config: dict[str, Any], checkpoint: Path, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    """Nạp checkpoint smoke vào model mới và kiểm tra schema prediction."""
    reloaded = build_detector_from_config(
        {**config, "model": {**config["model"], "weights": "NONE"}}
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.eval()
    images, _targets = next(iter(loader))
    predictions = reloaded([image.to(device, non_blocking=True) for image in images])
    if not predictions or not {"boxes", "labels", "scores"}.issubset(predictions[0]):
        raise ValueError("Inference sau reload không trả boxes, labels, scores")
    first = predictions[0]
    if not all(torch.isfinite(first[key]).all() for key in ("boxes", "scores")):
        raise FloatingPointError("Prediction sau reload có tensor không hữu hạn")
    labels = first["labels"].detach().cpu().tolist()
    valid_labels = {int(key) for key in config["classes"] if int(key) != 0}
    if any(int(label) not in valid_labels for label in labels):
        raise ValueError("Inference sau reload trả label ngoài class mapping")
    return {"prediction_count": int(first["boxes"].shape[0]), "schema_valid": True}


def run_training(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    smoke_test: bool = False,
    requested_device: str = "auto",
    smoke_output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Điều phối full training; dùng validation, tuyệt đối không gọi test split."""
    validate_config(config)
    run_type = run_type_from_config(config, smoke_test=smoke_test)
    if run_type not in SUPPORTED_RUN_TYPES:
        raise ValueError(f"Loại lần chạy không được hỗ trợ: {run_type}")
    _validate_frozen_training_inputs(config)
    set_seed(int(config["project"].get("seed", 42)))
    device = choose_device(requested_device, str(config["runtime"].get("device", "cuda")))
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.perf_counter()
    smoke_dir: Path | None = None
    if smoke_test:
        frozen_manifest = resolve_project_path(
            config["data"].get("frozen_manifest", "data/splits/frozen_manifest.json")
        )
        if not frozen_manifest.is_file():
            raise FileNotFoundError(
                "Thiếu frozen split manifest. Chạy tools/freeze_splits.py trước smoke test."
            )
        smoke_dir = _smoke_directory(config, smoke_output_dir)
        smoke_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_loaders(config, smoke_test=smoke_test)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = build_detector_from_config(config).to(device)
    initial_checkpoint = config["training"].get("initial_checkpoint")
    if resume and initial_checkpoint:
        print("Resume từ checkpoint hiện có; bỏ qua training.initial_checkpoint.")
    elif initial_checkpoint:
        initialization = load_initial_checkpoint(
            initial_checkpoint,
            model,
            device,
            str(config["model"]["name"]),
        )
        print(f"Khởi tạo fine-tune từ {initialization['path']}")
    parameter_counts = configure_trainable_parameters(model, config["training"])
    print(
        "Tham số: "
        f"trainable={parameter_counts['trainable']:,}, frozen={parameter_counts['frozen']:,}"
    )
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    use_amp = bool(config["runtime"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output = config["output"]
    best_path = resolve_project_path(output["best_checkpoint"])
    last_path = resolve_project_path(output["last_checkpoint"])
    history_path = resolve_project_path(output["directory"]) / "logs" / "history.json"
    run_directory = resolve_project_path(output["directory"])
    smoke_checkpoint: Path | None = None
    if smoke_dir is not None:
        smoke_checkpoint = smoke_dir / "checkpoint.pth"
        history_path = smoke_dir / "history.json"
    class_names = {int(key): str(value) for key, value in config["classes"].items()}
    evaluation = config["evaluation"]
    primary_metric = str(evaluation.get("primary_metric", "map_50_95"))
    start_epoch = 1
    best_metric = -math.inf
    history: list[dict[str, Any]] = []
    if resume:
        start_epoch = load_resume_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            device,
            config["model"]["name"],
            expected_run_type=run_type,
        )
        if not smoke_test:
            best_metric = _read_checkpoint_metric(best_path, primary_metric, device)
            history = _read_history(history_path)

    epochs = 1 if smoke_test else int(config["training"]["epochs"])
    stop_epoch = start_epoch + epochs - 1 if smoke_test else epochs
    validation_every = int(config["training"].get("validate_every_epochs", 1))
    progress_every_batches = int(config.get("run", {}).get("progress_every_batches", 0))
    early_stop_config = config["training"].get("early_stop") or {}
    early_stop_enabled = bool(early_stop_config.get("enabled", False)) and not smoke_test
    early_stop_patience = int(early_stop_config.get("patience", 2))
    early_stop_tolerance = float(early_stop_config.get("tolerance", 0.0))
    early_stop_classes = list(early_stop_config.get("protected_classes", ["NoHelmet"]))
    previous_validation: Mapping[str, Any] | None = None
    regression_streak = 0
    for prior_record in history:
        prior_validation = prior_record.get("validation")
        if prior_validation is None:
            continue
        if validation_regressed(
            previous_validation,
            prior_validation,
            primary_metric=primary_metric,
            protected_classes=early_stop_classes,
            tolerance=early_stop_tolerance,
        ):
            regression_streak += 1
        else:
            regression_streak = 0
        previous_validation = prior_validation
    early_stop_triggered = False

    print(f"Huấn luyện {config['model']['name']} trên {device.type}; AMP={use_amp}")
    for epoch in range(start_epoch, stop_epoch + 1):
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_summary = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
            max_batches=1 if smoke_test else None,
            progress_every_batches=progress_every_batches if not smoke_test else 0,
        )
        validation: dict[str, Any] | None = None
        regressed = False
        if epoch % validation_every == 0 or epoch == stop_epoch:
            validation = validate(
                model,
                val_loader,
                device,
                class_names,
                iou_threshold=float(evaluation.get("precision_recall_iou_threshold", 0.5)),
                confidence_threshold=evaluation.get("confidence_threshold"),
                max_batches=1 if smoke_test else None,
            )
            if primary_metric not in validation:
                raise ValueError(f"primary_metric không có trong validation: {primary_metric}")
            if validation[primary_metric] > best_metric:
                best_metric = float(validation[primary_metric])
                if not smoke_test:
                    save_checkpoint(
                        best_path,
                        _checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            config,
                            validation,
                            run_type=run_type,
                        ),
                    )
            regressed = validation_regressed(
                previous_validation,
                validation,
                primary_metric=primary_metric,
                protected_classes=early_stop_classes,
                tolerance=early_stop_tolerance,
            )
            regression_streak = regression_streak + 1 if regressed else 0
            previous_validation = validation
            early_stop_triggered = early_stop_enabled and regression_streak >= early_stop_patience

        # Smoke test chỉ xác nhận forward/backward/validation/checkpoint trong
        # một batch. AMP có thể bỏ qua optimizer step đầu tiên khi tự điều chỉnh
        # loss scale, nên không step scheduler để tránh ghi nhận một lịch LR sai.
        if scheduler is not None and not smoke_test:
            scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
            "cuda_memory_bytes": _cuda_memory_bytes(device),
            **train_summary,
            "validation": validation,
        }
        if early_stop_enabled:
            record["early_stop"] = {
                "enabled": True,
                "regressed": regressed,
                "regression_streak": regression_streak,
                "patience": early_stop_patience,
                "triggered": early_stop_triggered,
                "protected_classes": early_stop_classes,
            }
        history.append(record)
        if not smoke_test:
            save_checkpoint(
                last_path,
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    validation,
                    run_type=run_type,
                ),
            )
            _write_history(history_path, history)
        loss_text = f"loss={record['train_loss']:.5f}"
        metric_text = (
            f", {primary_metric}={validation[primary_metric]:.5f}" if validation else ""
        )
        print(f"Epoch {epoch}/{stop_epoch}: {loss_text}{metric_text}")
        if early_stop_triggered:
            print(
                "Dừng sớm: validation mAP hoặc AP lớp bảo vệ giảm "
                f"{regression_streak} lần liên tiếp."
            )
            break

    if smoke_dir is not None and smoke_checkpoint is not None:
        final_validation = history[-1].get("validation") if history else None
        save_checkpoint(
            smoke_checkpoint,
            _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                stop_epoch,
                config,
                final_validation,
                smoke_test=True,
                run_type="smoke",
            ),
        )
        _write_history(history_path, history)
        memory = None
        if device.type == "cuda":
            memory = {
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
        inference = _smoke_checkpoint_inference(config, smoke_checkpoint, val_loader, device)
        manifest = _smoke_manifest(config, device, smoke_checkpoint, history, memory)
        manifest["checkpoint_reload_inference"] = inference
        _write_json(smoke_dir / "run_manifest.json", manifest)
        _write_json(
            smoke_dir / "environment.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
        )

    if run_type in {"pilot", "baseline", "finetune"}:
        manifest = _run_manifest(
            config,
            run_type=run_type,
            status="completed_early_stop" if early_stop_triggered else "completed",
            device=device,
            started_at_utc=started_at_utc,
            duration_seconds=time.perf_counter() - started_monotonic,
            history=history,
            best_checkpoint=best_path,
            last_checkpoint=last_path,
        )
        _write_json(run_directory / "run_manifest.json", manifest)
        _write_json(
            run_directory / "environment.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "run_type": run_type,
            },
        )

    return history


def main() -> None:
    _configure_console_encoding()
    args = parse_args()
    config = load_config(args.config)
    run_training(
        config,
        resume=args.resume,
        smoke_test=args.smoke_test,
        requested_device=args.device,
        smoke_output_dir=args.smoke_output_dir,
    )


if __name__ == "__main__":
    main()
