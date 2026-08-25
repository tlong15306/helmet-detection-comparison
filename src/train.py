"""Fine-tune detector theo cùng pipeline cho Faster R-CNN và RetinaNet.

Tập ``test`` không xuất hiện trong mô-đun này. Validation chỉ dùng để theo dõi
quá trình fine-tune và chọn checkpoint; đánh giá cuối cùng phải chạy qua
``src.evaluate`` sau khi cấu hình được chốt.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .dataset import CocoBoxDataset, collate_fn
from .metrics import DetectionEvaluator
from .models import build_model
from .transforms import build_transforms
from .utils import load_config, resolve_project_path, set_seed


CHECKPOINT_FORMAT_VERSION = 1


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
    if int(runtime.get("num_workers", 0)) < 0:
        raise ValueError("runtime.num_workers không được âm")
    if not output.get("best_checkpoint") or not output.get("last_checkpoint"):
        raise ValueError("Cấu hình phải chỉ định best_checkpoint và last_checkpoint")


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
    return (
        DataLoader(train_dataset, shuffle=True, **common),
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
) -> dict[str, float]:
    """Cập nhật trọng số trên train split và trả average loss có thể ghi log."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    batches = 0

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
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload["epoch"]) + 1


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


def run_training(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    smoke_test: bool = False,
    requested_device: str = "auto",
) -> list[dict[str, Any]]:
    """Điều phối full training; dùng validation, tuyệt đối không gọi test split."""
    validate_config(config)
    set_seed(int(config["project"].get("seed", 42)))
    device = choose_device(requested_device, str(config["runtime"].get("device", "cuda")))
    train_loader, val_loader = build_loaders(config, smoke_test=smoke_test)
    model = build_model(
        config["model"]["name"],
        num_classes=int(config["model"]["num_classes"]),
        min_size=int(config["image"].get("min_size", 512)),
        max_size=int(config["image"].get("max_size", 768)),
        trainable_backbone_layers=int(config["model"].get("trainable_backbone_layers", 3)),
    ).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    use_amp = bool(config["runtime"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output = config["output"]
    best_path = resolve_project_path(output["best_checkpoint"])
    last_path = resolve_project_path(output["last_checkpoint"])
    history_path = resolve_project_path(output["directory"]) / "logs" / "history.json"
    class_names = {int(key): str(value) for key, value in config["classes"].items()}
    evaluation = config["evaluation"]
    primary_metric = str(evaluation.get("primary_metric", "map_50_95"))
    start_epoch = 1
    best_metric = -math.inf
    history: list[dict[str, Any]] = []
    if resume:
        start_epoch = load_resume_checkpoint(
            resume, model, optimizer, scheduler, device, config["model"]["name"]
        )
        if not smoke_test:
            best_metric = _read_checkpoint_metric(best_path, primary_metric, device)
            history = _read_history(history_path)

    epochs = 1 if smoke_test else int(config["training"]["epochs"])
    stop_epoch = start_epoch + epochs - 1 if smoke_test else epochs
    validation_every = int(config["training"].get("validate_every_epochs", 1))

    print(f"Huấn luyện {config['model']['name']} trên {device.type}; AMP={use_amp}")
    for epoch in range(start_epoch, stop_epoch + 1):
        train_summary = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
            max_batches=1 if smoke_test else None,
        )
        validation: dict[str, Any] | None = None
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
                        _checkpoint_payload(model, optimizer, scheduler, epoch, config, validation),
                    )

        if scheduler is not None:
            scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_summary,
            "validation": validation,
        }
        history.append(record)
        if not smoke_test:
            save_checkpoint(
                last_path,
                _checkpoint_payload(model, optimizer, scheduler, epoch, config, validation),
            )
            _write_history(history_path, history)
        loss_text = f"loss={record['train_loss']:.5f}"
        metric_text = (
            f", {primary_metric}={validation[primary_metric]:.5f}" if validation else ""
        )
        print(f"Epoch {epoch}/{stop_epoch}: {loss_text}{metric_text}")

    return history


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(
        config,
        resume=args.resume,
        smoke_test=args.smoke_test,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
