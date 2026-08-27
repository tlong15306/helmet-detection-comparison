"""Pipeline suy luận dùng chung cho Faster R-CNN và RetinaNet."""

from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms import functional as F

from .evaluate import (
    build_evaluation_model,
    class_names_from_config,
    load_checkpoint_into_model,
)
from .utils import load_config, resolve_project_path


CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (59, 130, 246),   # BikeWithRider
    2: (239, 68, 68),    # NoHelmet
    3: (34, 197, 94),    # Helmet
}


def normalize_pil_image(image: Image.Image) -> Image.Image:
    """Sửa EXIF orientation và chuẩn hóa mọi ảnh về RGB."""
    return ImageOps.exif_transpose(image).convert("RGB")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Chuyển PIL RGB thành tensor float32 [3, H, W] trong miền [0, 1]."""
    rgb_image = normalize_pil_image(image)
    tensor = F.pil_to_tensor(rgb_image)
    return F.convert_image_dtype(tensor, torch.float32)


def validate_threshold(confidence_threshold: float) -> float:
    threshold = float(confidence_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold phải nằm trong [0, 1]")
    return threshold


def filter_predictions(
    prediction: Mapping[str, torch.Tensor], confidence_threshold: float
) -> dict[str, torch.Tensor]:
    """Lọc đồng bộ boxes, labels và scores theo điều kiện score >= threshold."""
    threshold = validate_threshold(confidence_threshold)
    required = ("boxes", "labels", "scores")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"Prediction thiếu trường bắt buộc: {missing}")

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    if not all(isinstance(value, torch.Tensor) for value in (boxes, labels, scores)):
        raise TypeError("boxes, labels và scores phải là torch.Tensor")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes phải có dạng [N, 4]")
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels và scores phải có dạng [N]")
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("boxes, labels và scores phải có cùng số phần tử")

    keep = scores >= threshold
    return {
        "boxes": boxes[keep].detach().cpu(),
        "labels": labels[keep].detach().cpu(),
        "scores": scores[keep].detach().cpu(),
    }


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    confidence_threshold: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Chạy inference batch size 1 và trả prediction đã lọc cùng latency mili-giây.

    Latency bao gồm chuyển tensor lên thiết bị, model forward và hậu xử lý NMS
    của Torchvision; không bao gồm đọc tệp hoặc vẽ giao diện.
    """
    threshold = validate_threshold(confidence_threshold)
    tensor = image_to_tensor(image)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    device_tensor = tensor.to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        output = model([device_tensor])[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return filter_predictions(output, threshold), latency_ms


def summarize_detections(
    prediction: Mapping[str, torch.Tensor], class_names: Mapping[int, str]
) -> dict[str, int]:
    """Đếm detection theo đúng ánh xạ lớp cấu hình."""
    summary = {str(name): 0 for class_id, name in class_names.items() if int(class_id) != 0}
    labels = prediction.get("labels")
    if labels is None:
        raise ValueError("Prediction thiếu labels")
    for raw_label in labels.tolist():
        class_id = int(raw_label)
        if class_id not in class_names:
            raise ValueError(f"Prediction có nhãn không xác định: {class_id}")
        summary[str(class_names[class_id])] += 1
    return summary


def prediction_records(
    prediction: Mapping[str, torch.Tensor], class_names: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Chuyển prediction thành JSON metadata an toàn cho frontend."""
    records: list[dict[str, Any]] = []
    for box, label, score in zip(
        prediction["boxes"].tolist(),
        prediction["labels"].tolist(),
        prediction["scores"].tolist(),
        strict=True,
    ):
        class_id = int(label)
        if class_id not in class_names:
            raise ValueError(f"Prediction có nhãn không xác định: {class_id}")
        records.append(
            {
                "class_id": class_id,
                "class_name": str(class_names[class_id]),
                "confidence": round(float(score), 6),
                "box": [round(float(value), 2) for value in box],
            }
        )
    return records


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    prediction: Mapping[str, torch.Tensor],
    class_names: Mapping[int, str],
    colors: Mapping[int, tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Vẽ bounding box, tên lớp và confidence trên bản sao ảnh RGB."""
    result = normalize_pil_image(image).copy()
    draw = ImageDraw.Draw(result)
    palette = dict(CLASS_COLORS if colors is None else colors)
    font_size = max(13, min(26, round(min(result.size) * 0.026)))
    font = _load_font(font_size)
    line_width = max(2, round(min(result.size) * 0.004))

    for box, label, score in zip(
        prediction["boxes"].tolist(),
        prediction["labels"].tolist(),
        prediction["scores"].tolist(),
        strict=True,
    ):
        class_id = int(label)
        if class_id not in class_names:
            raise ValueError(f"Prediction có nhãn không xác định: {class_id}")
        color = palette.get(class_id, (245, 158, 11))
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = max(0.0, min(x1, result.width - 1))
        y1 = max(0.0, min(y1, result.height - 1))
        x2 = max(x1, min(x2, result.width - 1))
        y2 = max(y1, min(y2, result.height - 1))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        text = f"{class_names[class_id]} {float(score):.2f}"
        text_box = draw.textbbox((0, 0), text, font=font, stroke_width=0)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        pad_x, pad_y = 6, 4
        label_top = max(0.0, y1 - text_height - 2 * pad_y)
        label_right = min(float(result.width), x1 + text_width + 2 * pad_x)
        draw.rectangle((x1, label_top, label_right, y1), fill=color)
        draw.text((x1 + pad_x, label_top + pad_y), text, fill="white", font=font)
    return result


def encode_png(image: Image.Image) -> bytes:
    stream = BytesIO()
    normalize_pil_image(image).save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector inference on one image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng PyTorch không nhận diện được GPU")
    model = build_evaluation_model(config, device)
    checkpoint = resolve_project_path(args.checkpoint)
    metadata = load_checkpoint_into_model(model, checkpoint, device)
    with Image.open(resolve_project_path(args.input)) as source:
        image = normalize_pil_image(source)
        prediction, latency_ms = predict_image(model, image, device, args.threshold)
        result = draw_detections(image, prediction, class_names_from_config(config))
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(json.dumps({"output": str(output), "latency_ms": latency_ms, "checkpoint": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
