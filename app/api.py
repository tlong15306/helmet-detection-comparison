"""FastAPI cục bộ cho giao diện Helmet Detection AI."""

from __future__ import annotations

import base64
import json
import threading
import warnings
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Mapping

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, UnidentifiedImageError

from app.model_loader import DetectorManager, list_model_metadata
from app.video_jobs import ALLOWED_VIDEO_SUFFIXES, MAX_VIDEO_BYTES, VideoJobManager
from src.infer import (
    draw_detections,
    encode_png,
    normalize_pil_image,
    summarize_detections,
    validate_threshold,
)
from src.postprocess import postprocess_detections
from src.rider_association import load_role_decision_config
from src.role_annotations import VALID_REVIEW_STATUSES, VALID_ROLES, validate_role_tasks
from src.utils import resolve_project_path


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
GENERIC_UPLOAD_TYPES = {None, "", "application/octet-stream"}
DETECTOR_MANAGER = DetectorManager(requested_device="auto")
VIDEO_JOBS = VideoJobManager(DETECTOR_MANAGER)
ROLE_TASKS_PATH = resolve_project_path("data/role_association/annotations/role_dev.pending.json")
ROLE_IMAGE_ROOT = resolve_project_path("data/raw/edgevision/images")
ROLE_TASKS_LOCK = threading.RLock()
RIDER_ASSOCIATION_CONFIG, RIDER_ROLE_CONFIG = load_role_decision_config("configs/rider_association.yaml")

app = FastAPI(
    title="Helmet Detection AI API",
    description="Backend cục bộ cho Faster R-CNN và RetinaNet.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type"],
)


class RoleReviewUpdate(BaseModel):
    reviewer: str
    driver_head_annotation_id: int | None
    head_roles: dict[str, str]
    notes: str | None = None
    status: str = "needs_second_review"


def _api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _read_role_tasks() -> dict[str, Any]:
    if not ROLE_TASKS_PATH.is_file():
        raise FileNotFoundError(f"Chưa có task role_dev: {ROLE_TASKS_PATH}")
    with ROLE_TASKS_PATH.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    validate_role_tasks(payload)
    return payload


def _role_task(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in payload["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def _role_task_preview(task: dict[str, Any]) -> Image.Image:
    source_path = resolve_project_path(task["image_path"])
    try:
        source_path.relative_to(ROLE_IMAGE_ROOT)
    except ValueError as error:
        raise ValueError("Đường dẫn ảnh review nằm ngoài image root được phép") from error
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as source:
        image = normalize_pil_image(source)
    x1, y1, x2, y2 = (float(value) for value in task["bike_box_xyxy"])
    pad_x = max(28.0, (x2 - x1) * 0.45)
    pad_y = max(28.0, (y2 - y1) * 0.45)
    left, top, right, bottom = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(image.width, int(x2 + pad_x)),
        min(image.height, int(y2 + pad_y)),
    )
    preview = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(preview)
    draw.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=(245, 158, 11), width=4)
    colors = {"helmet": (34, 197, 94), "no_helmet": (239, 68, 68)}
    for head in task["heads"]:
        hx1, hy1, hx2, hy2 = (float(value) for value in head["box_xyxy"])
        color = colors[head["helmet_status"]]
        draw.rectangle((hx1 - left, hy1 - top, hx2 - left, hy2 - top), outline=color, width=4)
        draw.text((hx1 - left + 3, max(0, hy1 - top - 14)), f"H{head['annotation_id']}", fill=color)
    return preview


def decode_uploaded_image(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("Tệp ảnh rỗng")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 20 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                source.load()
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Ảnh vượt quá giới hạn 40 triệu pixel")
                return normalize_pil_image(source)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Không thể đọc tệp như ảnh JPG hoặc PNG hợp lệ") from error


def parse_class_thresholds(
    raw_value: str | None,
    model_id: str,
    scalar_threshold: float | None,
) -> float | dict[int, float] | None:
    """Đọc override threshold theo lớp, đồng thời giữ tương thích API cũ."""
    if raw_value is None or not raw_value.strip():
        return scalar_threshold
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError("class_thresholds phải là JSON object hợp lệ") from error
    if not isinstance(payload, Mapping):
        raise ValueError("class_thresholds phải là object theo tên lớp")
    metadata = next((item for item in list_model_metadata() if item["id"] == model_id), None)
    if metadata is None:
        raise ValueError(f"Model không được hỗ trợ: {model_id}")
    result: dict[int, float] = {}
    for raw_class_id, class_name in metadata["classes"].items():
        class_id = int(raw_class_id)
        if class_id == 0:
            continue
        if class_name not in payload:
            raise ValueError(f"Thiếu threshold cho lớp {class_name}")
        result[class_id] = validate_threshold(float(payload[class_name]))
    return result


@app.get("/api/health")
def health() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "status": "ready",
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "loaded_model": DETECTOR_MANAGER.current_model_id,
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    try:
        return {"models": list_model_metadata()}
    except (ValueError, FileNotFoundError) as error:
        raise _api_error(500, "MODEL_CONFIG_ERROR", str(error)) from error


@app.get("/api/role-review/tasks")
def role_review_tasks() -> dict[str, Any]:
    """Danh sách task pending/review cho công cụ gán nhãn cục bộ."""
    try:
        with ROLE_TASKS_LOCK:
            payload = _read_role_tasks()
        tasks = payload["tasks"]
        status_counts = {
            status: sum(task["review"]["status"] == status for task in tasks)
            for status in VALID_REVIEW_STATUSES
        }
        return {
            "schema_version": payload["schema_version"],
            "tasks": tasks,
            "summary": {"tasks": len(tasks), **status_counts},
        }
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise _api_error(503, "ROLE_TASKS_UNAVAILABLE", str(error)) from error


@app.get("/api/role-review/tasks/{task_id}/preview")
def role_review_preview(task_id: str) -> Response:
    try:
        with ROLE_TASKS_LOCK:
            task = _role_task(_read_role_tasks(), task_id)
        return Response(content=encode_png(_role_task_preview(task)), media_type="image/png")
    except KeyError as error:
        raise _api_error(404, "ROLE_TASK_NOT_FOUND", f"Không tìm thấy task: {task_id}") from error
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise _api_error(503, "ROLE_PREVIEW_UNAVAILABLE", str(error)) from error


@app.put("/api/role-review/tasks/{task_id}")
def update_role_review(task_id: str, update: RoleReviewUpdate) -> dict[str, Any]:
    """Lưu review thủ công sau khi validator kiểm tra tính nhất quán."""
    if update.status not in {"reviewed", "needs_second_review"}:
        raise _api_error(422, "INVALID_ROLE_STATUS", "status phải là reviewed hoặc needs_second_review")
    if not update.reviewer.strip():
        raise _api_error(422, "INVALID_REVIEWER", "Cần nhập tên người review")
    if any(role not in VALID_ROLES for role in update.head_roles.values()):
        raise _api_error(422, "INVALID_HEAD_ROLE", "head_roles chỉ được driver, passenger hoặc unknown")
    try:
        with ROLE_TASKS_LOCK:
            payload = _read_role_tasks()
            task = _role_task(payload, task_id)
            if payload.get("team_confirmation", {}).get("status") == "confirmed_by_team":
                raise _api_error(409, "ROLE_DEV_FROZEN", "role_dev đã được nhóm xác nhận và khóa; không sửa qua giao diện")
            task["review"] = {
                "status": update.status,
                "reviewer": update.reviewer.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(),
                "driver_head_annotation_id": update.driver_head_annotation_id,
                "head_roles": update.head_roles,
                "notes": update.notes.strip() if update.notes and update.notes.strip() else None,
            }
            validate_role_tasks(payload)
            temporary = ROLE_TASKS_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(ROLE_TASKS_PATH)
        return {"task": task, "message": "Đã lưu review; cần kiểm tra chéo trước khi dùng để chọn quy tắc."}
    except KeyError as error:
        raise _api_error(404, "ROLE_TASK_NOT_FOUND", f"Không tìm thấy task: {task_id}") from error
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise _api_error(422, "INVALID_ROLE_REVIEW", str(error)) from error


@app.post("/api/infer/image")
def infer_image(
    file: Annotated[UploadFile, File(description="Ảnh JPG hoặc PNG")],
    model_id: Annotated[str, Form()],
    threshold: Annotated[float | None, Form()] = None,
    class_thresholds: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    if file.content_type not in ALLOWED_IMAGE_TYPES | GENERIC_UPLOAD_TYPES:
        raise _api_error(415, "UNSUPPORTED_IMAGE_TYPE", "Chỉ hỗ trợ ảnh JPG, JPEG hoặc PNG")
    if threshold is not None:
        try:
            threshold = validate_threshold(threshold)
        except ValueError as error:
            raise _api_error(422, "INVALID_THRESHOLD", str(error)) from error
    try:
        requested_thresholds = parse_class_thresholds(class_thresholds, model_id, threshold)
    except ValueError as error:
        raise _api_error(422, "INVALID_CLASS_THRESHOLDS", str(error)) from error

    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        image = decode_uploaded_image(content)
    except ValueError as error:
        raise _api_error(400, "INVALID_IMAGE", str(error)) from error
    finally:
        file.file.close()

    try:
        detector, raw_prediction, latency_ms = DETECTOR_MANAGER.predict(
            model_id=model_id,
            image=image,
            confidence_threshold=requested_thresholds,
        )
        postprocess_config = getattr(
            detector,
            "postprocess_config",
            {"head_conflict_iou_threshold": 0.70, "head_confidence_margin": 0.10},
        )
        processed = postprocess_detections(
            raw_prediction,
            detector.class_names,
            RIDER_ASSOCIATION_CONFIG,
            RIDER_ROLE_CONFIG,
            head_conflict_iou_threshold=float(postprocess_config["head_conflict_iou_threshold"]),
            head_confidence_margin=float(postprocess_config["head_confidence_margin"]),
        )
        rendered = draw_detections(
            image,
            processed["prediction"],
            detector.class_names,
            display_annotations=processed["display_annotations"],
        )
        rendered_png = encode_png(rendered)
        effective_thresholds = (
            dict(getattr(detector, "default_thresholds", {}))
            if requested_thresholds is None
            else ({class_id: float(requested_thresholds) for class_id in detector.class_names if class_id != 0}
                  if isinstance(requested_thresholds, float) else dict(requested_thresholds))
        )
        if not effective_thresholds:
            effective_thresholds = {
                int(class_id): float(detector.default_threshold)
                for class_id in detector.class_names
                if int(class_id) != 0
            }
        return {
            "model": {
                "id": detector.model_id,
                "name": detector.display_name,
                "architecture": detector.config["model"]["name"],
            },
            "device": {
                "type": detector.device.type,
                "name": (
                    torch.cuda.get_device_name(detector.device)
                    if detector.device.type == "cuda"
                    else "CPU"
                ),
            },
            "input": {
                "filename": file.filename or "image",
                "width": image.width,
                "height": image.height,
            },
            "threshold": effective_thresholds.get(2, float(detector.default_threshold)),
            "thresholds": {
                detector.class_names[int(class_id)]: value
                for class_id, value in effective_thresholds.items()
            },
            "threshold_source": (
                "validation_default" if requested_thresholds is None
                else "user_override"
            ),
            "latency_ms": round(latency_ms, 3),
            "summary": summarize_detections(processed["prediction"], detector.class_names),
            "detections": processed["detections"],
            "raw_detections": processed["raw_detections"],
            "rider_analysis": processed["rider_analysis"],
            "alerts": processed["alerts"],
            "result_image": "data:image/png;base64," + base64.b64encode(rendered_png).decode("ascii"),
        }
    except ValueError as error:
        raise _api_error(422, "INVALID_REQUEST", str(error)) from error
    except FileNotFoundError as error:
        raise _api_error(503, "CHECKPOINT_NOT_FOUND", str(error)) from error
    except RuntimeError as error:
        message = str(error)
        code = "CUDA_OUT_OF_MEMORY" if "out of memory" in message.lower() else "MODEL_LOAD_ERROR"
        raise _api_error(503, code, message) from error


@app.post("/api/infer/video", status_code=202)
def infer_video(
    file: Annotated[UploadFile, File(description="Video MP4, MOV hoặc AVI")],
    model_id: Annotated[str, Form()],
    threshold: Annotated[float | None, Form()] = None,
    class_thresholds: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise _api_error(415, "UNSUPPORTED_VIDEO_TYPE", "Chỉ hỗ trợ video MP4, MOV hoặc AVI")
    if threshold is not None:
        try:
            threshold = validate_threshold(threshold)
        except ValueError as error:
            raise _api_error(422, "INVALID_THRESHOLD", str(error)) from error
    try:
        requested_thresholds = parse_class_thresholds(class_thresholds, model_id, threshold)
    except ValueError as error:
        raise _api_error(422, "INVALID_CLASS_THRESHOLDS", str(error)) from error
    try:
        content = file.file.read(MAX_VIDEO_BYTES + 1)
        metadata = next(model for model in list_model_metadata() if model["id"] == model_id)
        effective_thresholds = requested_thresholds
        if effective_thresholds is None:
            effective_thresholds = {
                int(class_id): float(metadata["default_thresholds"][class_name])
                for class_id, class_name in metadata["classes"].items()
                if int(class_id) != 0
            }
        job = VIDEO_JOBS.submit(content, file.filename or "video.mp4", model_id, effective_thresholds)
        return VIDEO_JOBS.payload(job.job_id) or {}
    except StopIteration as error:
        raise _api_error(422, "INVALID_REQUEST", f"Model không được hỗ trợ: {model_id}") from error
    except ValueError as error:
        raise _api_error(400, "INVALID_VIDEO", str(error)) from error
    finally:
        file.file.close()


@app.get("/api/infer/video/jobs/{job_id}")
def get_video_job(job_id: str) -> dict[str, Any]:
    payload = VIDEO_JOBS.payload(job_id)
    if payload is None:
        raise _api_error(404, "VIDEO_JOB_NOT_FOUND", "Không tìm thấy phiên xử lý video")
    return payload


@app.get("/api/infer/video/jobs/{job_id}/download")
def download_video_result(job_id: str) -> FileResponse:
    job = VIDEO_JOBS.get(job_id)
    if job is None:
        raise _api_error(404, "VIDEO_JOB_NOT_FOUND", "Không tìm thấy phiên xử lý video")
    if job.status != "completed" or not job.output_path.is_file():
        raise _api_error(409, "VIDEO_NOT_READY", "Video kết quả chưa sẵn sàng")
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=f"detected_{Path(job.input_filename).stem}.mp4",
    )


@app.get("/api/infer/video/jobs/{job_id}/preview")
def preview_video_result(job_id: str) -> FileResponse:
    """Phục vụ MP4 inline để thẻ video của trình duyệt phát trực tiếp."""
    job = VIDEO_JOBS.get(job_id)
    if job is None:
        raise _api_error(404, "VIDEO_JOB_NOT_FOUND", "Không tìm thấy phiên xử lý video")
    if job.status != "completed" or not job.output_path.is_file():
        raise _api_error(409, "VIDEO_NOT_READY", "Video kết quả chưa sẵn sàng")
    return FileResponse(job.output_path, media_type="video/mp4")
