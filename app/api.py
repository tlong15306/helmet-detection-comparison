"""FastAPI cục bộ cho giao diện Helmet Detection AI."""

from __future__ import annotations

import base64
import warnings
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.model_loader import DetectorManager, list_model_metadata
from app.video_jobs import ALLOWED_VIDEO_SUFFIXES, MAX_VIDEO_BYTES, VideoJobManager
from src.infer import (
    draw_detections,
    encode_png,
    normalize_pil_image,
    prediction_records,
    summarize_detections,
    validate_threshold,
)


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
GENERIC_UPLOAD_TYPES = {None, "", "application/octet-stream"}
DETECTOR_MANAGER = DetectorManager(requested_device="auto")
VIDEO_JOBS = VideoJobManager(DETECTOR_MANAGER)

app = FastAPI(
    title="Helmet Detection AI API",
    description="Backend cục bộ cho Faster R-CNN và RetinaNet.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


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


@app.post("/api/infer/image")
def infer_image(
    file: Annotated[UploadFile, File(description="Ảnh JPG hoặc PNG")],
    model_id: Annotated[str, Form()],
    threshold: Annotated[float | None, Form()] = None,
) -> dict[str, Any]:
    if file.content_type not in ALLOWED_IMAGE_TYPES | GENERIC_UPLOAD_TYPES:
        raise _api_error(415, "UNSUPPORTED_IMAGE_TYPE", "Chỉ hỗ trợ ảnh JPG, JPEG hoặc PNG")
    if threshold is not None:
        try:
            threshold = validate_threshold(threshold)
        except ValueError as error:
            raise _api_error(422, "INVALID_THRESHOLD", str(error)) from error

    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        image = decode_uploaded_image(content)
    except ValueError as error:
        raise _api_error(400, "INVALID_IMAGE", str(error)) from error
    finally:
        file.file.close()

    try:
        detector, prediction, latency_ms = DETECTOR_MANAGER.predict(
            model_id=model_id,
            image=image,
            confidence_threshold=threshold,
        )
        rendered = draw_detections(image, prediction, detector.class_names)
        rendered_png = encode_png(rendered)
        effective_threshold = (
            detector.default_threshold if threshold is None else float(threshold)
        )
        records = prediction_records(prediction, detector.class_names)
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
            "threshold": effective_threshold,
            "threshold_source": (
                "validation_default" if threshold is None or threshold == detector.default_threshold
                else "user_override"
            ),
            "latency_ms": round(latency_ms, 3),
            "summary": summarize_detections(prediction, detector.class_names),
            "detections": records,
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
        content = file.file.read(MAX_VIDEO_BYTES + 1)
        effective_threshold = (
            threshold
            if threshold is not None
            else next(model["default_threshold"] for model in list_model_metadata() if model["id"] == model_id)
        )
        job = VIDEO_JOBS.submit(content, file.filename or "video.mp4", model_id, effective_threshold)
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
