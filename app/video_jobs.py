"""Hàng đợi video cục bộ cho demo, xử lý tuần tự để an toàn bộ nhớ GPU."""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from app.model_loader import DetectorManager
from src.infer import draw_detections, normalize_pil_image, summarize_detections
from src.utils import resolve_project_path


MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 5 * 60
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class VideoJob:
    job_id: str
    model_id: str
    threshold: float
    input_filename: str
    input_path: Path
    output_path: Path
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    processed_frames: int = 0
    total_frames: int | None = None
    progress_percent: float = 0.0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    average_latency_ms: float | None = None
    summary: dict[str, int] = field(default_factory=dict)
    model_name: str | None = None
    device_name: str | None = None
    error: str | None = None


class VideoJobManager:
    """Một worker xử lý duy nhất: ổn định model/GPU và có tiến độ để polling."""

    def __init__(self, detector_manager: DetectorManager, jobs_dir: Path | None = None) -> None:
        self.detector_manager = detector_manager
        self.jobs_dir = jobs_dir or resolve_project_path("app/assets/generated/video_jobs")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-inference")

    def submit(
        self,
        content: bytes,
        filename: str,
        model_id: str,
        threshold: float,
    ) -> VideoJob:
        safe_name = Path(filename or "video.mp4").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError("Chỉ hỗ trợ video MP4, MOV hoặc AVI")
        if not content:
            raise ValueError("Tệp video rỗng")
        if len(content) > MAX_VIDEO_BYTES:
            raise ValueError("Video vượt quá giới hạn 200 MB")

        job_id = uuid.uuid4().hex
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        output_path = job_dir / "detected.mp4"
        input_path.write_bytes(content)
        try:
            self._validate_video(input_path)
        except Exception:
            input_path.unlink(missing_ok=True)
            job_dir.rmdir()
            raise
        job = VideoJob(
            job_id=job_id,
            model_id=model_id,
            threshold=float(threshold),
            input_filename=safe_name,
            input_path=input_path,
            output_path=output_path,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._worker.submit(self._run, job_id)
        return job

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def payload(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        with self._lock:
            return {
                "id": job.job_id,
                "status": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "model": {"id": job.model_id, "name": job.model_name},
                "threshold": job.threshold,
                "input": {
                    "filename": job.input_filename,
                    "width": job.width,
                    "height": job.height,
                    "fps": job.fps,
                    "total_frames": job.total_frames,
                },
                "progress": {
                    "processed_frames": job.processed_frames,
                    "total_frames": job.total_frames,
                    "percent": round(job.progress_percent, 2),
                },
                "summary": dict(job.summary),
                "average_latency_ms": (
                    None if job.average_latency_ms is None else round(job.average_latency_ms, 3)
                ),
                "device_name": job.device_name,
                "download_url": (
                    f"/api/infer/video/jobs/{job.job_id}/download"
                    if job.status == "completed" and job.output_path.is_file()
                    else None
                ),
                "preview_url": (
                    f"/api/infer/video/jobs/{job.job_id}/preview"
                    if job.status == "completed" and job.output_path.is_file()
                    else None
                ),
                "error": job.error,
            }

    def _validate_video(self, input_path: Path) -> None:
        capture = cv2.VideoCapture(str(input_path))
        try:
            if not capture.isOpened():
                raise ValueError("Không thể đọc video; hãy dùng tệp MP4, MOV hoặc AVI hợp lệ")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise ValueError("Video không có thông tin khung hình hợp lệ")
            if frames > 0 and frames / fps > MAX_VIDEO_DURATION_SECONDS:
                raise ValueError("Video dài quá 5 phút; hãy cắt ngắn trước khi xử lý")
            ok, _frame = capture.read()
            if not ok:
                raise ValueError("Video không chứa frame có thể đọc")
        finally:
            capture.release()

    def _update(self, job: VideoJob, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _utc_now()

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        capture: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        try:
            self._update(job, status="processing", error=None)
            capture = cv2.VideoCapture(str(job.input_path))
            if not capture.isOpened():
                raise RuntimeError("Không thể mở video đã tải lên")
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(job.output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Không thể tạo tệp MP4 kết quả trên máy này")

            with self.detector_manager.hold_detector(job.model_id) as detector:
                self._update(
                    job,
                    model_name=detector.display_name,
                    device_name=(
                        torch.cuda.get_device_name(detector.device)
                        if detector.device.type == "cuda"
                        else "CPU"
                    ),
                    fps=round(fps, 3),
                    width=width,
                    height=height,
                    total_frames=total_frames,
                    summary={name: 0 for class_id, name in detector.class_names.items() if class_id != 0},
                )
                latency_total = 0.0
                frame_count = 0
                accumulated = {name: 0 for class_id, name in detector.class_names.items() if class_id != 0}
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = normalize_pil_image(Image.fromarray(rgb))
                    prediction, latency_ms = self.detector_manager.predict_loaded(
                        detector, image, job.threshold
                    )
                    rendered = draw_detections(image, prediction, detector.class_names)
                    output_frame = cv2.cvtColor(np.asarray(rendered), cv2.COLOR_RGB2BGR)
                    writer.write(output_frame)
                    frame_count += 1
                    latency_total += latency_ms
                    frame_summary = summarize_detections(prediction, detector.class_names)
                    for key, value in frame_summary.items():
                        accumulated[key] += value
                    percent = (frame_count / total_frames * 100.0) if total_frames else 0.0
                    self._update(
                        job,
                        processed_frames=frame_count,
                        progress_percent=min(percent, 99.9) if total_frames else 0.0,
                        average_latency_ms=latency_total / frame_count,
                        summary=accumulated,
                    )
            if frame_count == 0:
                raise RuntimeError("Không đọc được frame nào từ video")
            self._update(job, status="completed", progress_percent=100.0)
        except Exception as error:  # noqa: BLE001 - lỗi cần phản hồi cho giao diện
            traceback.print_exc()
            self._update(job, status="failed", error=str(error))
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
