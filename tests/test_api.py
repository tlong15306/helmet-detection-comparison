"""Kiểm thử hợp đồng FastAPI không nạp checkpoint thật."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import api


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (32, 24), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_health_and_models_endpoints() -> None:
    client = TestClient(api.app)
    assert client.get("/api/health").status_code == 200
    response = client.get("/api/models")
    assert response.status_code == 200
    assert len(response.json()["models"]) == 2


def test_infer_image_contract(monkeypatch) -> None:
    prediction = {
        "boxes": torch.tensor([[1.0, 2.0, 20.0, 22.0]]),
        "labels": torch.tensor([2]),
        "scores": torch.tensor([0.91]),
    }
    detector = SimpleNamespace(
        model_id="faster_rcnn",
        display_name="Faster R-CNN",
        device=torch.device("cpu"),
        config={"model": {"name": "fasterrcnn_resnet50_fpn_v2"}},
        class_names={1: "BikeWithRider", 2: "NoHelmet", 3: "Helmet"},
        default_threshold=0.85,
    )
    monkeypatch.setattr(
        api.DETECTOR_MANAGER,
        "predict",
        lambda model_id, image, confidence_threshold: (detector, prediction, 12.3456),
    )
    client = TestClient(api.app)
    response = client.post(
        "/api/infer/image",
        data={"model_id": "faster_rcnn", "threshold": "0.85"},
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["NoHelmet"] == 1
    assert payload["detections"][0]["confidence"] == 0.91
    assert payload["detections"][0]["detection_id"] == "detection_1"
    assert payload["rider_analysis"]["summary"]["unassigned_heads"] == 1
    assert payload["rider_analysis"]["summary"]["confirmed_driver_no_helmet"] == 0
    assert payload["latency_ms"] == 12.346
    assert payload["result_image"].startswith("data:image/png;base64,")


def test_infer_image_rejects_wrong_content_type() -> None:
    client = TestClient(api.app)
    response = client.post(
        "/api/infer/image",
        data={"model_id": "faster_rcnn"},
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_infer_video_queues_job_and_exposes_status(monkeypatch) -> None:
    payload = {
        "id": "video-job-1",
        "status": "queued",
        "threshold": 0.6,
        "progress": {"processed_frames": 0, "total_frames": None, "percent": 0},
        "download_url": None,
        "preview_url": None,
        "error": None,
    }
    monkeypatch.setattr(
        api.VIDEO_JOBS,
        "submit",
        lambda content, filename, model_id, threshold: SimpleNamespace(job_id="video-job-1"),
    )
    monkeypatch.setattr(api.VIDEO_JOBS, "payload", lambda job_id: payload if job_id == "video-job-1" else None)
    client = TestClient(api.app)
    response = client.post(
        "/api/infer/video",
        data={"model_id": "retinanet", "threshold": "0.60"},
        files={"file": ("sample.mp4", b"small test video", "video/mp4")},
    )
    assert response.status_code == 202
    assert response.json()["id"] == "video-job-1"
    assert response.json()["status"] == "queued"
    assert client.get("/api/infer/video/jobs/video-job-1").json()["threshold"] == 0.6


def test_infer_video_rejects_unsupported_extension() -> None:
    client = TestClient(api.app)
    response = client.post(
        "/api/infer/video",
        data={"model_id": "retinanet"},
        files={"file": ("sample.mkv", b"not accepted", "video/x-matroska")},
    )
    assert response.status_code == 415


def test_video_preview_is_inline_while_download_is_attachment(monkeypatch, tmp_path) -> None:
    result_path = tmp_path / "detected.mp4"
    result_path.write_bytes(b"fake-mp4")
    job = SimpleNamespace(
        status="completed",
        output_path=result_path,
        input_filename="source.mp4",
    )
    monkeypatch.setattr(api.VIDEO_JOBS, "get", lambda job_id: job if job_id == "done" else None)
    client = TestClient(api.app)
    preview = client.get("/api/infer/video/jobs/done/preview")
    download = client.get("/api/infer/video/jobs/done/download")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "video/mp4"
    assert "content-disposition" not in preview.headers
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
