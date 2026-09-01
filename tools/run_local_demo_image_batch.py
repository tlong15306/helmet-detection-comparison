"""Upload a local image folder to the running local demo API and save results."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import secrets
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def multipart_image(path: Path, model_id: str) -> tuple[bytes, str]:
    boundary = f"----helmet-demo-{secrets.token_hex(12)}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    lines = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="model_id"\r\n\r\n',
        model_id.encode("utf-8"), b"\r\n",
        f'--{boundary}\r\n'.encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {mime}\r\n\r\n".encode(), path.read_bytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(lines), boundary


def call_api(base_url: str, path: Path, model_id: str) -> dict:
    body, boundary = multipart_image(path, model_id)
    request = Request(
        f"{base_url.rstrip('/')}/api/infer/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST",
    )
    with urlopen(request, timeout=180) as response:  # local API only
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local demo API on every image in a folder")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model cần chạy; bỏ trống để chạy Faster R-CNN và RetinaNet.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output đã tồn tại: {args.output_dir}")
    paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"Không có ảnh JPG/PNG trong {args.input_dir}")
    args.output_dir.mkdir(parents=True)
    requested_models = args.model or ["faster_rcnn", "retinanet"]
    results = []
    for path in paths:
        for model in dict.fromkeys(requested_models):
            try:
                payload = call_api(args.api_url, path, model)
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(f"Demo API lỗi với {path.name}/{model}: {error}") from error
            image_data = str(payload.pop("result_image"))
            _, encoded = image_data.split(",", 1)
            result_name = f"{path.stem}__{model}.png"
            (args.output_dir / result_name).write_bytes(base64.b64decode(encoded))
            results.append({
                "input_file": path.name,
                "model_id": model,
                "result_image": result_name,
                "summary": payload.get("summary"),
                "alerts": payload.get("alerts"),
                "unknown_cases": payload.get("unknown_cases"),
                "latency_ms": payload.get("latency_ms"),
                "detections": payload.get("detections"),
                "thresholds": payload.get("thresholds"),
            })
            print(f"Đã chạy {path.name} / {model}")
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(paths), "runs": len(results), "output": args.output_dir.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
