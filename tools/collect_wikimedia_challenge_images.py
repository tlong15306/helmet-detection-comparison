"""Thu thập ảnh ứng viên từ Wikimedia Commons kèm manifest giấy phép.

Chỉ dùng cho Challenge Set v1. Script không tự chọn ảnh chính thức hoặc tạo
ground truth; Long vẫn phải duyệt contact sheet và annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "helmet-detection-comparison-challenge-set/1.0 (academic project)"
DEFAULT_QUERIES = (
    "Vietnam motorcycle traffic",
    "Hanoi motorcycle traffic",
    "Ho Chi Minh City motorcycle traffic",
    "Vietnam motorbike street",
    "Vietnam traffic night motorcycle",
)
ALLOWED_LICENSE_PREFIXES = ("CC BY", "CC BY-SA", "CC0", "Public domain")
IMAGE_MIMES = {"image/jpeg", "image/png"}
MANIFEST_FIELDS = (
    "challenge_image_id",
    "local_filename",
    "source_group_id",
    "source_title",
    "source_page_url",
    "direct_download_url",
    "creator",
    "license_name",
    "license_url",
    "downloaded_at",
    "sha256",
    "intended_use",
    "review_status",
    "selection_reason",
    "notes",
)


def strip_markup(value: object) -> str:
    """Biến metadata HTML của Commons thành chuỗi CSV đơn giản."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def fetch_json(params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS API
        return json.load(response)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - URL from Commons API
        content = response.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Ảnh vượt giới hạn 20 MB")
    return content


def search_files(query: str, limit: int) -> list[dict[str, Any]]:
    payload = fetch_json(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": query,
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata|size",
            "iiurlwidth": "1600",
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    return [pages[key] for key in sorted(pages, key=lambda value: int(value))]


def page_to_candidate(page: dict[str, Any], query: str) -> dict[str, str] | None:
    image_info = (page.get("imageinfo") or [{}])[0]
    mime = str(image_info.get("mime", ""))
    metadata = image_info.get("extmetadata") or {}
    license_name = strip_markup(metadata.get("LicenseShortName", {}).get("value"))
    if mime not in IMAGE_MIMES or not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
        return None
    direct_url = str(image_info.get("thumburl") or image_info.get("url") or "")
    if not direct_url:
        return None
    page_id = int(page["pageid"])
    title = str(page.get("title", ""))
    creator = strip_markup(metadata.get("Artist", {}).get("value"))
    license_url = strip_markup(metadata.get("LicenseUrl", {}).get("value"))
    return {
        "source_title": title,
        "source_page_url": f"https://commons.wikimedia.org/?curid={page_id}",
        "direct_download_url": direct_url,
        "creator": creator,
        "license_name": license_name,
        "license_url": license_url,
        "query": query,
        "mime": mime,
        "page_id": str(page_id),
        "width": str(image_info.get("width", "")),
        "height": str(image_info.get("height", "")),
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.read_text(encoding="utf-8-sig").strip():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def extension_for_mime(mime: str) -> str:
    return ".jpg" if mime == "image/jpeg" else ".png"


def validate_image(content: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Tệp tải về không phải ảnh JPG/PNG hợp lệ") from error
    if width < 400 or height < 300:
        raise ValueError(f"Ảnh quá nhỏ ({width}x{height})")
    return width, height


def collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = read_manifest(manifest_path)
    known_pages = {row["source_page_url"] for row in existing_rows if row.get("source_page_url")}
    known_hashes = {row["sha256"] for row in existing_rows if row.get("sha256")}
    rows = list(existing_rows)
    added = 0
    consecutive_errors = 0
    candidates: list[dict[str, str]] = []
    for query in args.query:
        for page in search_files(query, args.per_query):
            candidate = page_to_candidate(page, query)
            if candidate and candidate["source_page_url"] not in known_pages:
                candidates.append(candidate)

    for candidate in candidates:
        if added >= args.max_images:
            break
        source_title = candidate["source_title"]
        filename = f"wm_{len(rows) + added + 1:03d}_{candidate['page_id']}{extension_for_mime(candidate['mime'])}"
        if args.dry_run:
            print(
                f"[DRY RUN] {filename}: {source_title} "
                f"({candidate['width']}x{candidate['height']}, {candidate['license_name']})"
            )
            added += 1
            known_pages.add(candidate["source_page_url"])
            continue
        try:
            content = fetch_bytes(candidate["direct_download_url"])
            width, height = validate_image(content)
        except (OSError, ValueError) as error:
            print(f"Bỏ qua {source_title}: {error}", file=sys.stderr)
            consecutive_errors += 1
            if consecutive_errors >= args.max_consecutive_errors:
                print("Dừng an toàn vì nguồn trả lỗi liên tiếp; có thể chạy lại sau.", file=sys.stderr)
                break
            time.sleep(args.throttle_seconds)
            continue
        consecutive_errors = 0
        digest = hashlib.sha256(content).hexdigest()
        if digest in known_hashes:
            continue
        filename = f"wm_{len(rows) + 1:03d}_{digest[:10]}{extension_for_mime(candidate['mime'])}"
        (output_dir / filename).write_bytes(content)
        rows.append(
            {
                "challenge_image_id": f"candidate_{len(rows) + 1:03d}",
                "local_filename": filename,
                "source_group_id": f"wikimedia_page_{candidate['page_id']}",
                "source_title": source_title,
                "source_page_url": candidate["source_page_url"],
                "direct_download_url": candidate["direct_download_url"],
                "creator": candidate["creator"],
                "license_name": candidate["license_name"],
                "license_url": candidate["license_url"],
                "downloaded_at": datetime.now(UTC).isoformat(),
                "sha256": digest,
                "intended_use": "challenge_only",
                "review_status": "candidate",
                "selection_reason": "",
                "notes": f"query={candidate['query']}; dimensions={width}x{height}",
            }
        )
        # Ghi ngay sau từng ảnh thành công để không mất provenance nếu nguồn ngắt.
        write_manifest(manifest_path, rows)
        added += 1
        known_hashes.add(digest)
        known_pages.add(candidate["source_page_url"])
        print(f"Tải {filename}: {source_title} ({width}x{height})")
        time.sleep(args.throttle_seconds)

    if not args.dry_run:
        write_manifest(manifest_path, rows)
    print(f"Đã thêm {added} ảnh ứng viên; manifest hiện có {len(rows)} dòng.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tải ảnh challenge ứng viên từ Wikimedia Commons")
    parser.add_argument("--output-dir", default="data/challenge/candidates")
    parser.add_argument("--manifest", default="data/challenge/metadata/sources_manifest.csv")
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=3.0,
        help="Thời gian chờ giữa các lần tải thật để tôn trọng giới hạn nguồn",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
        help="Dừng đợt tải khi nguồn trả lỗi liên tiếp để tránh retry dồn",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", action="append", default=[], help="Có thể truyền nhiều lần")
    args = parser.parse_args()
    if args.max_images < 1 or args.per_query < 1 or args.throttle_seconds < 0 or args.max_consecutive_errors < 1:
        parser.error("--max-images/per-query/max-consecutive-errors phải lớn hơn 0; throttle không âm")
    if not args.query:
        args.query = list(DEFAULT_QUERIES)
    return args


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    raise SystemExit(collect(parse_args()))


if __name__ == "__main__":
    main()
