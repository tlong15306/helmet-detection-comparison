"""Tạo contact sheet crop vùng xe để người review gán vai trò thủ công."""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.infer import normalize_pil_image
from src.utils import resolve_project_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CELL_WIDTH = 360
CELL_HEIGHT = 270
HEADER_HEIGHT = 34
PADDING = 14
HEAD_COLORS = {"helmet": (34, 197, 94), "no_helmet": (239, 68, 68)}
BIKE_COLOR = (245, 158, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo contact sheet cho role_dev pending")
    parser.add_argument("--tasks", default="data/role_association/annotations/role_dev.pending.json")
    parser.add_argument("--output-dir", default="data/role_association/previews")
    parser.add_argument("--per-page", type=int, default=16)
    parser.add_argument("--page", type=int, default=None, help="Chỉ tạo một trang, bắt đầu từ 1")
    args = parser.parse_args()
    if args.per_page < 1:
        parser.error("--per-page phải lớn hơn 0")
    return args


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _crop_bounds(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    pad_x = max(28.0, (x2 - x1) * 0.45)
    pad_y = max(28.0, (y2 - y1) * 0.45)
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def _render_task(task: dict[str, Any]) -> Image.Image:
    image_path = resolve_project_path(task["image_path"])
    with Image.open(image_path) as source:
        image = normalize_pil_image(source)
    left, top, right, bottom = _crop_bounds(task["bike_box_xyxy"], image.width, image.height)
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    bike = task["bike_box_xyxy"]
    draw.rectangle((bike[0] - left, bike[1] - top, bike[2] - left, bike[3] - top), outline=BIKE_COLOR, width=4)
    for head in task["heads"]:
        box = head["box_xyxy"]
        color = HEAD_COLORS[head["helmet_status"]]
        local_box = (box[0] - left, box[1] - top, box[2] - left, box[3] - top)
        draw.rectangle(local_box, outline=color, width=4)
        label = f"H{head['annotation_id']} · {head['helmet_status']}"
        draw.rectangle((local_box[0], max(0, local_box[1] - 19), local_box[0] + 150, local_box[1]), fill=color)
        draw.text((local_box[0] + 4, max(0, local_box[1] - 17)), label, fill="white", font=_font(11))
    crop.thumbnail((CELL_WIDTH, CELL_HEIGHT - HEADER_HEIGHT), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "#ffffff")
    x = (CELL_WIDTH - crop.width) // 2
    y = HEADER_HEIGHT + (CELL_HEIGHT - HEADER_HEIGHT - crop.height) // 2
    cell.paste(crop, (x, y))
    cell_draw = ImageDraw.Draw(cell)
    label = f"{task['task_id']} · bike A{task['bike_annotation_id']} · {', '.join(task['difficulty_tags'])}"
    cell_draw.rectangle((0, 0, CELL_WIDTH, HEADER_HEIGHT), fill="#0f172a")
    cell_draw.text((8, 9), label[:58], fill="white", font=_font(12))
    return cell


def build_pages(tasks_path: str | Path, output_dir: str | Path, per_page: int, page: int | None = None) -> list[Path]:
    path = resolve_project_path(tasks_path)
    with path.open("r", encoding="utf-8") as stream:
        tasks = json.load(stream)["tasks"]
    if any(task["review"]["status"] != "pending" for task in tasks):
        raise ValueError("Contact sheet hiện chỉ dành cho task pending")
    page_count = ceil(len(tasks) / per_page)
    if page is not None and not 1 <= page <= page_count:
        raise ValueError(f"page phải trong [1, {page_count}]")
    page_numbers = [page] if page is not None else list(range(1, page_count + 1))
    destination = resolve_project_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    columns = 4
    rows = ceil(per_page / columns)
    outputs: list[Path] = []
    for page_number in page_numbers:
        start = (page_number - 1) * per_page
        page_tasks = tasks[start : start + per_page]
        canvas = Image.new("RGB", (columns * CELL_WIDTH + (columns + 1) * PADDING, rows * CELL_HEIGHT + (rows + 1) * PADDING), "#e2e8f0")
        for index, task in enumerate(page_tasks):
            cell = _render_task(task)
            row, column = divmod(index, columns)
            x = PADDING + column * (CELL_WIDTH + PADDING)
            y = PADDING + row * (CELL_HEIGHT + PADDING)
            canvas.paste(cell, (x, y))
        output = destination / f"role_dev_pending_page_{page_number:02d}.jpg"
        canvas.save(output, quality=92, optimize=True)
        outputs.append(output)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = build_pages(args.tasks, args.output_dir, args.per_page, args.page)
    print(json.dumps({"pages": [str(path) for path in outputs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
