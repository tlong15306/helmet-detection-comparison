"""Tạo contact sheet cục bộ để duyệt nhanh ảnh Challenge Set."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_sheet(input_dir: Path, output: Path, columns: int, max_images: int | None) -> int:
    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"Không tìm thấy ảnh JPG/PNG trong {input_dir}")
    cell_width, cell_height, label_height, pad = 250, 180, 26, 8
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * (cell_width + pad) + pad, rows * (cell_height + label_height + pad) + pad),
        "#0f172a",
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(13)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((cell_width, cell_height))
        row, column = divmod(index, columns)
        x = pad + column * (cell_width + pad)
        y = pad + row * (cell_height + label_height + pad)
        background = Image.new("RGB", (cell_width, cell_height), "#1e293b")
        background.paste(image, ((cell_width - image.width) // 2, (cell_height - image.height) // 2))
        sheet.paste(background, (x, y))
        label = f"{index + 1:03d} {path.name[:27]}"
        draw.text((x, y + cell_height + 5), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(f"Đã tạo {output} ({len(paths)} ảnh)")
    return len(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo contact sheet ảnh challenge")
    parser.add_argument("--input-dir", default="data/challenge/candidates")
    parser.add_argument("--output", default="data/challenge/previews/candidates_contact_sheet.jpg")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    if args.columns < 1:
        parser.error("--columns phải lớn hơn 0")
    return args


def main() -> None:
    args = parse_args()
    build_sheet(Path(args.input_dir), Path(args.output), args.columns, args.max_images)


if __name__ == "__main__":
    main()
