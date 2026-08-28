"""Generate dependency-free SVG figures for the experimental-report draft.

The script reads saved experiment metadata from ``outputs/`` and writes only
report assets. It never changes metrics, checkpoints, or other raw artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "report_drafts" / "assets"
MODEL_NAMES = {"faster_rcnn": "Faster R-CNN", "retinanet": "RetinaNet"}
COLORS = {"faster_rcnn": "#2F6B9A", "retinanet": "#E07A3F"}
GRID_COLOR = "#D9E1E8"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _document(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1F2933}</style>',
            *body,
            "</svg>",
        ]
    )


def _text(x: float, y: float, value: str, size: int = 14, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{html.escape(value)}</text>'
    )


def _axis_and_grid(body: list[str], x: float, y: float, width: float, height: float, maximum: float, ticks: int = 5) -> None:
    body.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#607080"/>')
    body.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#607080"/>')
    for index in range(ticks + 1):
        value = maximum * index / ticks
        y_pos = y + height - height * index / ticks
        body.append(f'<line x1="{x}" y1="{y_pos:.1f}" x2="{x + width}" y2="{y_pos:.1f}" stroke="{GRID_COLOR}"/>')
        body.append(_text(x - 8, y_pos + 4, f"{value:.2f}", 11, "end"))


def _legend(body: list[str], x: float, y: float) -> None:
    for index, model_key in enumerate(MODEL_NAMES):
        offset = index * 160
        body.append(f'<rect x="{x + offset}" y="{y - 10}" width="13" height="13" fill="{COLORS[model_key]}"/>')
        body.append(_text(x + offset + 19, y + 1, MODEL_NAMES[model_key], 12))


def build_test_metrics_figure(test_metrics: dict[str, dict], destination: Path) -> None:
    width, height = 930, 530
    left, top, plot_width, plot_height = 82, 92, 800, 340
    keys = ["map_50_95", "map_50", "map_75", "mar_100"]
    labels = ["mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "mAR@100"]
    body = [_text(width / 2, 34, "Test-set detection metrics (359 images)", 20, "middle", "bold")]
    _axis_and_grid(body, left, top, plot_width, plot_height, 1.0)
    group_width = plot_width / len(keys)
    bar_width = 42
    for group_index, (key, label) in enumerate(zip(keys, labels)):
        center = left + group_width * (group_index + 0.5)
        for model_index, model_key in enumerate(MODEL_NAMES):
            value = test_metrics[model_key]["metrics"][key]
            x = center + (model_index - 0.5) * (bar_width + 5) - bar_width / 2
            bar_height = plot_height * value
            y = top + plot_height - bar_height
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{COLORS[model_key]}"/>')
            body.append(_text(x + bar_width / 2, y - 6, f"{value:.3f}", 11, "middle"))
        body.append(_text(center, top + plot_height + 26, label, 12, "middle"))
    _legend(body, left + 245, 482)
    destination.write_text(_document(width, height, body), encoding="utf-8")


def _single_bar_panel(body: list[str], x: float, y: float, width: float, height: float, title: str, ylabel: str, labels: list[str], values: list[float]) -> None:
    maximum = max(values) * 1.22
    body.append(_text(x + width / 2, y - 20, title, 15, "middle", "bold"))
    _axis_and_grid(body, x, y, width, height, maximum)
    for index, (label, value) in enumerate(zip(labels, values)):
        slot = width / len(values)
        bar_width = slot * 0.48
        bar_x = x + slot * index + (slot - bar_width) / 2
        bar_height = height * value / maximum
        bar_y = y + height - bar_height
        model_key = list(MODEL_NAMES)[index]
        body.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{COLORS[model_key]}"/>')
        body.append(_text(bar_x + bar_width / 2, bar_y - 6, f"{value:.2f}", 11, "middle"))
        body.append(_text(bar_x + bar_width / 2, y + height + 26, label, 11, "middle"))
    body.append(_text(x - 44, y + height / 2, ylabel, 11, "middle"))


def build_latency_figure(latency_metrics: dict[str, dict], destination: Path) -> None:
    width, height = 940, 500
    labels = [MODEL_NAMES[key] for key in MODEL_NAMES]
    latency_values = [latency_metrics[key]["latency"]["mean_ms"] for key in MODEL_NAMES]
    fps_values = [latency_metrics[key]["latency"]["fps_from_mean_latency"] for key in MODEL_NAMES]
    body = [_text(width / 2, 34, "Validation-set inference benchmark (batch size = 1)", 20, "middle", "bold")]
    _single_bar_panel(body, 78, 100, 330, 285, "Mean inference latency", "ms / image", labels, latency_values)
    _single_bar_panel(body, 545, 100, 330, 285, "Throughput from mean latency", "frames / s", labels, fps_values)
    destination.write_text(_document(width, height, body), encoding="utf-8")


def _line_points(values: list[float], x: float, y: float, width: float, height: float, maximum: float) -> str:
    count = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        px = x + width * index / count
        py = y + height - height * value / maximum
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def _line_panel(body: list[str], x: float, y: float, width: float, height: float, title: str, series: dict[str, list[float]]) -> None:
    maximum = max(value for sequence in series.values() for value in sequence) * 1.08
    body.append(_text(x + width / 2, y - 20, title, 15, "middle", "bold"))
    _axis_and_grid(body, x, y, width, height, maximum)
    for model_key, sequence in series.items():
        points = _line_points(sequence, x, y, width, height, maximum)
        body.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[model_key]}" stroke-width="3"/>')
        for pair in points.split():
            px, py = pair.split(",")
            body.append(f'<circle cx="{px}" cy="{py}" r="2.5" fill="{COLORS[model_key]}"/>')
    for epoch in [1, 5, 10, 15, 20]:
        px = x + width * (epoch - 1) / 19
        body.append(_text(px, y + height + 23, str(epoch), 11, "middle"))
    body.append(_text(x + width / 2, y + height + 48, "Epoch", 11, "middle"))


def build_training_figure(run_manifests: dict[str, dict], destination: Path) -> None:
    width, height = 1020, 520
    loss_series = {}
    map_series = {}
    for model_key in MODEL_NAMES:
        history = run_manifests[model_key]["history"]
        loss_series[model_key] = [record["train_loss"] for record in history]
        map_series[model_key] = [record["validation"]["map_50_95"] for record in history]
    body = [_text(width / 2, 34, "Training and validation curves", 20, "middle", "bold")]
    _line_panel(body, 86, 102, 365, 315, "Training loss by epoch", loss_series)
    _line_panel(body, 595, 102, 365, 315, "Validation mAP@0.5:0.95 by epoch", map_series)
    _legend(body, 347, 480)
    destination.write_text(_document(width, height, body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = {key: _read_json(PROJECT_ROOT / "outputs" / key / "metrics" / "test_metrics.json") for key in MODEL_NAMES}
    latency_metrics = {key: _read_json(PROJECT_ROOT / "outputs" / key / "metrics" / "latency_validation.json") for key in MODEL_NAMES}
    run_manifests = {key: _read_json(PROJECT_ROOT / "outputs" / key / "run_manifest.json") for key in MODEL_NAMES}

    build_test_metrics_figure(test_metrics, output_dir / "test_metrics_comparison.svg")
    build_latency_figure(latency_metrics, output_dir / "latency_fps_comparison.svg")
    build_training_figure(run_manifests, output_dir / "training_curves.svg")
    print(f"Created report figures in: {output_dir}")


if __name__ == "__main__":
    main()
