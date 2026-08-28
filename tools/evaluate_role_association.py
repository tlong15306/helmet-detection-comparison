"""Xuất báo cáo tạm thời cho baseline liên kết tài xế/người ngồi sau."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.role_evaluation import evaluate_role_candidate_baseline
from src.utils import resolve_project_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá candidate role trên role_dev đã review")
    parser.add_argument("--tasks", default="data/role_association/annotations/role_dev.pending.json")
    parser.add_argument("--output", default="outputs/role_association/role_dev_provisional_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_path = resolve_project_path(args.tasks)
    with task_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    report = evaluate_role_candidate_baseline(payload)
    report["source"] = {
        "tasks": str(task_path),
        "split": payload.get("source", {}).get("split"),
    }
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
