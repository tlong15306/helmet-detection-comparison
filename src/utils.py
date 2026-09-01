"""Tiện ích dùng chung cho cấu hình, đường dẫn và khả năng tái lập."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Chuyển đường dẫn tương đối thành đường dẫn tuyệt đối trong dự án."""
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc một tệp YAML và trả về dictionary."""
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Cấu hình phải là mapping YAML: {config_path}")
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gộp cấu hình lồng nhau mà không thay đổi dictionary đầu vào."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc cấu hình mô hình và tự động gộp `base_config` nếu có."""
    config = load_yaml(path)
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    return deep_merge(load_config(base_path), config)


def set_seed(seed: int) -> None:
    """Thiết lập random seed cho Python, NumPy và PyTorch nếu khả dụng."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
