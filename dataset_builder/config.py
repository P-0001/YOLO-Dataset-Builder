from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "source_dir": "raw_images",
    "output_dir": "dataset",
    "classes": ["object"],
    "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
    "seed": 42,
    "extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
    "workers": 0,
    "create_empty_labels": False,
    "near_duplicate_threshold": 4,
    "quality": {"min_width": 32, "min_height": 32, "max_aspect_ratio": 5.0},
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}. Copy config.yaml and edit it first."
        )
    with path.open("r", encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    if not isinstance(supplied, dict):
        raise TypeError("Configuration must be a YAML mapping.")
    config = deepcopy(DEFAULT_CONFIG)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    base = path.parent.resolve()
    for key in ("source_dir", "output_dir"):
        item = Path(config[key])
        config[key] = str(item if item.is_absolute() else base / item)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    splits = config["splits"]
    required = {"train", "val", "test"}
    if (
        set(splits) != required
        or abs(sum(float(v) for v in splits.values()) - 1.0) > 1e-6
    ):
        raise ValueError(
            "splits must contain train, val, and test values that sum to 1.0."
        )
    if not isinstance(config["classes"], list) or not config["classes"]:
        raise ValueError("classes must be a non-empty YAML list.")
    if not all(isinstance(name, str) and name.strip() for name in config["classes"]):
        raise ValueError("classes must contain non-empty names.")
    if not isinstance(config["extensions"], list) or not all(
        isinstance(extension, str) and extension.startswith(".")
        for extension in config["extensions"]
    ):
        raise ValueError(
            "extensions must be a list of file extensions beginning with '.'."
        )
    if not isinstance(config["workers"], int) or config["workers"] < 0:
        raise ValueError("workers must be a non-negative integer.")
    if (
        not isinstance(config["near_duplicate_threshold"], int)
        or not 0 <= config["near_duplicate_threshold"] <= 7
    ):
        raise ValueError("near_duplicate_threshold must be an integer from 0 to 7.")
    quality = config["quality"]
    if not all(
        isinstance(quality[key], (int, float)) and quality[key] > 0
        for key in ("min_width", "min_height", "max_aspect_ratio")
    ):
        raise ValueError("quality limits must be positive numbers.")
