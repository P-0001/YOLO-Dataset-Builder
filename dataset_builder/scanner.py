from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .utils import Progress, map_progress, report


def discover_images(source: Path, extensions: list[str]) -> list[Path]:
    allowed = {extension.lower() for extension in extensions}
    return sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        ),
        key=lambda p: str(p).lower(),
    )


def discover_unsupported_files(source: Path, extensions: list[str]) -> list[Path]:
    """Return non-label files that the configured image reader will ignore."""
    allowed = {extension.lower() for extension in extensions}
    return sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix.lower() not in allowed
            and path.suffix.lower() != ".txt"
        ),
        key=lambda p: str(p).lower(),
    )


def _scan_one(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
        return {
            "path": path,
            "width": width,
            "height": height,
            "format": image_format.upper(),
            "file_size": path.stat().st_size,
            "error": None,
        }
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "path": path,
            "width": None,
            "height": None,
            "format": None,
            "file_size": path.stat().st_size if path.exists() else 0,
            "error": str(exc),
        }


def scan_images(
    source: Path,
    extensions: list[str],
    workers: int = 0,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    report(progress, "Discovering images")
    paths = discover_images(source, extensions)
    max_workers = workers or min(32, max(1, (len(paths) // 100) + 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return map_progress(executor, _scan_one, paths, progress, "Scanning images")
