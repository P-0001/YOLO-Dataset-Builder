from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Reports (phase, completed, total). A total of 0 means "indeterminate".
Progress = Callable[[str, int, int], None]


def report(progress: Progress | None, phase: str, done: int = 0, total: int = 0) -> None:
    if progress is not None:
        progress(phase, done, total)


def map_progress(
    executor: Executor,
    function: Callable[[T], R],
    items: Sequence[T],
    progress: Progress | None,
    phase: str,
) -> list[R]:
    """``list(executor.map(...))`` that reports completion as results arrive.

    ``Executor.map`` yields in submission order, so the count is a faithful
    measure of how much of the input has been consumed.
    """
    total = len(items)
    results: list[R] = []
    for result in executor.map(function, items):
        results.append(result)
        report(progress, phase, len(results), total)
    return results


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
