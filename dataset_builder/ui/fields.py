"""One declarative description per form field.

Every field appears exactly once, here. The spec drives widget creation,
validation, config building, config loading, preference save/restore, and
reset-to-defaults, so adding a setting means adding a single entry below.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import DEFAULT_CONFIG

Parser = Callable[[str], Any]

UNSET = object()


# --------------------------------------------------------------- parsers
def _range_text(low: float | None, high: float | None) -> str:
    if low is not None and high is not None:
        return f" from {_trim(low)} to {_trim(high)}"
    if low is not None:
        return f" of at least {_trim(low)}"
    if high is not None:
        return f" of at most {_trim(high)}"
    return ""


def _trim(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _bounded(cast: Callable[[str], Any], noun: str, low, high) -> Parser:
    message = f"Enter a {noun}{_range_text(low, high)}."

    def parse(raw: str) -> Any:
        try:
            value = cast(raw.strip())
        except ValueError:
            raise ValueError(message) from None
        if (
            (isinstance(value, float) and not math.isfinite(value))
            or (low is not None and value < low)
            or (high is not None and value > high)
        ):
            raise ValueError(message)
        return value

    return parse


def integer(low: int | None = None, high: int | None = None) -> Parser:
    return _bounded(int, "whole number", low, high)


def number(low: float | None = None, high: float | None = None) -> Parser:
    return _bounded(float, "number", low, high)


def text(message: str) -> Parser:
    def parse(raw: str) -> str:
        value = raw.strip()
        if not value:
            raise ValueError(message)
        return value

    return parse


def names(message: str) -> Parser:
    def parse(raw: str) -> list[str]:
        items = [part.strip() for part in raw.split(",") if part.strip()]
        if not items:
            raise ValueError(message)
        return items

    return parse


def optional_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def suffixes(raw: str) -> list[str]:
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not items or not all(item.startswith(".") for item in items):
        raise ValueError("List file extensions separated by commas, each starting '.'")
    return items


def class_ids(raw: str) -> list[int]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        values = [int(item) for item in items]
    except ValueError:
        raise ValueError("Enter whole numbers separated by commas, e.g. 0, 2") from None
    if any(value < 0 for value in values):
        raise ValueError("Class IDs cannot be negative.")
    return values


def instance_id(raw: str) -> str:
    value = raw.strip()
    if value and (not value.isdigit() or int(value) < 1):
        raise ValueError("Enter a positive numeric Vast.ai instance ID, or leave blank.")
    return value


def join(values: Any) -> str:
    return ", ".join(str(value) for value in values)


# ------------------------------------------------------------------ spec
@dataclass(frozen=True)
class Field:
    """A single form control and its mapping to a config key."""

    name: str
    label: str
    path: tuple[str, ...]
    parse: Parser | None = None  # None marks a boolean checkbox
    fmt: Callable[[Any], str] = str
    group: str = ""  # shared error/hint line for side-by-side fields
    hint: str = ""
    width: int = 0
    default: Any = UNSET

    @property
    def boolean(self) -> bool:
        return self.parse is None

    def initial(self) -> Any:
        """Starting value: an explicit default, else the one from DEFAULT_CONFIG."""
        if self.default is not UNSET:
            return self.default
        value = DEFAULT_CONFIG
        for key in self.path:
            value = value[key]
        return value if self.boolean else self.fmt(value)


@dataclass(frozen=True)
class Group:
    """A row of related fields sharing one caption, hint, and error line."""

    name: str
    label: str
    hint: str
    members: tuple[str, ...]
    rule: Callable[[dict[str, Any]], str] | None = None


def _splits_total(values: dict[str, Any]) -> str:
    total = sum(values[name] for name in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        return f"Splits must total 1.0 (currently {total:.2f})."
    return ""


FIELDS: tuple[Field, ...] = (
    Field(
        "source",
        "Source folder",
        ("source_dir",),
        # Never a parse error: only Build needs a source, so its absence is
        # checked as a build-time gate rather than a form-wide error.
        str.strip,
        hint="The folder of images to read. Required for Build.",
        default="",
    ),
    Field(
        "output",
        "Dataset folder",
        ("output_dir",),
        text("Choose where the dataset should be created."),
        default="",
    ),
    Field(
        "classes",
        "Classes",
        ("classes",),
        names("Enter at least one class name."),
        fmt=join,
        hint="Comma-separated, in class-ID order.",
    ),
    Field("train", "Train", ("splits", "train"), number(0, 1), group="splits", width=7),
    Field("val", "Validation", ("splits", "val"), number(0, 1), group="splits", width=7),
    Field("test", "Test", ("splits", "test"), number(0, 1), group="splits", width=7),
    Field(
        "seed",
        "Random seed",
        ("seed",),
        integer(),
        hint="The same seed always produces the same split.",
    ),
    Field(
        "empty_labels",
        "Create an empty label when no sibling .txt file exists",
        ("create_empty_labels",),
        default=True,
    ),
    Field(
        "extensions",
        "Extensions",
        ("extensions",),
        suffixes,
        fmt=join,
        hint="Image types to read from the source folder.",
    ),
    Field(
        "workers",
        "Workers",
        ("workers",),
        integer(0),
        hint="0 picks a thread count based on how many images there are.",
    ),
    Field(
        "near_duplicate_threshold",
        "Near-duplicate threshold",
        ("near_duplicate_threshold",),
        integer(0, 7),
        hint="Bits of pHash difference still counted as a duplicate. Higher is stricter about similarity.",
    ),
    Field(
        "min_width",
        "Min width",
        ("quality", "min_width"),
        number(1),
        group="quality",
        width=8,
    ),
    Field(
        "min_height",
        "Min height",
        ("quality", "min_height"),
        number(1),
        group="quality",
        width=8,
    ),
    Field(
        "max_aspect_ratio",
        "Max aspect",
        ("quality", "max_aspect_ratio"),
        number(0.01),
        group="quality",
        width=8,
    ),
    Field(
        "train_model",
        "Base model",
        ("train", "model"),
        text("Enter a base model, e.g. yolo11s.pt"),
        hint="Use a model name or browse for local weights to include in the bundle.",
    ),
    Field(
        "train_classes",
        "Class names",
        ("train", "classes"),
        optional_names,
        fmt=join,
        hint="Comma-separated in class-ID order. Blank preserves or auto-detects names.",
    ),
    Field(
        "export_onnx",
        "Export best model to ONNX after training",
        ("train", "export_onnx"),
    ),
    Field(
        "vastai_instance",
        "Vast.ai instance ID",
        ("vastai", "instance_id"),
        instance_id,
        hint="Optional. Export will ask before uploading and starting GPU training.",
    ),
    Field(
        "compress_enabled",
        "Compress images (downscale, then re-encode as JPEG)",
        ("compress", "enabled"),
    ),
    Field(
        "compress_max_size",
        "Max size (px)",
        ("compress", "max_size"),
        integer(1),
        group="compress",
        width=10,
    ),
    Field(
        "compress_jpeg_quality",
        "JPEG quality",
        ("compress", "jpeg_quality"),
        integer(1, 100),
        group="compress",
        width=10,
    ),
    Field(
        "skip_classes",
        "Skip class IDs",
        ("skip_classes",),
        class_ids,
        fmt=join,
        hint="Dropped from labels; remaining IDs are not renumbered. Images left with no labels are excluded.",
    ),
)

GROUPS: tuple[Group, ...] = (
    Group(
        "splits",
        "Splits",
        "Fractions of the dataset per split; they must total 1.0.",
        ("train", "val", "test"),
        _splits_total,
    ),
    Group(
        "quality",
        "Quality",
        "Images outside these limits are flagged in quality_flags.csv.",
        ("min_width", "min_height", "max_aspect_ratio"),
    ),
    Group(
        "compress",
        "Compression",
        "Applied only to the exported training bundle.",
        ("compress_max_size", "compress_jpeg_quality"),
    ),
)

BY_NAME: dict[str, Field] = {spec.name: spec for spec in FIELDS}
GROUP_BY_NAME: dict[str, Group] = {group.name: group for group in GROUPS}


def assign(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Write ``value`` into the nested ``config`` location named by ``path``."""
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def lookup(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value = config
    for key in path:
        value = value[key]
    return value
