from __future__ import annotations

from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

from .scanner import scan_images


def _read_label(path: Path, class_count: int) -> tuple[int, int, Counter[int], list[str]]:
    if not path.exists():
        return 0, 0, Counter(), ["missing label"]
    objects = 0
    invalid = 0
    classes: Counter[int] = Counter()
    issues: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
            if len(parts) != 5 or not 0 <= class_id < class_count or any(not isfinite(value) or not 0 <= value <= 1 for value in values) or values[2] == 0 or values[3] == 0:
                raise ValueError
            objects += 1; classes[class_id] += 1
        except (ValueError, IndexError):
            invalid += 1
            issues.append(f"invalid label line {number}")
    return objects, invalid, classes, issues


def collect_dataset_stats(dataset: Path, config: dict[str, Any]) -> dict[str, Any]:
    image_root = dataset / "images"
    records = scan_images(image_root, config["extensions"], config["workers"])
    valid = [record for record in records if not record["error"]]
    resolutions = Counter(f"{record['width']}x{record['height']}" for record in valid)
    formats = Counter(record["format"] for record in valid)
    class_counts: Counter[int] = Counter()
    missing_labels = empty_labels = invalid_labels = objects = max_objects = 0
    issues: list[dict[str, str]] = []
    for record in records:
        image = record["path"]
        try:
            split = image.parent.name
            label = dataset / "labels" / split / f"{image.stem}.txt"
            count, invalid, valid_classes, label_issues = _read_label(label, len(config["classes"]))
            if not label.exists(): missing_labels += 1
            if label.exists() and not label.read_text(encoding="utf-8", errors="replace").strip(): empty_labels += 1
            objects += count; invalid_labels += invalid; max_objects = max(max_objects, count); class_counts.update(valid_classes)
            for issue in label_issues:
                issues.append({"path": str(image), "issue": issue})
        except Exception as exc:  # Defensive: collect all integrity issues.
            issues.append({"path": str(image), "issue": str(exc)})
    expected_labels = {image.relative_to(image_root).with_suffix(".txt") for image in (record["path"] for record in records)}
    actual_labels = {label.relative_to(dataset / "labels") for label in (dataset / "labels").rglob("*.txt")} if (dataset / "labels").exists() else set()
    orphan_labels = actual_labels - expected_labels
    filenames = Counter(record["path"].name for record in records)
    duplicate_filenames = [name for name, count in filenames.items() if count > 1]
    for label in sorted(orphan_labels):
        issues.append({"path": str(dataset / "labels" / label), "issue": "label has no matching image"})
    for name in duplicate_filenames:
        issues.append({"path": name, "issue": "duplicate image filename"})
    return {
        "total_images": len(records), "valid_images": len(valid), "corrupt_images": len(records) - len(valid),
        "dataset_bytes": sum(record["file_size"] for record in records), "average_resolution": {"width": round(sum(r["width"] for r in valid) / len(valid), 1) if valid else 0, "height": round(sum(r["height"] for r in valid) / len(valid), 1) if valid else 0},
        "resolutions": dict(resolutions), "formats": dict(formats), "images_labeled": len(records) - missing_labels, "images_missing_labels": missing_labels,
        "empty_labels": empty_labels, "total_objects": objects, "objects_per_class": {config["classes"][key]: value for key, value in class_counts.items() if 0 <= key < len(config["classes"])},
        "average_objects_per_image": round(objects / len(records), 2) if records else 0, "max_objects_per_image": max_objects, "invalid_labels": invalid_labels, "issues": issues,
        "labels_missing_images": len(orphan_labels), "duplicate_filenames": len(duplicate_filenames),
    }
