from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .config import load_config
from .dedupe import find_duplicates
from .report import render_report
from .scanner import discover_unsupported_files, scan_images
from .splitter import split_records
from .stats import collect_dataset_stats
from .utils import Progress, ensure_dir, report, utc_now, write_csv, write_json

TRAIN_REQUIRED_ARTIFACTS = (
    "dataset.yaml",
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
)

TRAIN_PY = """\
\"\"\"Train a YOLO object-detection model on this dataset and export to ONNX.\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL = __MODEL__
DEFAULT_EXPORT_ONNX = __EXPORT_ONNX__


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO on the bundled dataset.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1 (default: 0)")
    parser.add_argument("--export", action=argparse.BooleanOptionalAction, default=DEFAULT_EXPORT_ONNX, help="Export best.pt to ONNX after training")
    args = parser.parse_args()

    data_yaml = str(Path(__file__).resolve().parent / "dataset.yaml")
    model = YOLO(args.model)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)

    if args.export:
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        model.export(model=str(best), format="onnx")
        print(f"Exported ONNX: {best.with_suffix('.onnx')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

TRAIN_REQUIREMENTS = """\
ultralytics>=8.3.0
onnx>=1.16.0
onnxruntime-gpu>=1.18.0
"""

ENTRYPOINT_SH = """\
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying GPU..."
nvidia-smi || echo "WARNING: nvidia-smi not found; training may fall back to CPU"

echo "Starting training..."
python train.py "$@"
"""


def _validate_train_dataset(dataset: Path) -> list[str]:
    missing = [
        artifact
        for artifact in TRAIN_REQUIRED_ARTIFACTS
        if not (dataset / artifact).exists()
    ]
    return missing


_TRAIN_FILES = {
    "train.py": TRAIN_PY,
    "requirements.txt": TRAIN_REQUIREMENTS,
    "entrypoint.sh": ENTRYPOINT_SH,
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _filter_label_text(text: str, skip_ids: set[int]) -> str:
    """Drop YOLO label lines whose class id is in skip_ids. IDs are not renumbered."""
    if not skip_ids:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            cid = int(stripped.split()[0])
        except (ValueError, IndexError):
            kept.append(line)
            continue
        if cid not in skip_ids:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def _read_label_filtered(path: Path, skip_ids: set[int]) -> str:
    return _filter_label_text(
        path.read_text(encoding="utf-8", errors="replace"), skip_ids
    )


def _is_built_dataset(folder: Path) -> bool:
    return (folder / "images" / "train").is_dir() and (
        folder / "labels" / "train"
    ).is_dir()


def _discover_split_first(folder: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Find image/label pairs in a split-first layout: {split}/images, {split}/labels."""
    splits: dict[str, list[tuple[Path, Path]]] = {}
    for child in sorted(folder.iterdir(), key=lambda p: str(p).lower()):
        if not child.is_dir():
            continue
        images_dir = child / "images"
        labels_dir = child / "labels"
        if not (images_dir.is_dir() and labels_dir.is_dir()):
            continue
        pairs: list[tuple[Path, Path]] = []
        for image_path in sorted(images_dir.rglob("*"), key=lambda p: str(p).lower()):
            if image_path.is_file() and image_path.suffix.lower() in _IMAGE_EXTS:
                label_path = labels_dir / image_path.relative_to(
                    images_dir
                ).with_suffix(".txt")
                if label_path.exists():
                    pairs.append((image_path, label_path))
        if pairs:
            splits[child.name] = pairs
    return splits


def _read_dataset_yaml(folder: Path) -> dict[str, Any] | None:
    for name in ("dataset.yaml", "data.yaml"):
        path = folder / name
        if path.exists():
            with path.open("r", encoding="utf-8") as stream:
                return yaml.safe_load(stream) or {}
    return None


def _detect_classes(label_paths: list[Path]) -> list[str]:
    ids: set[int] = set()
    for path in label_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    ids.add(int(line.split()[0]))
                except (ValueError, IndexError):
                    pass
    return [f"class_{i}" for i in sorted(ids)]


def _discover_flat_pairs(folder: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for path in sorted(folder.rglob("*"), key=lambda p: str(p).lower()):
        if path.suffix.lower() in _IMAGE_EXTS:
            label = path.with_suffix(".txt")
            if label.exists():
                pairs.append((path, label))
    return pairs


def _compress_image_bytes(image_path: Path, max_size: int, jpeg_quality: int) -> bytes:
    """Downscale (longest side <= max_size) and re-encode an image as JPEG."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(max_size) / float(max(width, height)))
        if scale < 1.0:
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return buffer.getvalue()


def _write_train_files(
    archive: zipfile.ZipFile, model: str, export_onnx: bool = True
) -> None:
    model_path = Path(model).expanduser()
    default_model = model_path.name if model_path.is_file() else model
    for name, content in _TRAIN_FILES.items():
        if name == "train.py":
            content = content.replace("__MODEL__", repr(default_model)).replace(
                "__EXPORT_ONNX__", repr(export_onnx)
            )
        info = zipfile.ZipInfo(name)
        info.compress_type = zipfile.ZIP_DEFLATED
        if name == "entrypoint.sh":
            info.external_attr = 0o755 << 16
        archive.writestr(info, content)
    if model_path.is_file() and model_path.name not in archive.namelist():
        archive.write(model_path, model_path.name)


def _write_vastai_script(zip_path: Path) -> Path:
    """Write the native launcher that uploads this bundle and starts it remotely."""
    if sys.platform == "win32":
        path = zip_path.with_suffix(".ps1")
        content = r'''param([Parameter(Mandatory)][string]$InstanceId)
$ErrorActionPreference = "Stop"
$archive = [IO.Path]::ChangeExtension($PSCommandPath, ".zip")
vastai copy "local:$archive" "${InstanceId}:/workspace/dataset_train.zip"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
vastai execute $InstanceId 'run_dir=/workspace/training-$(date +%Y%m%d-%H%M%S) && mkdir -p "$run_dir" && python -m zipfile -e /workspace/dataset_train.zip "$run_dir" && ln -sfn "$run_dir" /workspace/training-latest && cd "$run_dir" && nohup bash entrypoint.sh > training.log 2>&1 < /dev/null &'
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Write-Host "Training started. Check: vastai execute $InstanceId 'tail -f /workspace/training-latest/training.log'"
'''
    else:
        path = zip_path.with_suffix(".sh")
        content = r'''#!/usr/bin/env sh
set -eu
instance_id="${1:?Usage: $0 INSTANCE_ID}"
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
archive="$script_dir/$(basename "${0%.*}").zip"
vastai copy "local:$archive" "${instance_id}:/workspace/dataset_train.zip"
vastai execute "$instance_id" 'run_dir=/workspace/training-$(date +%Y%m%d-%H%M%S) && mkdir -p "$run_dir" && python -m zipfile -e /workspace/dataset_train.zip "$run_dir" && ln -sfn "$run_dir" /workspace/training-latest && cd "$run_dir" && nohup bash entrypoint.sh > training.log 2>&1 < /dev/null &'
echo "Training started. Check: vastai execute $instance_id 'tail -f /workspace/training-latest/training.log'"
'''
    path.write_text(content, encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(path.stat().st_mode | 0o111)
    return path


def _zip_built_dataset(
    dataset: Path,
    zip_path: Path,
    compress: dict[str, Any],
    model: str,
    export_onnx: bool,
    classes: list[str],
    skip_ids: set[int],
    progress: Progress | None = None,
) -> tuple[Path, int]:
    zip_path = zip_path.with_suffix(".zip")
    kept = 0
    queued: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.rglob("*"), key=lambda p: str(p).lower()):
            if image_path.is_file() and image_path.suffix.lower() in _IMAGE_EXTS:
                relative = image_path.relative_to(images_dir).with_suffix(".txt")
                queued.append((image_path, labels_dir / relative))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for done, (image_path, label_path) in enumerate(queued, start=1):
            report(progress, "Bundling images", done, len(queued))
            rel_img = image_path.relative_to(dataset)
            filtered = None
            if skip_ids and label_path.is_file():
                filtered = _read_label_filtered(label_path, skip_ids)
                if not filtered.strip():
                    continue
            if compress["enabled"]:
                data = _compress_image_bytes(
                    image_path,
                    int(compress["max_size"]),
                    int(compress["jpeg_quality"]),
                )
                archive.writestr(rel_img.with_suffix(".jpg").as_posix(), data)
            else:
                archive.write(image_path, rel_img)
            if label_path.is_file():
                rel_lbl = label_path.relative_to(dataset)
                if filtered is not None:
                    archive.writestr(rel_lbl.as_posix(), filtered)
                else:
                    archive.write(label_path, rel_lbl)
            kept += 1
        # Non-image/label artifacts (dataset.yaml, reports, configs).
        for path in dataset.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(dataset)
            parts = rel.parts
            if len(parts) >= 2 and parts[0] in ("images", "labels"):
                continue
            if rel.as_posix() == "dataset.yaml":
                continue
            archive.write(path, rel)
        data = _read_dataset_yaml(dataset) or {}
        data["path"] = "."
        if classes:
            data["names"] = {index: name for index, name in enumerate(classes)}
        archive.writestr("dataset.yaml", yaml.safe_dump(data, sort_keys=False))
        _write_train_files(archive, model, export_onnx)
    return zip_path, kept


def _zip_split_first_dataset(
    dataset: Path,
    splits: dict[str, list[tuple[Path, Path]]],
    classes: list[str],
    zip_path: Path,
    compress: dict[str, Any],
    model: str,
    export_onnx: bool,
    skip_ids: set[int],
    progress: Progress | None = None,
) -> tuple[Path, int]:
    zip_path = zip_path.with_suffix(".zip")
    total = 0
    queued = sum(len(pairs) for pairs in splits.values())
    done = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for split_name, pairs in splits.items():
            for image_path, label_path in pairs:
                done += 1
                report(progress, "Bundling images", done, queued)
                stem = image_path.stem
                if skip_ids:
                    filtered = _read_label_filtered(label_path, skip_ids)
                    if not filtered.strip():
                        continue
                if compress["enabled"]:
                    data = _compress_image_bytes(
                        image_path,
                        int(compress["max_size"]),
                        int(compress["jpeg_quality"]),
                    )
                    archive.writestr(f"{split_name}/images/{stem}.jpg", data)
                else:
                    ext = image_path.suffix.lower()
                    archive.write(image_path, f"{split_name}/images/{stem}{ext}")
                if skip_ids:
                    archive.writestr(f"{split_name}/labels/{stem}.txt", filtered)
                else:
                    archive.write(label_path, f"{split_name}/labels/{stem}.txt")
                total += 1
        data = {
            "path": ".",
            **{name: f"{name}/images" for name in splits},
            "names": {index: name for index, name in enumerate(classes)},
        }
        archive.writestr("dataset.yaml", yaml.safe_dump(data, sort_keys=False))
        _write_train_files(archive, model, export_onnx)
    return zip_path, total


def _zip_flat_dataset(
    pairs: list[tuple[Path, Path]],
    classes: list[str],
    splits: dict[str, float],
    seed: int,
    zip_path: Path,
    compress: dict[str, Any],
    model: str,
    export_onnx: bool,
    skip_ids: set[int],
    progress: Progress | None = None,
) -> Path:
    # Drop pairs whose label becomes empty after class filtering, before
    # computing splits so train/val/test ratios stay proportional.
    filtered_pairs: list[tuple[Path, Path, str]] = []
    for image_path, label_path in pairs:
        if skip_ids:
            filtered = _read_label_filtered(label_path, skip_ids)
            if not filtered.strip():
                continue
            filtered_pairs.append((image_path, label_path, filtered))
        else:
            filtered_pairs.append((image_path, label_path, None))

    indices = list(range(len(filtered_pairs)))
    random.Random(seed).shuffle(indices)
    total = len(indices)
    train_end = round(total * float(splits["train"]))
    val_end = train_end + round(total * float(splits["val"]))
    split_map = {}
    split_map.update(dict.fromkeys(indices[:train_end], "train"))
    split_map.update(dict.fromkeys(indices[train_end:val_end], "val"))
    split_map.update(dict.fromkeys(indices[val_end:], "test"))

    zip_path = zip_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        counter = 1
        for i, (image_path, label_path, filtered) in enumerate(filtered_pairs):
            report(progress, "Bundling images", i + 1, len(filtered_pairs))
            split = split_map[i]
            stem = f"{counter:06d}"
            counter += 1
            if compress["enabled"]:
                data = _compress_image_bytes(
                    image_path,
                    int(compress["max_size"]),
                    int(compress["jpeg_quality"]),
                )
                archive.writestr(f"images/{split}/{stem}.jpg", data)
            else:
                ext = image_path.suffix.lower()
                archive.write(image_path, f"images/{split}/{stem}{ext}")
            if filtered is not None:
                archive.writestr(f"labels/{split}/{stem}.txt", filtered)
            else:
                archive.write(label_path, f"labels/{split}/{stem}.txt")

        data = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(classes)},
        }
        archive.writestr("dataset.yaml", yaml.safe_dump(data, sort_keys=False))
        _write_train_files(archive, model, export_onnx)
    return zip_path


def format_to_train(config: dict[str, Any], progress: Progress | None = None) -> int:
    dataset = Path(config["output_dir"])
    if not dataset.is_dir():
        raise FileNotFoundError(f"output_dir does not exist: {dataset}")
    dataset = dataset.resolve()
    model = config["train"]["model"]
    export_onnx = config["train"].get("export_onnx", True)
    chosen_classes = config["train"].get("classes", [])
    skip_ids = {int(cid) for cid in config.get("skip_classes", [])}
    report(progress, "Inspecting dataset")

    if _is_built_dataset(dataset):
        missing = _validate_train_dataset(dataset)
        if missing:
            raise ValueError(
                f"Dataset is missing required artifacts for training: {', '.join(missing)}"
            )
        zip_path, kept = _zip_built_dataset(
            dataset,
            dataset.parent / f"{dataset.name}_train",
            config["compress"],
            model,
            export_onnx,
            chosen_classes,
            skip_ids,
            progress,
        )
        script_path = _write_vastai_script(zip_path)
        print(
            f"Training bundle created: {zip_path} ({kept} images, "
            f"{zip_path.stat().st_size / (1024 * 1024):.1f} MB, model={model}); "
            f"Vast.ai launcher: {script_path}"
        )
        return 0

    split_first = _discover_split_first(dataset)
    if split_first:
        existing = _read_dataset_yaml(dataset)
        if chosen_classes:
            classes = chosen_classes
        elif existing and isinstance(existing.get("names"), list):
            classes = existing["names"]
        elif existing and isinstance(existing.get("names"), dict):
            classes = [existing["names"][i] for i in sorted(existing["names"])]
        else:
            all_labels = [label for pairs in split_first.values() for _, label in pairs]
            classes = _detect_classes(all_labels)
        zip_path, total = _zip_split_first_dataset(
            dataset,
            split_first,
            classes,
            dataset.parent / f"{dataset.name}_train",
            config["compress"],
            model,
            export_onnx,
            skip_ids,
            progress,
        )
        script_path = _write_vastai_script(zip_path)
        print(
            f"Training bundle created: {zip_path} ({total} images, "
            f"{zip_path.stat().st_size / (1024 * 1024):.1f} MB, model={model}); "
            f"Vast.ai launcher: {script_path}"
        )
        return 0

    pairs = _discover_flat_pairs(dataset)
    if not pairs:
        raise ValueError(
            f"No paired image+label files found in {dataset}. "
            "Expected images with sibling .txt label files."
        )
    classes = chosen_classes or _detect_classes([label for _, label in pairs])
    zip_path = _zip_flat_dataset(
        pairs,
        classes,
        config["splits"],
        int(config["seed"]),
        dataset.parent / f"{dataset.name}_train",
        config["compress"],
        model,
        export_onnx,
        skip_ids,
        progress,
    )
    script_path = _write_vastai_script(zip_path)
    kept = sum(
        1
        for _, label in pairs
        if not skip_ids or _read_label_filtered(label, skip_ids).strip()
    )
    print(
        f"Training bundle created: {zip_path} ({kept} images, "
        f"{zip_path.stat().st_size / (1024 * 1024):.1f} MB, model={model}); "
        f"Vast.ai launcher: {script_path}"
    )
    return 0


def quality_flags(record: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if record["error"]:
        return ["corrupt"]
    quality = config["quality"]
    flags: list[str] = []
    if (
        record["width"] < quality["min_width"]
        or record["height"] < quality["min_height"]
    ):
        flags.append("very small")
    ratio = max(record["width"] / record["height"], record["height"] / record["width"])
    if ratio > quality["max_aspect_ratio"]:
        flags.append("extreme aspect ratio")
    return flags


def write_dataset_yaml(
    output: Path, classes: list[str], dataset_path: Path | None = None
) -> None:
    data = {
        "path": str((dataset_path or output).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(classes)},
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


def build(config: dict[str, Any], progress: Progress | None = None) -> int:
    source, output = Path(config["source_dir"]), Path(config["output_dir"])
    if not source.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source}")
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError(
            "output_dir must be separate from source_dir; neither path may contain the other."
        )
    if output.exists():
        raise FileExistsError(
            f"output_dir already exists and will not be replaced: {output}. Choose a new output path."
        )
    ensure_dir(output.parent)
    staging = Path(tempfile.mkdtemp(prefix=f"{output.name}-build-"))
    try:
        records = scan_images(
            source, config["extensions"], config["workers"], progress
        )
        clean, duplicates = find_duplicates(
            records,
            int(config["near_duplicate_threshold"]),
            config["workers"],
            progress,
        )
        split_data = split_records(clean, config["splits"], int(config["seed"]))
        for split in split_data:
            ensure_dir(staging / "images" / split)
            ensure_dir(staging / "labels" / split)
        name = 1
        to_copy = sum(len(items) for items in split_data.values())
        for split, items in split_data.items():
            for record in items:
                report(progress, "Copying images", name, to_copy)
                source_image = record["path"]
                destination = (
                    staging
                    / "images"
                    / split
                    / f"{name:06d}{source_image.suffix.lower()}"
                )
                shutil.copy2(source_image, destination)
                source_label = source_image.with_suffix(".txt")
                destination_label = staging / "labels" / split / f"{name:06d}.txt"
                if source_label.exists():
                    shutil.copy2(source_label, destination_label)
                elif config["create_empty_labels"]:
                    destination_label.touch(exist_ok=True)
                name += 1
        flags = [
            {
                "path": str(record["path"]),
                "flags": "; ".join(quality_flags(record, config)),
            }
            for record in records
            if quality_flags(record, config)
        ]
        flags.extend(
            {"path": str(path), "flags": "unsupported format"}
            for path in discover_unsupported_files(source, config["extensions"])
        )
        report(progress, "Writing reports")
        ensure_dir(staging / "reports")
        write_csv(
            staging / "reports" / "duplicates.csv",
            duplicates,
            ["duplicate", "kept", "kind"],
        )
        write_csv(staging / "reports" / "quality_flags.csv", flags, ["path", "flags"])
        (staging / "build_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        write_dataset_yaml(staging, config["classes"], output)
        stats = collect_dataset_stats(staging, config)
        stats.update(
            {
                "source_images": len(records),
                "source_corrupt_images": sum(
                    bool(record["error"]) for record in records
                ),
                "unique_images": len(clean),
                "duplicates_removed": len(duplicates),
                "quality_flags": len(flags),
                "generated_at": utc_now(),
            }
        )
        write_json(staging / "reports" / "stats.json", stats)
        write_json(
            staging / "reports" / "build_manifest.json",
            {
                "required_artifacts": [
                    "build_config.yaml",
                    "dataset.yaml",
                    "images",
                    "labels",
                    "reports/stats.json",
                    "reports/report.html",
                    "reports/duplicates.csv",
                    "reports/quality_flags.csv",
                ],
                "generated_at": utc_now(),
            },
        )
        render_report(staging / "reports" / "report.html", stats, duplicates)
        if output.exists():
            raise FileExistsError(
                f"output_dir already exists and will not be replaced: {output}. Choose a new output path."
            )
        shutil.move(str(staging), str(output))
    except Exception:
        shutil.rmtree(
            staging, ignore_errors=True
        )  # Safe: this process created the staging directory.
        raise
    print(f"Built {len(clean)} unique images in {output}")
    return 0


def verify(config: dict[str, Any], progress: Progress | None = None) -> int:
    output = Path(config["output_dir"])
    if not output.is_dir():
        raise FileNotFoundError(f"output_dir does not exist: {output}")
    report(progress, "Verifying dataset")
    stats = collect_dataset_stats(output, config)
    write_json(output / "reports" / "stats.json", stats)
    render_report(output / "reports" / "report.html", stats)
    problems = (
        stats["corrupt_images"]
        + stats["images_missing_labels"]
        + stats["invalid_labels"]
        + stats["labels_missing_images"]
        + stats["duplicate_filenames"]
    )
    print(
        f"Verification: {stats['total_images']} images, {problems} issue(s) (missing labels: {stats['images_missing_labels']}; orphan labels: {stats['labels_missing_images']})"
    )
    return 1 if problems else 0


def stats(config: dict[str, Any], progress: Progress | None = None) -> int:
    output = Path(config["output_dir"])
    if not output.is_dir():
        raise FileNotFoundError(f"output_dir does not exist: {output}")
    report(progress, "Collecting statistics")
    data = collect_dataset_stats(output, config)
    write_json(output / "reports" / "stats.json", data)
    render_report(output / "reports" / "report.html", data)
    print(f"Statistics regenerated for {data['total_images']} image(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a clean YOLO dataset.")
    parser.add_argument(
        "command", choices=["build", "verify", "stats", "format-to-train"]
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration (default: config.yaml)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    return {
        "build": build,
        "verify": verify,
        "stats": stats,
        "format-to-train": format_to_train,
    }[args.command](config)


if __name__ == "__main__":
    raise SystemExit(main())
