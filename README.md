# YOLO Dataset Builder

A lightweight CLI and desktop GUI that scans image folders, detects corrupt and duplicate images, creates deterministic YOLO splits, and writes validation reports.

## Setup

```bash
python -m pip install -r requirements.txt
```

Edit `config.yaml`: set `source_dir`, `output_dir`, and your class names. Images may have sibling YOLO `.txt` labels; they are copied and renamed to match their output image. Otherwise empty labels are created by default.

## Commands

```bash
python builder.py build
python builder.py verify
python builder.py stats
python builder.py format-to-train
python gui.py
```

Use `--config path/to/config.yaml` to select a different configuration. The build writes `dataset.yaml` and reports under `output_dir/reports/`.

Quality issues are flagged in `quality_flags.csv`; corrupt files are excluded and duplicate files are excluded from the output. The builder never modifies source images or overwrites an existing output directory; select a new output path for every build.

## Deduplication

Exact duplicates are detected via SHA-256 file hashing. Near duplicates are detected via a DCT-based perceptual hash (pHash): each image is resized to 32x32 grayscale, transformed with a 2D DCT, and the top-left 8x8 low-frequency coefficients are thresholded against their median to produce a 64-bit hash. Two images are considered near duplicates when their pHash values differ in at most `near_duplicate_threshold` bits (0-7, default 4) and their mean RGB colours are within 16 of each other.

The mean-colour guard prevents unrelated uniformly coloured images from being collapsed. The matching index uses an 8-block bucketing scheme that guarantees no missed matches for any threshold up to 7 while avoiding O(n^2) scans.

## Format to Train

`format-to-train` prepares a dataset for YOLO training on vast.ai. It accepts two kinds of input:

1. **Built dataset** — a folder with `dataset.yaml`, `images/{train,val,test}`, and `labels/{train,val,test}` (produced by `build`).
2. **Flat folder** — a folder of paired images and sibling `.txt` label files (e.g. `image1.png` + `image1.txt`). Class IDs are auto-detected from the label files, and train/val/test splits are created using the ratios and seed from `config.yaml`.

In both cases, `train.py`, `requirements.txt`, and `entrypoint.sh` are written directly into a zip archive alongside the dataset files, leaving the original folder untouched. A selected local base-model file is included too. The zip is written as `<dataset_name>_train.zip` beside the input folder. No temporary copy is made — files are streamed from their original location into the zip.

`train.py` uses the ultralytics API with configurable `--model` (default `yolo11m.pt`), `--epochs` (default 100), `--imgsz` (default 640), `--batch` (default 16), and `--device` (default 0). Exporting `best.pt` to ONNX defaults on; the GUI checkbox controls the bundled default, and `--export` / `--no-export` can override it at runtime.

Export also writes a platform-native Vast.ai launcher beside the zip (`.ps1` on Windows, `.sh` elsewhere). Pass it a running instance ID; it uploads the bundle and starts training remotely with `nohup`, so closing the local shell does not stop training:

```bash
./dataset_train.sh <INSTANCE_ID>
# Windows: .\dataset_train.ps1 <INSTANCE_ID>
```

In the GUI, enter the optional Vast.ai instance ID on the Export page. When you export, a billing-aware confirmation asks before the app runs that launcher; choosing No still creates the local bundle and script.

The optional Export **Class names** field accepts comma-separated names in class-ID order. Leave it blank to preserve names from `dataset.yaml` or auto-detect generic names from label IDs.

## Desktop GUI

`gui.py` is a thin entry point for `dataset_builder.ui`, a dark-theme interface with Build, Verify, Statistics, and Format to Train workflows. It shows live progress and streams each command's output into an activity log, so the same messages the CLI prints appear in the window.

Features: compact no-scroll forms, inline validation, a build confirmation dialog, saved/loadable YAML settings, persistent local preferences, a random-seed control, an image-count estimate for the selected source folder, and a draggable split between the form and the activity log. Folder drag-and-drop is enabled when `tkinterdnd2` is installed; Browse buttons work everywhere.

Keyboard shortcuts: `Ctrl+B` build, `Ctrl+R` verify, `Ctrl+T` statistics, `Ctrl+E` format to train, `Ctrl+O` open the dataset folder, `Ctrl+S` save config, `Ctrl+Shift+O` load config, `Ctrl+L` clear the log, `F5` recount the estimate.

The dataset field has two buttons: **Choose location** picks a parent directory and names a new folder for Build, and **Browse existing** selects an already-built dataset or a flat folder of paired images and labels for Verify, Statistics, or Format to Train.

Every completed build includes `build_config.yaml`, `dataset.yaml`, image and label split directories, plus report files under `reports/`. The GUI checks these expected output artifacts after Build and reports any missing files.

### Adding a setting

Form fields are declared once in `dataset_builder/ui/fields.py`. A single `Field` entry names the control, its config key path, and how to parse it; that one entry drives the widget, its validation message, the built config, YAML load/save, saved preferences, and reset-to-defaults.
