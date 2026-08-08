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

In both cases, `train.py`, `requirements.txt`, and `entrypoint.sh` are written directly into a zip archive alongside the dataset files, leaving the original folder untouched. The zip is written as `<dataset_name>_train.zip` beside the input folder. No temporary copy is made — files are streamed from their original location into the zip.

`train.py` uses the ultralytics API with configurable `--model` (default `yolo11s.pt`), `--epochs` (default 100), `--imgsz` (default 640), `--batch` (default 16), and `--device` (default 0). It exports `best.pt` to ONNX after training; pass `--no-export` to skip.

On vast.ai, upload the zip, then use `entrypoint.sh` as the `--onstart` script:

```bash
vastai create instance <OFFER_ID> --image vastai/pytorch:@vastai-automatic-tag --disk 50 --ssh --direct --onstart entrypoint.sh
```

## Desktop GUI

`gui.py` provides a dark-theme interface with Build, Verify, Statistics, and Format to Train workflows. Features include inline validation, a build confirmation dialog, saved/loadable YAML settings, persistent local preferences, a random-seed control, and an image-count estimate based on the selected source folder. Folder drag-and-drop is enabled when `tkinterdnd2` is installed; Browse buttons work in every supported environment.

The output field has two buttons: **Choose location** picks a parent directory and names a new folder for Build, and **Browse existing** selects an already-built dataset or a flat folder of paired images and labels for Verify, Statistics, or Format to Train.

Every completed build includes `build_config.yaml`, `dataset.yaml`, image and label split directories, plus report files under `reports/`. The GUI checks these expected output artifacts after Build and reports any missing files.
