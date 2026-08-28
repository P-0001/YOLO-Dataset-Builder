import io
import queue
import tempfile
import time
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from dataset_builder import builder
from dataset_builder.builder import _write_train_files, _write_vastai_script
from dataset_builder.config import DEFAULT_CONFIG, validate_config
from dataset_builder.ui import app as app_module
from dataset_builder.ui.app import DatasetBuilderGui
from dataset_builder.ui.fields import FIELDS, lookup, number


class Value:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value


class Root:
    def after(self, *_):
        pass


class DisplayRoot:
    def update_idletasks(self):
        pass

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def winfo_width(self):
        return 3000

    def winfo_height(self):
        return 2000

    def winfo_x(self):
        return -100

    def winfo_y(self):
        return 2000

    def geometry(self, value=None):
        if value is None:
            return "3000x2000-100+2000"
        self.value = value


class ReviewFixesTest(unittest.TestCase):
    def test_number_rejects_non_finite_values(self):
        parse = number(0, 1)
        for raw in ("nan", "inf", "-inf"):
            with self.assertRaises(ValueError):
                parse(raw)

    def test_stale_estimate_cannot_discard_latest_result(self):
        app = DatasetBuilderGui.__new__(DatasetBuilderGui)
        app._output = queue.Queue()
        app._estimate_results = queue.Queue()
        app._estimate_results.put((2, 10, 1024))
        app._estimate_results.put((1, 20, 2048))
        app._estimate_token = 2
        app.estimate = Value("Counting source images…")
        app._running = False
        app._outcome = None
        app.root = Root()
        app.log_line = lambda *_: None

        app._poll()

        self.assertIn("10 images", app.estimate.value)

    def test_restored_window_is_clamped_to_display(self):
        app = DatasetBuilderGui.__new__(DatasetBuilderGui)
        app.root = DisplayRoot()
        app._saved_geometry = app.root.geometry()

        app._fit_to_display()

        self.assertEqual(app.root.value, "1280x960+0+120")

    def test_progress_estimates_time_remaining(self):
        app = DatasetBuilderGui.__new__(DatasetBuilderGui)
        app._progress_state = ("Bundling images", 50, 100)
        app._phase_start_done = 0
        app._phase_started_at = time.monotonic() - 10

        self.assertEqual(app._remaining(), "0:00:10")

    def test_sample_yaml_contains_every_ui_setting(self):
        sample = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        validate_config(sample)
        for spec in FIELDS:
            with self.subTest(field=spec.name):
                lookup(sample, spec.path)

    def test_local_model_and_native_vastai_launcher_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "base.pt"
            model.write_bytes(b"weights")
            bundle = root / "dataset_train.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                train = {
                    "model": str(model),
                    "classes": [],
                    "epochs": 25,
                    "imgsz": 960,
                    "batch": 8,
                    "device": "0,1",
                    "export_onnx": True,
                }
                _write_train_files(archive, train)

            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(archive.read("base.pt"), b"weights")
                trainer = archive.read("train.py").decode()
                compile(trainer, "train.py", "exec")
                self.assertIn("'model': 'base.pt'", trainer)
                self.assertIn("'epochs': 25", trainer)
                self.assertIn("'imgsz': 960", trainer)
                self.assertIn("'batch': 8", trainer)
                self.assertIn("'device': '0,1'", trainer)
                self.assertIn("'export_onnx': True", trainer)

            without_onnx = root / "without_onnx.zip"
            with zipfile.ZipFile(without_onnx, "w") as archive:
                train["model"] = "yolo11m.pt"
                train["export_onnx"] = False
                _write_train_files(archive, train)
            with zipfile.ZipFile(without_onnx) as archive:
                self.assertIn(
                    "'export_onnx': False",
                    archive.read("train.py").decode(),
                )

            with patch.object(builder.sys, "platform", "win32"):
                launcher = _write_vastai_script(bundle)
            self.assertEqual(launcher.suffix, ".ps1")
            script = launcher.read_text(encoding="utf-8")
            self.assertIn("vastai copy", script)
            self.assertIn("nohup bash entrypoint.sh", script)

            with patch.object(builder.sys, "platform", "linux"):
                launcher = _write_vastai_script(bundle)
            self.assertEqual(launcher.suffix, ".sh")
            self.assertIn('instance_id="${1:', launcher.read_text(encoding="utf-8"))

    def test_gui_runs_native_vastai_launcher_with_instance_id(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            config = {
                "output_dir": str(dataset),
                "vastai": {"instance_id": "12345"},
            }
            completed = app_module.subprocess.CompletedProcess(
                [], 0, "Training started.\n", ""
            )
            output = io.StringIO()

            with (
                patch.object(app_module.sys, "platform", "win32"),
                patch.object(app_module.subprocess, "run", return_value=completed) as run,
            ):
                DatasetBuilderGui._launch_vastai(config, output)

            command = run.call_args.args[0]
            self.assertEqual(command[-1], "12345")
            self.assertTrue(command[-2].endswith("dataset_train.ps1"))
            self.assertIn("Training started.", output.getvalue())

    def test_exported_class_names_override_built_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            for split in ("train", "val", "test"):
                (dataset / "images" / split).mkdir(parents=True)
                (dataset / "labels" / split).mkdir(parents=True)
            (dataset / "dataset.yaml").write_text(
                builder.yaml.safe_dump(
                    {
                        "path": str(dataset),
                        "train": "images/train",
                        "val": "images/val",
                        "test": "images/test",
                        "names": {0: "old"},
                    }
                ),
                encoding="utf-8",
            )

            bundle, _ = builder._zip_built_dataset(
                dataset,
                dataset.parent / "dataset_train",
                {"enabled": False, "max_size": 1280, "jpeg_quality": 85},
                {
                    "model": "yolo11m.pt",
                    "classes": [],
                    "epochs": 100,
                    "imgsz": 640,
                    "batch": 16,
                    "device": "0",
                    "export_onnx": True,
                },
                ["cat", "dog"],
                set(),
                0,
            )

            with zipfile.ZipFile(bundle) as archive:
                exported = builder.yaml.safe_load(archive.read("dataset.yaml"))
            self.assertEqual(exported["path"], ".")
            self.assertEqual(exported["names"], {0: "cat", 1: "dog"})

    def test_flat_export_uses_training_defaults_and_stores_images(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "flat"
            dataset.mkdir()
            for index in range(3):
                image = dataset / f"image-{index}.png"
                Image.new("RGB", (32, 32), (index * 20, 0, 0)).save(image)
                image.with_suffix(".txt").write_text(
                    "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )
            config = deepcopy(DEFAULT_CONFIG)
            config["output_dir"] = str(dataset)
            config["workers"] = 2
            config["train"].update(
                {"epochs": 7, "imgsz": 320, "batch": 4, "device": "cpu"}
            )
            phases = []

            self.assertEqual(
                builder.format_to_train(
                    config, lambda phase, done, total: phases.append((phase, done, total))
                ),
                0,
            )

            with zipfile.ZipFile(dataset.parent / "flat_train.zip") as archive:
                trainer = archive.read("train.py").decode()
                images = [
                    info
                    for info in archive.infolist()
                    if info.filename.startswith("images/")
                ]
            self.assertIn("'epochs': 7", trainer)
            self.assertIn("'imgsz': 320", trainer)
            self.assertIn("'batch': 4", trainer)
            self.assertIn("'device': 'cpu'", trainer)
            self.assertTrue(images)
            self.assertTrue(
                all(info.compress_type == zipfile.ZIP_STORED for info in images)
            )
            self.assertTrue(
                any(phase == "Compressing and bundling images" for phase, _, _ in phases)
            )


if __name__ == "__main__":
    unittest.main()
