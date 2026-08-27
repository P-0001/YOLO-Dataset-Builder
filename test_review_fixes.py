import io
import queue
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from dataset_builder import builder
from dataset_builder.builder import _write_train_files, _write_vastai_script
from dataset_builder.ui import app as app_module
from dataset_builder.ui.app import DatasetBuilderGui
from dataset_builder.ui.fields import number


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

        self.assertEqual(app.root.value, "1180x900+0+180")

    def test_local_model_and_native_vastai_launcher_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "base.pt"
            model.write_bytes(b"weights")
            bundle = root / "dataset_train.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                _write_train_files(archive, str(model))

            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(archive.read("base.pt"), b"weights")
                trainer = archive.read("train.py").decode()
                self.assertIn("DEFAULT_MODEL = 'base.pt'", trainer)
                self.assertIn("DEFAULT_EXPORT_ONNX = True", trainer)

            without_onnx = root / "without_onnx.zip"
            with zipfile.ZipFile(without_onnx, "w") as archive:
                _write_train_files(archive, "yolo11m.pt", False)
            with zipfile.ZipFile(without_onnx) as archive:
                self.assertIn(
                    "DEFAULT_EXPORT_ONNX = False",
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
                "yolo11m.pt",
                True,
                ["cat", "dog"],
                set(),
            )

            with zipfile.ZipFile(bundle) as archive:
                exported = builder.yaml.safe_load(archive.read("dataset.yaml"))
            self.assertEqual(exported["path"], ".")
            self.assertEqual(exported["names"], {0: "cat", 1: "dog"})


if __name__ == "__main__":
    unittest.main()
