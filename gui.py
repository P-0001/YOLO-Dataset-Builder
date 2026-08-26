"""A dark local desktop interface for the dataset-builder CLI."""

from __future__ import annotations

import copy
import json
import os
import secrets
import sys
import threading
import webbrowser
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tkinter import (
    BooleanVar,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    simpledialog,
    ttk,
)
from tkinter.scrolledtext import ScrolledText
from typing import Any

import yaml

from dataset_builder.builder import build, format_to_train, stats, verify
from dataset_builder.config import DEFAULT_CONFIG, load_config, validate_config


class DatasetBuilderGui:
    """A polished, keyboard-friendly front end for local dataset operations."""

    BG, SURFACE, SURFACE_ALT = "#130d23", "#211735", "#2b1e43"
    BORDER, TEXT, MUTED = "#493567", "#f5efff", "#bcaed4"
    ACCENT, ACCENT_HOVER, ACCENT_DARK = "#a855f7", "#c084fc", "#7135b9"
    SUCCESS, WARNING, DANGER, LOG_BG = "#65d6a4", "#f6c56b", "#fb7185", "#100b1c"
    APP_NAME = "YoloDatasetBuilder"
    LEGACY_PREFERENCES_FILE = Path(__file__).with_name(".dataset_builder_gui.json")

    @classmethod
    def _app_data_dir(cls) -> Path:
        if os.name == "nt":
            base = os.environ.get("APPDATA") or str(Path.home())
        elif sys.platform == "darwin":
            base = str(Path.home() / "Library" / "Application Support")
        else:
            base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / cls.APP_NAME

    @classmethod
    def _preferences_file(cls) -> Path:
        directory = cls._app_data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "preferences.json"

    REQUIRED_OUTPUT_ARTIFACTS = (
        "build_config.yaml",
        "dataset.yaml",
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "reports/build_manifest.json",
        "reports/duplicates.csv",
        "reports/quality_flags.csv",
        "reports/report.html",
        "reports/stats.json",
    )

    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("YOLO Dataset Builder")
        root.geometry("1120x960")
        root.minsize(900, 760)
        root.configure(background=self.BG)
        self._configure_styles()
        self.config_base = copy.deepcopy(DEFAULT_CONFIG)
        self.source, self.output = StringVar(), StringVar()
        self.classes = StringVar(value="object")
        self.train, self.val, self.test = (
            StringVar(value="0.8"),
            StringVar(value="0.1"),
            StringVar(value="0.1"),
        )
        self.seed, self.status = StringVar(value="42"), StringVar(value="Ready")
        self.empty_labels = BooleanVar(value=True)
        # Newly exposed config fields (kept in sync with config_base at build time).
        self.extensions = StringVar(
            value=", ".join(DEFAULT_CONFIG["extensions"])
        )
        self.workers = StringVar(value=str(DEFAULT_CONFIG["workers"]))
        self.near_duplicate_threshold = StringVar(
            value=str(DEFAULT_CONFIG["near_duplicate_threshold"])
        )
        self.min_width = StringVar(
            value=str(DEFAULT_CONFIG["quality"]["min_width"])
        )
        self.min_height = StringVar(
            value=str(DEFAULT_CONFIG["quality"]["min_height"])
        )
        self.max_aspect_ratio = StringVar(
            value=str(DEFAULT_CONFIG["quality"]["max_aspect_ratio"])
        )
        self.compress_enabled = BooleanVar(
            value=bool(DEFAULT_CONFIG["compress"]["enabled"])
        )
        self.compress_max_size = StringVar(
            value=str(DEFAULT_CONFIG["compress"]["max_size"])
        )
        self.compress_jpeg_quality = StringVar(
            value=str(DEFAULT_CONFIG["compress"]["jpeg_quality"])
        )
        self.train_model = StringVar(
            value=str(DEFAULT_CONFIG["train"]["model"])
        )
        self.skip_classes = StringVar(value="")
        self.buttons: list[ttk.Button] = []
        self.action_buttons: list[ttk.Button] = []
        self.fields: dict[str, ttk.Entry] = {}
        self.field_errors: dict[str, ttk.Label] = {}
        self.metrics: dict[str, ttk.Label] = {}
        self._estimate_token = 0
        self._restore_preferences()

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        content = ttk.Frame(root, style="App.TFrame", padding=(28, 24))
        content.grid(sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        self._make_header(content)

        self.notebook = ttk.Notebook(content, style="App.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(20, 14))
        self._build_tab = ttk.Frame(
            self.notebook, style="App.TFrame", padding=(0, 6)
        )
        self._export_tab = ttk.Frame(
            self.notebook, style="App.TFrame", padding=(0, 6)
        )
        self._build_tab.columnconfigure(0, weight=1)
        self._export_tab.columnconfigure(0, weight=1)
        self.notebook.add(self._build_tab, text="Build & Verify", sticky="nsew")
        self.notebook.add(self._export_tab, text="Export", sticky="nsew")

        self._build_build_tab(self._build_tab)
        self._build_export_tab(self._export_tab)
        self._build_activity(content)

        self.source.trace_add("write", self._on_source_change)
        for variable in (
            self.output,
            self.classes,
            self.train,
            self.val,
            self.test,
            self.seed,
            self.extensions,
            self.workers,
            self.near_duplicate_threshold,
            self.min_width,
            self.min_height,
            self.max_aspect_ratio,
            self.compress_max_size,
            self.compress_jpeg_quality,
            self.train_model,
            self.skip_classes,
        ):
            variable.trace_add("write", self._on_form_change)
        self.empty_labels.trace_add("write", self._on_form_change)
        self.compress_enabled.trace_add("write", self._on_form_change)
        self._enable_drop_targets()
        self._validate_form()
        self._update_shortcuts()
        self._update_estimate()
        self._append(
            "Ready. Choose your source and a new output folder to begin.", "info"
        )

    # ------------------------------------------------------------------ tabs
    def _build_build_tab(self, parent: ttk.Frame) -> None:
        setup = self._card(
            parent,
            "DATASET SETUP",
            "Choose where to read images and where to create the new dataset.",
        )
        setup.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        setup.columnconfigure(1, weight=1)
        self._path_row(setup, 0, "Source folder", self.source, True, "source")
        self._path_row(setup, 1, "New output folder", self.output, False, "output")
        browse_existing = ttk.Button(
            setup,
            text="Browse existing",
            command=self.browse_existing_output,
            style="Secondary.TButton",
        )
        browse_existing.grid(row=4, column=3, padx=(9, 0), pady=8)
        self.buttons.append(browse_existing)
        self.drop_hint = ttk.Label(
            setup,
            text="Tip: drop a source folder, then choose a parent and new name for output. Use Browse existing to verify or format a built dataset.",
            style="Muted.TLabel",
        )
        self.drop_hint.grid(row=6, column=1, sticky="w", pady=(0, 3))
        self.estimate = StringVar(value="Estimate: choose a source folder.")
        ttk.Label(setup, textvariable=self.estimate, style="Estimate.TLabel").grid(
            row=7, column=1, sticky="w", pady=(4, 3)
        )

        options = self._card(
            parent,
            "BUILD OPTIONS",
            "Fine-tune labels, deterministic splits, and image handling.",
        )
        options.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        options.columnconfigure(1, weight=1)
        self._entry(options, 2, "Classes", self.classes, "classes")
        splits = ttk.Frame(options, style="Card.TFrame")
        ttk.Label(splits, text="SPLITS", style="FieldLabel.TLabel").pack(
            side="left", padx=(0, 16)
        )
        for label, variable, name in (
            ("Train", self.train, "train"),
            ("Validation", self.val, "val"),
            ("Test", self.test, "test"),
        ):
            group = ttk.Frame(splits, style="Card.TFrame")
            group.pack(side="left", padx=(0, 14))
            ttk.Label(group, text=label, style="Muted.TLabel").pack(anchor="w")
            entry = ttk.Entry(
                group, textvariable=variable, width=7, style="Input.TEntry"
            )
            entry.pack(pady=(3, 0))
            self.fields[name] = entry
        ttk.Label(options, text="Splits", style="FieldLabel.TLabel").grid(
            row=4, column=0, sticky="w", pady=8
        )
        splits.grid(row=4, column=1, sticky="w", pady=8)
        self.field_errors["splits"] = ttk.Label(options, text="", style="Error.TLabel")
        self.field_errors["splits"].grid(row=5, column=1, sticky="w")
        self._entry(options, 6, "Random seed", self.seed, "seed")
        seed_button = ttk.Button(
            options,
            text="Randomize",
            command=self.randomize_seed,
            style="Quiet.TButton",
        )
        seed_button.grid(row=6, column=2, padx=(9, 0), pady=8)
        self.buttons.append(seed_button)
        ttk.Checkbutton(
            options,
            text="Create empty labels when no sibling .txt label exists",
            variable=self.empty_labels,
            style="Purple.TCheckbutton",
        ).grid(row=8, column=1, sticky="w", pady=(8, 4))

        advanced = self._card(
            parent,
            "ADVANCED",
            "File types, parallelism, deduplication, and quality gates.",
        )
        advanced.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        advanced.columnconfigure(1, weight=1)
        self._entry(advanced, 0, "Extensions", self.extensions, "extensions")
        self._entry(advanced, 2, "Workers", self.workers, "workers")
        self._entry(
            advanced,
            4,
            "Near-duplicate threshold",
            self.near_duplicate_threshold,
            "near_duplicate_threshold",
        )
        quality = ttk.Frame(advanced, style="Card.TFrame")
        ttk.Label(quality, text="QUALITY", style="FieldLabel.TLabel").pack(
            side="left", padx=(0, 16)
        )
        for label, variable, name in (
            ("Min width", self.min_width, "min_width"),
            ("Min height", self.min_height, "min_height"),
            ("Max aspect", self.max_aspect_ratio, "max_aspect_ratio"),
        ):
            group = ttk.Frame(quality, style="Card.TFrame")
            group.pack(side="left", padx=(0, 14))
            ttk.Label(group, text=label, style="Muted.TLabel").pack(anchor="w")
            entry = ttk.Entry(
                group, textvariable=variable, width=8, style="Input.TEntry"
            )
            entry.pack(pady=(3, 0))
            self.fields[name] = entry
        ttk.Label(advanced, text="Quality", style="FieldLabel.TLabel").grid(
            row=6, column=0, sticky="w", pady=8
        )
        quality.grid(row=6, column=1, sticky="w", pady=8)
        self.field_errors["quality"] = ttk.Label(
            advanced, text="", style="Error.TLabel"
        )
        self.field_errors["quality"].grid(row=7, column=1, sticky="w")

        actions = ttk.Frame(parent, style="App.TFrame")
        actions.grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._button(
            actions,
            "Build dataset",
            lambda: self._start(build),
            "Primary.TButton",
            action=True,
        )
        self._button(
            actions,
            "Verify",
            lambda: self._start(verify),
            "Secondary.TButton",
            action=True,
        )
        self._button(
            actions,
            "Statistics",
            lambda: self._start(stats),
            "Secondary.TButton",
            action=True,
        )
        self._button(actions, "Load config", self.load_config, "Secondary.TButton")
        self._button(actions, "Save config", self.save_config, "Secondary.TButton")

    def _build_export_tab(self, parent: ttk.Frame) -> None:
        source_card = self._card(
            parent,
            "DATASET TO EXPORT",
            "Pick the built dataset or flat image+label folder to bundle for training.",
        )
        source_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        source_card.columnconfigure(1, weight=1)
        self._path_row(
            source_card, 0, "Dataset folder", self.output, True, "output"
        )
        ttk.Label(
            source_card,
            text="This is the same output folder used on the Build & Verify tab.",
            style="Muted.TLabel",
        ).grid(row=4, column=1, sticky="w", pady=(0, 3))

        export = self._card(
            parent,
            "EXPORT OPTIONS",
            "Training bundle (.zip) settings. Labels are read from sibling .txt files.",
        )
        export.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        export.columnconfigure(1, weight=1)
        self._entry(export, 0, "Base model", self.train_model, "train_model")
        ttk.Checkbutton(
            export,
            text="Compress images (downscale + JPEG re-encode)",
            variable=self.compress_enabled,
            style="Purple.TCheckbutton",
        ).grid(row=2, column=1, sticky="w", pady=(8, 4))
        compress = ttk.Frame(export, style="Card.TFrame")
        ttk.Label(compress, text="COMPRESS", style="FieldLabel.TLabel").pack(
            side="left", padx=(0, 16)
        )
        for label, variable, name in (
            ("Max size (px)", self.compress_max_size, "compress_max_size"),
            ("JPEG quality", self.compress_jpeg_quality, "compress_jpeg_quality"),
        ):
            group = ttk.Frame(compress, style="Card.TFrame")
            group.pack(side="left", padx=(0, 14))
            ttk.Label(group, text=label, style="Muted.TLabel").pack(anchor="w")
            entry = ttk.Entry(
                group, textvariable=variable, width=10, style="Input.TEntry"
            )
            entry.pack(pady=(3, 0))
            self.fields[name] = entry
        ttk.Label(export, text="Compress", style="FieldLabel.TLabel").grid(
            row=4, column=0, sticky="w", pady=8
        )
        compress.grid(row=4, column=1, sticky="w", pady=8)
        self.field_errors["compress"] = ttk.Label(
            export, text="", style="Error.TLabel"
        )
        self.field_errors["compress"].grid(row=5, column=1, sticky="w")
        self._entry(export, 6, "Skip class IDs", self.skip_classes, "skip_classes")
        ttk.Label(
            export,
            text="Comma-separated class IDs to drop from labels (e.g. 0). "
            "Remaining IDs are NOT renumbered. Images whose labels become empty are dropped.",
            style="Muted.TLabel",
        ).grid(row=8, column=1, sticky="w", pady=(0, 3))

        actions = ttk.Frame(parent, style="App.TFrame")
        actions.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._button(
            actions,
            "Format to train",
            lambda: self._start(format_to_train),
            "Primary.TButton",
            action=True,
        )
        ttk.Label(
            parent,
            text=(
                "Format to train creates <folder>_train.zip next to the dataset, "
                "with images, labels, dataset.yaml, and a ready-to-run train.py."
            ),
            style="Muted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))

    def _build_activity(self, parent: ttk.Frame) -> None:
        activity = self._card(
            parent,
            "ACTIVITY",
            "Progress, useful shortcuts, and the latest dataset summary.",
        )
        activity.grid(row=2, column=0, sticky="nsew")
        activity.columnconfigure(0, weight=1)
        activity.rowconfigure(6, weight=1)
        shortcuts = ttk.Frame(activity, style="Card.TFrame")
        shortcuts.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(shortcuts, textvariable=self.status, style="Status.TLabel").pack(
            side="left", padx=(0, 16)
        )
        self.open_output_button = ttk.Button(
            shortcuts,
            text="Open output folder",
            command=self.open_output,
            style="Quiet.TButton",
        )
        self.open_output_button.pack(side="right", padx=(8, 0))
        self.open_report_button = ttk.Button(
            shortcuts,
            text="Open report",
            command=self.open_report,
            style="Quiet.TButton",
        )
        self.open_report_button.pack(side="right", padx=(8, 0))
        ttk.Button(
            shortcuts,
            text="Reset preferences",
            command=self.reset_preferences,
            style="Quiet.TButton",
        ).pack(side="right")
        self.progress = ttk.Progressbar(
            activity, mode="indeterminate", style="Purple.Horizontal.TProgressbar"
        )
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        summary = ttk.Frame(activity, style="Card.TFrame")
        summary.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        for column, (key, label) in enumerate(
            (
                ("total_images", "IMAGES"),
                ("unique_images", "UNIQUE"),
                ("duplicates_removed", "DUPLICATES"),
                ("issues", "ISSUES"),
            )
        ):
            summary.columnconfigure(column, weight=1)
            metric = ttk.Label(
                summary, text="—", style="Metric.TLabel", anchor="center"
            )
            metric.grid(row=0, column=column, sticky="ew")
            ttk.Label(
                summary, text=label, style="MetricCaption.TLabel", anchor="center"
            ).grid(row=1, column=column, sticky="ew")
            self.metrics[key] = metric
        self.log = ScrolledText(
            activity,
            height=8,
            state="disabled",
            wrap="word",
            relief="flat",
            borderwidth=0,
            background=self.LOG_BG,
            foreground=self.TEXT,
            insertbackground=self.TEXT,
            selectbackground=self.ACCENT_DARK,
            font=("Cascadia Mono", 10),
            padx=13,
            pady=11,
        )
        self.log.grid(row=6, column=0, sticky="nsew")
        log_actions = ttk.Frame(activity, style="Card.TFrame")
        log_actions.grid(row=7, column=0, sticky="e", pady=(7, 0))
        ttk.Button(
            log_actions, text="Clear log", command=self.clear_log, style="Quiet.TButton"
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            log_actions, text="Copy log", command=self.copy_log, style="Quiet.TButton"
        ).pack(side="left")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.SURFACE)
        style.configure(
            "CardTitle.TLabel",
            background=self.SURFACE,
            foreground=self.ACCENT_HOVER,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "CardHint.TLabel",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 11),
        )
        style.configure(
            "Badge.TLabel",
            background=self.ACCENT_DARK,
            foreground=self.TEXT,
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=self.SURFACE,
            foreground=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Error.TLabel",
            background=self.SURFACE,
            foreground=self.DANGER,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background=self.SURFACE,
            foreground=self.SUCCESS,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Estimate.TLabel",
            background=self.SURFACE,
            foreground=self.ACCENT_HOVER,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Metric.TLabel",
            background=self.SURFACE_ALT,
            foreground=self.TEXT,
            font=("Segoe UI", 16, "bold"),
            padding=(8, 4),
        )
        style.configure(
            "MetricCaption.TLabel",
            background=self.SURFACE_ALT,
            foreground=self.MUTED,
            font=("Segoe UI", 8, "bold"),
            padding=(8, 5),
        )
        style.configure(
            "Input.TEntry",
            fieldbackground=self.SURFACE_ALT,
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=(10, 7),
        )
        style.configure(
            "Invalid.TEntry",
            fieldbackground="#3c1b35",
            foreground=self.TEXT,
            bordercolor=self.DANGER,
            lightcolor=self.DANGER,
            darkcolor=self.DANGER,
            padding=(10, 7),
        )
        style.map(
            "Input.TEntry",
            bordercolor=[("focus", self.ACCENT)],
            lightcolor=[("focus", self.ACCENT)],
            darkcolor=[("focus", self.ACCENT)],
        )
        style.configure(
            "Primary.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
            padding=(15, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.ACCENT_HOVER), ("disabled", self.ACCENT_DARK)],
            foreground=[("disabled", "#d7c8ed")],
        )
        style.configure(
            "Secondary.TButton",
            background=self.SURFACE_ALT,
            foreground=self.TEXT,
            borderwidth=0,
            font=("Segoe UI", 10),
            padding=(12, 9),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", self.BORDER), ("disabled", self.SURFACE_ALT)],
            foreground=[("disabled", self.MUTED)],
        )
        style.configure(
            "Quiet.TButton",
            background=self.SURFACE,
            foreground=self.MUTED,
            borderwidth=0,
            font=("Segoe UI", 9),
            padding=(6, 3),
        )
        style.map(
            "Quiet.TButton",
            foreground=[("active", self.TEXT), ("disabled", self.BORDER)],
        )
        style.configure(
            "Purple.TCheckbutton",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.map(
            "Purple.TCheckbutton",
            background=[("active", self.SURFACE)],
            foreground=[("active", self.TEXT)],
        )
        style.configure(
            "Purple.Horizontal.TProgressbar",
            troughcolor=self.SURFACE_ALT,
            background=self.ACCENT,
            bordercolor=self.SURFACE_ALT,
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
        )
        # Notebook (tabs) styling to match the dark theme.
        style.configure(
            "App.TNotebook", background=self.BG, borderwidth=0, tabmargins=(0, 4, 0, 0)
        )
        style.configure(
            "App.TNotebook.Tab",
            background=self.SURFACE_ALT,
            foreground=self.MUTED,
            borderwidth=0,
            padding=(18, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", self.SURFACE)],
            foreground=[("selected", self.TEXT)],
        )

    def _make_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="YOLO Dataset Builder", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Create clean, reproducible training datasets from local image folders.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(header, text="LOCAL TOOL", style="Badge.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="e"
        )

    def _card(self, parent: ttk.Frame, title: str, hint: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(20, 16))
        ttk.Label(card, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(card, text=hint, style="CardHint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(3, 12)
        )
        return card

    def _button(
        self,
        parent: ttk.Frame,
        text: str,
        command: Callable[[], None],
        style: str,
        action: bool = False,
    ) -> None:
        button = ttk.Button(parent, text=text, command=command, style=style)
        button.pack(side="left", padx=(0, 8))
        self.buttons.append(button)
        if action:
            self.action_buttons.append(button)
            if text == "Build dataset":
                self.build_button = button

    def _entry(
        self, parent: ttk.Frame, row: int, label: str, variable: StringVar, name: str
    ) -> None:
        grid_row = row
        ttk.Label(parent, text=label, style="FieldLabel.TLabel").grid(
            row=grid_row, column=0, sticky="w", pady=8
        )
        entry = ttk.Entry(parent, textvariable=variable, style="Input.TEntry")
        entry.grid(row=grid_row, column=1, sticky="ew", pady=8)
        self.fields[name] = entry
        error = ttk.Label(parent, text="", style="Error.TLabel")
        error.grid(row=grid_row + 1, column=1, sticky="w")
        self.field_errors[name] = error

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: StringVar,
        must_exist: bool,
        name: str,
    ) -> None:
        grid_row = 2 + row * 2
        self._entry(parent, grid_row, label, variable, name)

        def choose() -> None:
            if must_exist:
                result = filedialog.askdirectory(
                    mustexist=True, title="Choose source folder"
                )
                if result:
                    variable.set(result)
                return
            parent = filedialog.askdirectory(
                mustexist=True, title="Choose where to create the dataset"
            )
            if not parent:
                return
            candidate = self._new_output_path(parent)
            if candidate:
                variable.set(str(candidate))

        button_text = "Browse" if must_exist else "Choose location"
        button = ttk.Button(
            parent, text=button_text, command=choose, style="Secondary.TButton"
        )
        button.grid(row=grid_row, column=2, padx=(9, 0), pady=8)
        self.buttons.append(button)

    def _enable_drop_targets(self) -> None:
        if not hasattr(self.root, "drop_target_register"):
            self.drop_hint.configure(
                text="Browse works everywhere. Install tkinterdnd2 to enable folder drop."
            )
            return
        try:
            from tkinterdnd2 import DND_FILES  # type: ignore[import-not-found]

            for name, entry in (
                ("source", self.fields["source"]),
                ("output", self.fields["output"]),
            ):
                entry.drop_target_register(DND_FILES)
                entry.dnd_bind(
                    "<<Drop>>",
                    lambda event, field=name: self._receive_drop(event, field),
                )
            self.drop_hint.configure(
                text="Tip: drop a source folder, then choose a parent and new name for output."
            )
        except (ImportError, OSError, RuntimeError):
            self.drop_hint.configure(
                text="Browse works everywhere. Folder drop is unavailable in this session."
            )

    def _receive_drop(self, event: Any, field: str) -> str:
        paths = self.root.tk.splitlist(event.data)
        if paths:
            path = Path(paths[0])
            if path.is_dir():
                if field == "source":
                    self.source.set(str(path))
                else:
                    candidate = self._new_output_path(path)
                    if candidate:
                        self.output.set(str(candidate))
            else:
                self._append("Drop a folder, not an individual file.", "warning")
        return "break"

    def _new_output_path(self, parent: str | Path) -> Path | None:
        name = simpledialog.askstring(
            "New output folder", "Dataset folder name:", parent=self.root
        )
        if not name:
            return None
        candidate = Path(parent) / name.strip()
        if not name.strip() or candidate.name != name.strip():
            messagebox.showerror(
                "Invalid folder name",
                "Enter a single new folder name, without path separators.",
            )
            return None
        return candidate

    def browse_existing_output(self) -> None:
        """Pick an existing dataset folder for Verify, Statistics, or Format to train."""
        result = filedialog.askdirectory(
            mustexist=True, title="Choose an existing dataset output folder"
        )
        if result:
            self.output.set(result)

    def _on_form_change(self, *_: object) -> None:
        self._validate_form()
        self._save_preferences()
        self._update_shortcuts()

    def _on_source_change(self, *_: object) -> None:
        self._on_form_change()
        self._update_estimate()

    def _update_estimate(self) -> None:
        self._estimate_token += 1
        token = self._estimate_token
        source = Path(self.source.get().strip()) if self.source.get().strip() else None
        if not source or not source.is_dir():
            self.estimate.set("Estimate: choose a source folder.")
            return
        self.estimate.set("Estimate: counting source images…")
        extensions = {
            extension.strip().lower()
            for extension in self.extensions.get().split(",")
            if extension.strip()
        }

        def count_images() -> None:
            files = total_bytes = 0
            try:
                for path in source.rglob("*"):
                    if path.is_file() and path.suffix.lower() in extensions:
                        files += 1
                        total_bytes += path.stat().st_size
            except OSError:
                pass
            seconds = max(5, files / 35, total_bytes / (75 * 1024 * 1024))
            self.root.after(
                0, lambda: self._show_estimate(token, files, total_bytes, seconds)
            )

        threading.Thread(target=count_images, daemon=True).start()

    def _show_estimate(
        self, token: int, files: int, total_bytes: int, seconds: float
    ) -> None:
        if token != self._estimate_token:
            return
        if files == 0:
            self.estimate.set("Estimate: no supported images found yet.")
            return
        duration = (
            f"{round(seconds)} sec"
            if seconds < 60
            else f"{seconds // 60:.0f} min {round(seconds % 60)} sec"
        )
        size = (
            f"{total_bytes / (1024 * 1024):.0f} MB"
            if total_bytes < 1024**3
            else f"{total_bytes / (1024**3):.1f} GB"
        )
        self.estimate.set(f"Estimate: about {duration} • {files:,} images • {size}")

    def _set_error(self, name: str, message: str) -> None:
        if name == "splits":
            for key in ("train", "val", "test"):
                self.fields[key].configure(
                    style="Invalid.TEntry" if message else "Input.TEntry"
                )
        elif name == "quality":
            for key in ("min_width", "min_height", "max_aspect_ratio"):
                self.fields[key].configure(
                    style="Invalid.TEntry" if message else "Input.TEntry"
                )
        elif name == "compress":
            for key in ("compress_max_size", "compress_jpeg_quality"):
                self.fields[key].configure(
                    style="Invalid.TEntry" if message else "Input.TEntry"
                )
        elif name in self.fields:
            self.fields[name].configure(
                style="Invalid.TEntry" if message else "Input.TEntry"
            )
        if name in self.field_errors:
            self.field_errors[name].configure(text=message)

    def _validate_form(self) -> bool:
        source = Path(self.source.get().strip()) if self.source.get().strip() else None
        output = Path(self.output.get().strip()) if self.output.get().strip() else None
        self._set_error(
            "source",
            "" if source and source.is_dir() else "Choose an existing source folder.",
        )
        output_error = ""
        if not output:
            output_error = "Choose a new output folder."
        elif output.exists():
            output_error = "Output already exists; choose a new folder for Build."
        self._set_error("output", output_error)
        self._set_error(
            "classes",
            ""
            if [item.strip() for item in self.classes.get().split(",") if item.strip()]
            else "Enter at least one class name.",
        )
        try:
            values = [float(value.get()) for value in (self.train, self.val, self.test)]
            split_error = (
                ""
                if all(value >= 0 for value in values) and abs(sum(values) - 1.0) < 1e-6
                else "Splits must be non-negative and total 1.0."
            )
        except ValueError:
            split_error = "Splits must be numbers totaling 1.0."
        self._set_error("splits", split_error)
        try:
            int(self.seed.get())
            seed_error = ""
        except ValueError:
            seed_error = "Seed must be a whole number."
        self._set_error("seed", seed_error)
        # Extensions
        exts = [
            e.strip()
            for e in self.extensions.get().split(",")
            if e.strip()
        ]
        ext_error = (
            ""
            if exts and all(e.startswith(".") for e in exts)
            else "Extensions must be a comma-separated list starting with '.'"
        )
        self._set_error("extensions", ext_error)
        # Workers
        try:
            workers = int(self.workers.get())
            workers_error = "" if workers >= 0 else "Workers must be 0 or a positive integer."
        except ValueError:
            workers_error = "Workers must be a whole number."
            workers = -1
        self._set_error("workers", workers_error)
        # Near-duplicate threshold
        try:
            ndt = int(self.near_duplicate_threshold.get())
            ndt_error = (
                "" if 0 <= ndt <= 7 else "Near-duplicate threshold must be 0 to 7."
            )
        except ValueError:
            ndt_error = "Near-duplicate threshold must be a whole number."
            ndt = -1
        self._set_error("near_duplicate_threshold", ndt_error)
        # Quality
        quality_error = ""
        try:
            mw = float(self.min_width.get())
            mh = float(self.min_height.get())
            mar = float(self.max_aspect_ratio.get())
            if not (mw > 0 and mh > 0 and mar > 0):
                quality_error = "Quality limits must be positive numbers."
        except ValueError:
            quality_error = "Quality limits must be numbers."
        self._set_error("quality", quality_error)
        # Compress
        compress_error = ""
        try:
            ms = int(self.compress_max_size.get())
            jq = int(self.compress_jpeg_quality.get())
            if ms < 1:
                compress_error = "Max size must be a positive integer."
            elif not 1 <= jq <= 100:
                compress_error = "JPEG quality must be 1 to 100."
        except ValueError:
            compress_error = "Compress values must be whole numbers."
        self._set_error("compress", compress_error)
        # Train model
        model_error = "" if self.train_model.get().strip() else "Enter a base model."
        self._set_error("train_model", model_error)
        # Skip class IDs (optional, comma-separated non-negative integers)
        skip_error = ""
        skip_raw = [
            token.strip()
            for token in self.skip_classes.get().split(",")
            if token.strip()
        ]
        try:
            skip_ids = [int(token) for token in skip_raw]
            if any(cid < 0 for cid in skip_ids):
                skip_error = "Skip class IDs must be non-negative integers."
        except ValueError:
            skip_error = "Skip class IDs must be whole numbers (e.g. 0, 2)."
        self._set_error("skip_classes", skip_error)

        valid = not any(
            (
                not source or not source.is_dir(),
                bool(output_error),
                bool(split_error),
                bool(seed_error),
                bool(ext_error),
                bool(workers_error),
                bool(ndt_error),
                bool(quality_error),
                bool(compress_error),
                bool(model_error),
                bool(skip_error),
                not [
                    item.strip()
                    for item in self.classes.get().split(",")
                    if item.strip()
                ],
            )
        )
        if hasattr(self, "build_button"):
            self.build_button.configure(state="normal" if valid else "disabled")
        return valid

    def config(self) -> dict[str, Any]:
        config = copy.deepcopy(self.config_base)
        config.update(
            {
                "source_dir": self.source.get().strip(),
                "output_dir": self.output.get().strip(),
                "classes": [
                    name.strip()
                    for name in self.classes.get().split(",")
                    if name.strip()
                ],
                "splits": {
                    "train": float(self.train.get()),
                    "val": float(self.val.get()),
                    "test": float(self.test.get()),
                },
                "seed": int(self.seed.get()),
                "create_empty_labels": self.empty_labels.get(),
                "extensions": [
                    e.strip().lower()
                    for e in self.extensions.get().split(",")
                    if e.strip()
                ],
                "workers": int(self.workers.get()),
                "near_duplicate_threshold": int(self.near_duplicate_threshold.get()),
                "quality": {
                    "min_width": float(self.min_width.get()),
                    "min_height": float(self.min_height.get()),
                    "max_aspect_ratio": float(self.max_aspect_ratio.get()),
                },
                "compress": {
                    "enabled": bool(self.compress_enabled.get()),
                    "max_size": int(self.compress_max_size.get()),
                    "jpeg_quality": int(self.compress_jpeg_quality.get()),
                },
                "train": {"model": self.train_model.get().strip()},
                "skip_classes": [
                    int(token.strip())
                    for token in self.skip_classes.get().split(",")
                    if token.strip()
                ],
            }
        )
        validate_config(config)
        return config

    def _preflight(
        self, command: Callable[[dict[str, Any]], int]
    ) -> dict[str, Any] | None:
        try:
            config = self.config()
            source, output = Path(config["source_dir"]), Path(config["output_dir"])
            if command is build:
                if not source.is_dir():
                    raise ValueError("Choose an existing source folder.")
                if output.exists():
                    raise ValueError(
                        "Output already exists. Choose a new output folder for Build."
                    )
                source, output = source.resolve(), output.resolve()
                if (
                    output == source
                    or source in output.parents
                    or output in source.parents
                ):
                    raise ValueError(
                        "Source and output folders must be separate; neither may contain the other."
                    )
            elif not output.is_dir():
                if command is format_to_train:
                    raise ValueError(
                        "Format to train needs an existing folder: either a built "
                        "dataset or a flat folder of paired images and .txt labels."
                    )
                raise ValueError(
                    "Verify and Statistics need an existing output folder."
                )
            return config
        except (ValueError, TypeError) as exc:
            self._append(str(exc), "error")
            messagebox.showerror("Invalid settings", str(exc))
            return None

    def _confirm_build(self, config: dict[str, Any]) -> bool:
        dialog = Toplevel(self.root)
        dialog.title("Confirm build")
        dialog.configure(background=self.SURFACE)
        dialog.transient(self.root)
        dialog.grab_set()
        body = ttk.Frame(dialog, style="Card.TFrame", padding=22)
        body.grid(sticky="nsew")
        ttk.Label(body, text="Ready to build?", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            body,
            text="Review these settings before creating the new dataset.",
            style="CardHint.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 16))
        details = (
            ("Source", config["source_dir"]),
            ("Output", config["output_dir"]),
            ("Classes", ", ".join(config["classes"])),
            (
                "Splits",
                " / ".join(
                    f"{key}: {value:.0%}" for key, value in config["splits"].items()
                ),
            ),
            ("Seed", str(config["seed"])),
            (
                "Empty labels",
                "Create when missing"
                if config["create_empty_labels"]
                else "Do not create",
            ),
        )
        for row, (label, value) in enumerate(details, start=2):
            ttk.Label(body, text=label.upper(), style="CardTitle.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 16), pady=4
            )
            ttk.Label(body, text=value, style="CardHint.TLabel", wraplength=440).grid(
                row=row, column=1, sticky="w", pady=4
            )
        result = BooleanVar(value=False)
        actions = ttk.Frame(body, style="Card.TFrame")
        actions.grid(
            row=len(details) + 2, column=0, columnspan=2, sticky="e", pady=(18, 0)
        )
        ttk.Button(
            actions, text="Cancel", command=dialog.destroy, style="Secondary.TButton"
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            actions,
            text="Build dataset",
            command=lambda: (result.set(True), dialog.destroy()),
            style="Primary.TButton",
        ).pack(side="left")
        dialog.wait_window()
        return result.get()

    def _start(self, command: Callable[[dict[str, Any]], int]) -> None:
        config = self._preflight(command)
        if config is None or (command is build and not self._confirm_build(config)):
            return
        for button in self.buttons:
            button.configure(state="disabled")
        verb = (
            "Format to train"
            if command is format_to_train
            else getattr(command, "__name__", "Operation").title()
        )
        self.status.set(f"{verb} in progress…")
        self._append(f"Starting {verb.lower()}…", "info")
        self.progress.start(12)

        def worker() -> None:
            try:
                result = command(config)
                self.root.after(0, lambda: self._finished(command, result))
            except (
                OSError,
                ValueError,
                RuntimeError,
                FileNotFoundError,
                FileExistsError,
            ) as exc:
                message = str(exc)
                self.root.after(0, lambda: self._failed(message))

        threading.Thread(target=worker, daemon=True).start()

    def _finished(self, command: Callable[[dict[str, Any]], int], result: int) -> None:
        self.progress.stop()
        verb = (
            "Format to train"
            if command is format_to_train
            else getattr(command, "__name__", "Operation").title()
        )
        self.status.set(
            f"{verb} complete" if result == 0 else f"{verb} completed with issues"
        )
        self._append(
            f"{verb} finished (exit code {result}).",
            "success" if result == 0 else "warning",
        )
        if command is build and result == 0:
            self._verify_output_files()
        if command is format_to_train:
            self.notebook.select(self._export_tab)
        self._load_summary()
        for button in self.buttons:
            button.configure(state="normal")
        self._validate_form()
        self._update_shortcuts()

    def _verify_output_files(self) -> None:
        output = Path(self.output.get())
        missing = [
            artifact
            for artifact in self.REQUIRED_OUTPUT_ARTIFACTS
            if not (output / artifact).exists()
        ]
        if missing:
            self.status.set("Build completed, but output is incomplete")
            self._append(
                f"Missing expected output files: {', '.join(missing)}", "error"
            )
        else:
            self._append(
                "Output check passed: dataset, labels, configuration, and reports are all present.",
                "success",
            )

    def _failed(self, message: str) -> None:
        self.progress.stop()
        self.status.set("Operation failed")
        self._append(message, "error")
        for button in self.buttons:
            button.configure(state="normal")
        self._validate_form()
        self._update_shortcuts()
        messagebox.showerror("Dataset Builder", message)

    def _load_summary(self) -> None:
        report = Path(self.output.get()) / "reports" / "stats.json"
        if not report.exists():
            return
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            issues = (
                int(data.get("corrupt_images", 0))
                + int(data.get("images_missing_labels", 0))
                + int(data.get("invalid_labels", 0))
                + int(data.get("labels_missing_images", 0))
                + int(data.get("duplicate_filenames", 0))
            )
            self.metrics["total_images"].configure(
                text=str(data.get("total_images", "—"))
            )
            self.metrics["unique_images"].configure(
                text=str(data.get("unique_images", data.get("valid_images", "—")))
            )
            self.metrics["duplicates_removed"].configure(
                text=str(data.get("duplicates_removed", "0"))
            )
            self.metrics["issues"].configure(text=str(issues))
        except (OSError, ValueError, TypeError) as exc:
            self._append(f"Could not read build summary: {exc}", "warning")

    def _append(self, text: str, tag: str = "info") -> None:
        colors = {
            "info": self.MUTED,
            "success": self.SUCCESS,
            "warning": self.WARNING,
            "error": self.DANGER,
        }
        self.log.configure(state="normal")
        self.log.tag_configure(tag, foreground=colors[tag])
        timestamp = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        self.log.insert("end", f"[{timestamp}] {text}\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._append("Activity log cleared.")

    def copy_log(self) -> None:
        text = self.log.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._append("Activity log copied to clipboard.", "success")

    def _update_shortcuts(self) -> None:
        output = Path(self.output.get().strip()) if self.output.get().strip() else None
        self.open_output_button.configure(
            state="normal" if output and output.is_dir() else "disabled"
        )
        report = output / "reports" / "report.html" if output else None
        self.open_report_button.configure(
            state="normal" if report and report.exists() else "disabled"
        )

    def load_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            config = load_config(path)
            self.config_base = copy.deepcopy(config)
            self.source.set(config["source_dir"])
            self.output.set(config["output_dir"])
            self.classes.set(", ".join(config["classes"]))
            self.train.set(str(config["splits"]["train"]))
            self.val.set(str(config["splits"]["val"]))
            self.test.set(str(config["splits"]["test"]))
            self.seed.set(str(config["seed"]))
            self.empty_labels.set(bool(config["create_empty_labels"]))
            self.extensions.set(", ".join(config["extensions"]))
            self.workers.set(str(config["workers"]))
            self.near_duplicate_threshold.set(
                str(config["near_duplicate_threshold"])
            )
            self.min_width.set(str(config["quality"]["min_width"]))
            self.min_height.set(str(config["quality"]["min_height"]))
            self.max_aspect_ratio.set(str(config["quality"]["max_aspect_ratio"]))
            self.compress_enabled.set(bool(config["compress"]["enabled"]))
            self.compress_max_size.set(str(config["compress"]["max_size"]))
            self.compress_jpeg_quality.set(str(config["compress"]["jpeg_quality"]))
            self.train_model.set(str(config["train"]["model"]))
            self.skip_classes.set(
                ", ".join(str(cid) for cid in config.get("skip_classes", []))
            )
            self._append(f"Loaded configuration: {path}", "success")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._append(f"Could not load configuration: {exc}", "error")
            messagebox.showerror("Could not load configuration", str(exc))

    def save_config(self) -> None:
        try:
            config = self.config()
        except (ValueError, TypeError) as exc:
            self._append(str(exc), "error")
            messagebox.showerror("Invalid settings", str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml", filetypes=[("YAML", "*.yaml *.yml")]
        )
        if path:
            Path(path).write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            self._append(f"Saved configuration: {path}", "success")

    def randomize_seed(self) -> None:
        self.seed.set(str(secrets.randbelow(2_147_483_647) + 1))
        self._append("Generated a new random seed.", "success")

    def open_output(self) -> None:
        output = Path(self.output.get())
        if output.is_dir():
            webbrowser.open(output.resolve().as_uri())

    def open_report(self) -> None:
        report = Path(self.output.get()) / "reports" / "report.html"
        if report.exists():
            webbrowser.open(report.resolve().as_uri())

    def _restore_preferences(self) -> None:
        preferences_file = self._preferences_file()
        if not preferences_file.exists() and self.LEGACY_PREFERENCES_FILE.exists():
            try:
                self.LEGACY_PREFERENCES_FILE.replace(preferences_file)
            except OSError:
                pass
        try:
            saved = json.loads(preferences_file.read_text(encoding="utf-8"))
            self.root.geometry(saved.get("geometry", self.root.geometry()))
            self.source.set(saved.get("source", ""))
            self.output.set(saved.get("output", ""))
            self.classes.set(saved.get("classes", self.classes.get()))
            self.train.set(saved.get("train", self.train.get()))
            self.val.set(saved.get("val", self.val.get()))
            self.test.set(saved.get("test", self.test.get()))
            self.seed.set(saved.get("seed", self.seed.get()))
            self.empty_labels.set(saved.get("empty_labels", self.empty_labels.get()))
            self.extensions.set(saved.get("extensions", self.extensions.get()))
            self.workers.set(saved.get("workers", self.workers.get()))
            self.near_duplicate_threshold.set(
                saved.get(
                    "near_duplicate_threshold", self.near_duplicate_threshold.get()
                )
            )
            self.min_width.set(saved.get("min_width", self.min_width.get()))
            self.min_height.set(saved.get("min_height", self.min_height.get()))
            self.max_aspect_ratio.set(
                saved.get("max_aspect_ratio", self.max_aspect_ratio.get())
            )
            self.compress_enabled.set(
                saved.get("compress_enabled", self.compress_enabled.get())
            )
            self.compress_max_size.set(
                saved.get("compress_max_size", self.compress_max_size.get())
            )
            self.compress_jpeg_quality.set(
                saved.get("compress_jpeg_quality", self.compress_jpeg_quality.get())
            )
            self.train_model.set(saved.get("train_model", self.train_model.get()))
            self.skip_classes.set(saved.get("skip_classes", self.skip_classes.get()))
        except (OSError, ValueError, TypeError):
            pass

    def _save_preferences(self) -> None:
        preferences = {
            "geometry": self.root.geometry(),
            "source": self.source.get(),
            "output": self.output.get(),
            "classes": self.classes.get(),
            "train": self.train.get(),
            "val": self.val.get(),
            "test": self.test.get(),
            "seed": self.seed.get(),
            "empty_labels": self.empty_labels.get(),
            "extensions": self.extensions.get(),
            "workers": self.workers.get(),
            "near_duplicate_threshold": self.near_duplicate_threshold.get(),
            "min_width": self.min_width.get(),
            "min_height": self.min_height.get(),
            "max_aspect_ratio": self.max_aspect_ratio.get(),
            "compress_enabled": self.compress_enabled.get(),
            "compress_max_size": self.compress_max_size.get(),
            "compress_jpeg_quality": self.compress_jpeg_quality.get(),
            "train_model": self.train_model.get(),
            "skip_classes": self.skip_classes.get(),
        }
        try:
            self._preferences_file().write_text(
                json.dumps(preferences, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def reset_preferences(self) -> None:
        if not messagebox.askyesno(
            "Reset preferences",
            "Reset saved window and form preferences? Your dataset files will not be changed.",
        ):
            return
        try:
            self._preferences_file().unlink(missing_ok=True)
        except OSError as exc:
            self._append(f"Could not remove saved preferences: {exc}", "error")
            return
        self.config_base = copy.deepcopy(DEFAULT_CONFIG)
        self.source.set("")
        self.output.set("")
        self.classes.set("object")
        self.train.set("0.8")
        self.val.set("0.1")
        self.test.set("0.1")
        self.seed.set("42")
        self.empty_labels.set(True)
        self.extensions.set(", ".join(DEFAULT_CONFIG["extensions"]))
        self.workers.set(str(DEFAULT_CONFIG["workers"]))
        self.near_duplicate_threshold.set(
            str(DEFAULT_CONFIG["near_duplicate_threshold"])
        )
        self.min_width.set(str(DEFAULT_CONFIG["quality"]["min_width"]))
        self.min_height.set(str(DEFAULT_CONFIG["quality"]["min_height"]))
        self.max_aspect_ratio.set(str(DEFAULT_CONFIG["quality"]["max_aspect_ratio"]))
        self.compress_enabled.set(bool(DEFAULT_CONFIG["compress"]["enabled"]))
        self.compress_max_size.set(str(DEFAULT_CONFIG["compress"]["max_size"]))
        self.compress_jpeg_quality.set(str(DEFAULT_CONFIG["compress"]["jpeg_quality"]))
        self.train_model.set(str(DEFAULT_CONFIG["train"]["model"]))
        self.skip_classes.set("")
        self.status.set("Preferences reset")
        self._append("Saved preferences reset.", "success")


if __name__ == "__main__":
    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore[import-not-found]

        window = TkinterDnD.Tk()
    except ImportError:
        window = Tk()
    DatasetBuilderGui(window)
    window.mainloop()
