"""A dark local desktop interface for the dataset-builder commands."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from datetime import timedelta
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

from ..builder import build, format_to_train, stats, verify
from ..config import DEFAULT_CONFIG, load_config, validate_config
from . import fields, theme
from .fields import FIELDS, GROUPS
from .widgets import Card, LineStream

APP_NAME = "YoloDatasetBuilder"
LEGACY_PREFERENCES = Path(__file__).resolve().parents[2] / ".dataset_builder_gui.json"
LOG_LIMIT = 2000  # lines retained in the activity log
POLL_MS = 100
DOCK_HEIGHT = 140  # compact by default; the sash is still draggable
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960
DISPLAY_FRACTION = 0.96

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

# name, label, function, needs an existing output folder, shortcut
COMMANDS = (
    ("build", "Build dataset", build, False, "<Control-b>", "Ctrl+B"),
    ("verify", "Verify", verify, True, "<Control-r>", "Ctrl+R"),
    ("stats", "Statistics", stats, True, "<Control-t>", "Ctrl+T"),
    ("export", "Format to train", format_to_train, True, "<Control-e>", "Ctrl+E"),
)
COMMAND_LABELS = {name: label for name, label, *_ in COMMANDS}


def app_data_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home())
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_NAME


def preferences_file() -> Path:
    directory = app_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "preferences.json"


class DatasetBuilderGui:
    """A keyboard-friendly front end for local dataset operations."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("YOLO Dataset Builder")
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        width = max(1, min(WINDOW_WIDTH, round(screen_width * DISPLAY_FRACTION)))
        height = max(1, min(WINDOW_HEIGHT, round(screen_height * DISPLAY_FRACTION)))
        root.geometry(f"{width}x{height}")
        root.minsize(min(820, width), min(620, height))
        self._initial_size = (width, height)
        self._saved_geometry: str | None = None
        theme.apply(root)

        self.config_base = copy.deepcopy(DEFAULT_CONFIG)
        self.vars: dict[str, StringVar | BooleanVar] = {
            spec.name: (BooleanVar if spec.boolean else StringVar)(
                value=spec.initial()
            )
            for spec in FIELDS
        }
        # A field can appear on more than one page, so every widget map holds a
        # list; errors then highlight every copy rather than only the last one.
        self.entries: dict[str, list[ttk.Entry]] = {}
        self.messages: dict[str, list[ttk.Label]] = {}
        self.hints: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.values: dict[str, Any] = {}
        self.source_ok = False
        self.metrics: dict[str, ttk.Label] = {}
        self.buttons: list[ttk.Button] = []
        self.actions: dict[str, ttk.Button] = {}
        self.status = StringVar(value="Ready")
        self.estimate = StringVar(value="Choose a source folder to see an estimate.")
        self.page = StringVar(value="build")

        self._running = False
        self._started_at = 0.0
        self._estimate_token = 0
        self._progress_state: tuple[str, int, int] = ("", 0, 0)
        self._shown_progress: tuple[str, int, int] | None = None
        self._phase_started_at = 0.0
        self._phase_start_done = 0
        self._logged_phase = ""
        self._logged_progress = -1
        self._bar_mode = ""
        self._output: queue.Queue[str] = queue.Queue()
        # Written by background threads, consumed by _poll on the main thread.
        self._outcome: tuple[Any, ...] | None = None
        self._estimate_results: queue.Queue[tuple[int, int, int]] = queue.Queue()

        self._restore_preferences()
        self._fit_to_display()
        self._build_layout()
        self._bind_shortcuts()

        for name, variable in self.vars.items():
            variable.trace_add("write", lambda *_, key=name: self._on_change(key))
        self._enable_drop_targets()
        self._validate()
        self._update_estimate()
        self.root.after(POLL_MS, self._poll)
        self.log_line("Ready. Choose a source folder and a new dataset folder.", "info")

    # ------------------------------------------------------------ state
    def _on_change(self, name: str) -> None:
        self._validate()
        self._save_preferences()
        if name in ("source", "extensions"):
            self._update_estimate()

    def value(self, name: str) -> Any:
        return self.vars[name].get()

    def _restore_preferences(self) -> None:
        path = preferences_file()
        if not path.exists() and LEGACY_PREFERENCES.exists():
            with contextlib.suppress(OSError):
                LEGACY_PREFERENCES.replace(path)
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        geometry = saved.get("geometry")
        if isinstance(geometry, str) and geometry:
            with contextlib.suppress(Exception):
                self._saved_geometry = geometry
                self.root.geometry(geometry)
        for spec in FIELDS:
            if spec.name in saved:
                with contextlib.suppress(Exception):
                    self.vars[spec.name].set(saved[spec.name])

    def _save_preferences(self) -> None:
        data = {"geometry": self.root.geometry()}
        data.update({spec.name: self.value(spec.name) for spec in FIELDS})
        try:
            preferences_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _fit_to_display(self) -> None:
        """Clamp restored geometry to the current display and keep it visible."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        geometry = getattr(self, "_saved_geometry", None)
        match = re.match(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", geometry) if geometry else None
        initial = getattr(self, "_initial_size", None)
        saved_width = int(match.group(1)) if match else initial[0]
        saved_height = int(match.group(2)) if match else initial[1]
        width = min(
            saved_width, WINDOW_WIDTH, max(1, round(screen_width * DISPLAY_FRACTION))
        )
        height = min(
            saved_height, WINDOW_HEIGHT, max(1, round(screen_height * DISPLAY_FRACTION))
        )
        saved_x = int(match.group(3)) if match else self.root.winfo_x()
        saved_y = int(match.group(4)) if match else self.root.winfo_y()
        x = min(max(0, saved_x), screen_width - width)
        y = min(max(0, saved_y), screen_height - height)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def reset_preferences(self) -> None:
        if not messagebox.askyesno(
            "Reset preferences",
            "Reset the saved window size and every form value?\n\n"
            "Your dataset files are not touched.",
        ):
            return
        try:
            preferences_file().unlink(missing_ok=True)
        except OSError as exc:
            self.log_line(f"Could not remove saved preferences: {exc}", "error")
            return
        self.config_base = copy.deepcopy(DEFAULT_CONFIG)
        for spec in FIELDS:
            self.vars[spec.name].set(spec.initial())
        self.set_status("Preferences reset", "success")
        self.log_line("Preferences reset to defaults.", "success")

    # ----------------------------------------------------------- layout
    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self._header()
        self._nav()

        # A draggable split: the form and the activity log compete for height,
        # so let the user decide the balance rather than fixing it here.
        split = ttk.PanedWindow(self.root, orient="vertical", style="App.TPanedwindow")
        split.grid(row=2, column=0, sticky="nsew")

        container = ttk.Frame(split, style="App.TFrame", padding=(theme.PAD_L, 0))
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        self.pages: dict[str, ttk.Frame] = {}
        for name, builder in (("build", self._build_page), ("export", self._export_page)):
            page = ttk.Frame(container, style="App.TFrame")
            page.grid(row=0, column=0, sticky="nsew")
            page.columnconfigure(0, weight=1)
            page.columnconfigure(1, weight=1)
            self.pages[name] = page
            builder(page)
        split.add(container, weight=3)
        split.add(self._activity(split), weight=1)
        self._split = split
        self.root.after_idle(self._place_sash)
        self._show_page()

    def _place_sash(self) -> None:
        """Give the activity dock only the height it needs, once sizes are known."""
        height = self._split.winfo_height()
        if height <= 1:
            self.root.after(50, self._place_sash)
            return
        with contextlib.suppress(Exception):
            self._split.sashpos(0, max(200, height - DOCK_HEIGHT))

    def _header(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(theme.PAD_XL, theme.PAD_L))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="YOLO Dataset Builder", style="HeaderTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Clean, reproducible training datasets from local image folders.",
            style="HeaderSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, text="RUNS LOCALLY", style="Badge.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="e"
        )

    def _nav(self) -> None:
        bar = ttk.Frame(self.root, style="App.TFrame", padding=(theme.PAD_XL, theme.PAD_M, theme.PAD_XL, theme.PAD_M))
        bar.grid(row=1, column=0, sticky="ew")
        for column, (name, label) in enumerate(
            (("build", "Build & Verify"), ("export", "Export"))
        ):
            ttk.Radiobutton(
                bar,
                text=label,
                value=name,
                variable=self.page,
                style="Nav.Toolbutton",
                command=self._show_page,
            ).grid(row=0, column=column, sticky="w", padx=(0, theme.PAD_S))
        ttk.Frame(bar, style="Rule.TFrame", height=1).grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(theme.PAD_M, 0)
        )
        bar.columnconfigure(3, weight=1)

    def _show_page(self) -> None:
        self.pages[self.page.get()].tkraise()

    # ------------------------------------------------------------ pages
    def _build_page(self, body: ttk.Frame) -> None:
        pad = {"sticky": "nsew", "pady": (0, theme.PAD_M)}

        setup = Card(body, "Dataset setup", "Where images are read from, and where the dataset is written.")
        setup.grid(row=0, column=0, columnspan=2, **pad)
        self._path_row(setup, "source", must_exist=True)
        self._path_row(setup, "output", must_exist=False)
        row = setup.next_row()
        ttk.Label(setup, textvariable=self.estimate, style="Estimate.TLabel").grid(
            row=row, column=1, columnspan=3, sticky="w", pady=(theme.PAD_M, 0)
        )

        options = Card(body, "Labels & splits", "Deterministic, reproducible train/val/test splits.")
        options.grid(row=1, column=0, padx=(0, theme.PAD_S), **pad)
        self._entry_row(options, "classes")
        self._group_row(options, fields.GROUP_BY_NAME["splits"])
        self._entry_row(options, "seed", button=("Randomize", self.randomize_seed))
        self._check_row(options, "empty_labels")

        advanced = Card(body, "Scanning & quality", "File types, parallelism, deduplication, and quality gates.")
        advanced.grid(row=1, column=1, padx=(theme.PAD_S, 0), **pad)
        self._entry_row(advanced, "extensions")
        self._entry_row(advanced, "workers")
        self._entry_row(advanced, "near_duplicate_threshold")
        self._group_row(advanced, fields.GROUP_BY_NAME["quality"])

        self._command_row(body, 2, ("build", "verify", "stats"), extra=True)

    def _export_page(self, body: ttk.Frame) -> None:
        pad = {"sticky": "ew", "pady": (0, theme.PAD_M + theme.PAD_S)}

        source = Card(
            body,
            "Dataset to export",
            "A built dataset, a split-first folder, or a flat folder of paired images and .txt labels.",
        )
        source.grid(row=0, column=0, columnspan=2, **pad)
        self._path_row(source, "output", must_exist=True, label="Dataset folder")

        model = Card(body, "Model & GPU", "Base weights and optional live Vast.ai launch.")
        model.grid(row=1, column=0, padx=(0, theme.PAD_S), **pad)
        self._entry_row(model, "train_model", button=("Browse", self._pick_model))
        self._entry_row(model, "train_classes")
        self._group_row(model, fields.GROUP_BY_NAME["training"])
        self._entry_row(model, "vastai_instance")

        bundle = Card(body, "Bundle options", "Compression and class filtering.")
        bundle.grid(row=1, column=1, padx=(theme.PAD_S, 0), **pad)
        self._check_row(bundle, "export_onnx")
        self._check_row(bundle, "compress_enabled")
        self._group_row(bundle, fields.GROUP_BY_NAME["compress"])
        self._entry_row(bundle, "skip_classes")

        row = self._command_row(body, 2, ("export",))
        ttk.Label(
            body,
            text=(
                "The bundle contains images, labels, dataset.yaml, train.py, requirements.txt "
                "and entrypoint.sh. A native Vast.ai launcher is written beside the zip. "
                "The source folder is never modified."
            ),
            style="Subtitle.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w")

    # ------------------------------------------------------------- rows
    def _message(self, card: Card, key: str, hint: str) -> None:
        """One line under a control that shows its hint, or its error in red."""
        self.hints.setdefault(key, hint)
        row = card.next_row()
        label = ttk.Label(card, text="", style="Error.TLabel", justify="left")
        label.grid(row=row, column=1, columnspan=3, sticky="w", pady=(0, theme.PAD_S))
        label.grid_remove()
        self.messages.setdefault(key, []).append(label)

    def _label(self, card: Card, row: int, text: str) -> None:
        ttk.Label(card, text=text, style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, theme.PAD_M), pady=(theme.PAD_S, 1)
        )

    def _card_button(
        self, card: Card, row: int, column: int, text: str, command: Callable[[], None]
    ) -> None:
        button = ttk.Button(card, text=text, command=command, style="Ghost.TButton")
        button.grid(row=row, column=column, padx=(theme.PAD_S, 0), pady=(theme.PAD_S, 1))
        self.buttons.append(button)

    def _entry_row(
        self, card: Card, name: str, button: tuple[str, Callable[[], None]] | None = None
    ) -> None:
        spec = fields.BY_NAME[name]
        row = card.next_row()
        self._label(card, row, spec.label)
        entry = ttk.Entry(card, textvariable=self.vars[name], style="Input.TEntry")
        entry.grid(row=row, column=1, sticky="ew", pady=(theme.PAD_S, 1))
        self.entries.setdefault(name, []).append(entry)
        if button:
            self._card_button(card, row, 2, button[0], button[1])
        self._message(card, name, spec.hint)

    def _group_row(self, card: Card, group: fields.Group) -> None:
        row = card.next_row()
        self._label(card, row, group.label)
        holder = ttk.Frame(card, style="Card.TFrame")
        holder.grid(row=row, column=1, columnspan=3, sticky="w", pady=(theme.PAD_S, 1))
        for column, member in enumerate(group.members):
            spec = fields.BY_NAME[member]
            cell = ttk.Frame(holder, style="Card.TFrame")
            cell.grid(row=0, column=column, padx=(0, theme.PAD_M + theme.PAD_S))
            ttk.Label(cell, text=spec.label.upper(), style="GroupLabel.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            entry = ttk.Entry(
                cell,
                textvariable=self.vars[member],
                width=spec.width or 8,
                style="Input.TEntry",
            )
            entry.grid(row=1, column=0, sticky="w", pady=(3, 0))
            self.entries.setdefault(member, []).append(entry)
        self._message(card, group.name, group.hint)

    def _check_row(self, card: Card, name: str) -> None:
        spec = fields.BY_NAME[name]
        row = card.next_row()
        ttk.Checkbutton(
            card,
            text=spec.label,
            variable=self.vars[name],
            style="Switch.TCheckbutton",
        ).grid(row=row, column=1, columnspan=3, sticky="w", pady=(theme.PAD_S, 1))

    def _path_row(
        self, card: Card, name: str, must_exist: bool, label: str | None = None
    ) -> None:
        spec = fields.BY_NAME[name]
        row = card.next_row()
        self._label(card, row, label or spec.label)
        entry = ttk.Entry(card, textvariable=self.vars[name], style="Input.TEntry")
        entry.grid(row=row, column=1, sticky="ew", pady=(theme.PAD_S, 1))
        self.entries.setdefault(name, []).append(entry)
        if must_exist:
            self._card_button(card, row, 2, "Browse", lambda: self._pick_existing(name))
        else:
            self._card_button(card, row, 2, "Choose location", self._pick_new_output)
            self._card_button(
                card, row, 3, "Browse existing", lambda: self._pick_existing(name)
            )
        self._message(card, name, spec.hint)

    def _command_row(
        self, body: ttk.Frame, row: int, names: tuple[str, ...], extra: bool = False
    ) -> int:
        """Grid the action buttons plus a shortcut legend; return the next free row."""
        bar = ttk.Frame(body, style="App.TFrame")
        bar.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, theme.PAD_S))
        for name in names:
            style = "Primary.TButton" if name in ("build", "export") else "Secondary.TButton"
            button = ttk.Button(
                bar,
                text=COMMAND_LABELS[name],
                command=lambda target=name: self.start(target),
                style=style,
            )
            button.pack(side="left", padx=(0, theme.PAD_M))
            self.buttons.append(button)
            self.actions[name] = button
        if extra:
            for text, command in (
                ("Load config", self.load_config),
                ("Save config", self.save_config),
            ):
                button = ttk.Button(bar, text=text, command=command, style="Ghost.TButton")
                button.pack(side="left", padx=(0, theme.PAD_M))
                self.buttons.append(button)
        legend = "  ·  ".join(
            f"{COMMAND_LABELS[name]} {key}"
            for name, _, _, _, _, key in COMMANDS
            if name in names
        )
        ttk.Label(body, text=legend, style="Subtitle.TLabel").grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, theme.PAD_M)
        )
        return row + 2

    # --------------------------------------------------------- activity
    def _activity(self, parent: ttk.Widget) -> ttk.Frame:
        dock = ttk.Frame(
            parent, style="Header.TFrame", padding=(theme.PAD_XL, theme.PAD_M)
        )
        dock.columnconfigure(0, weight=1)
        dock.rowconfigure(3, weight=1)

        top = ttk.Frame(dock, style="Header.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        self.status_label = ttk.Label(top, textvariable=self.status, style="Status.TLabel")
        self.status_label.pack(side="left")
        self.elapsed = StringVar(value="")
        ttk.Label(top, textvariable=self.elapsed, style="Elapsed.TLabel").pack(
            side="left", padx=(theme.PAD_M, 0)
        )
        self.open_output_button = ttk.Button(
            top, text="Open folder", command=self.open_output, style="Quiet.TButton"
        )
        self.open_report_button = ttk.Button(
            top, text="Open report", command=self.open_report, style="Quiet.TButton"
        )
        self.log_toggle = ttk.Button(
            top, text="Hide log", command=self.toggle_log, style="Quiet.TButton"
        )
        # Packed right-to-left, so this reads in reverse of the on-screen order.
        for button in (
            self.log_toggle,
            ttk.Button(top, text="Copy log", command=self.copy_log, style="Quiet.TButton"),
            ttk.Button(top, text="Clear log", command=self.clear_log, style="Quiet.TButton"),
            ttk.Button(
                top,
                text="Reset preferences",
                command=self.reset_preferences,
                style="Quiet.TButton",
            ),
            self.open_report_button,
            self.open_output_button,
        ):
            button.pack(side="right", padx=(theme.PAD_S, 0))

        self.progress = ttk.Progressbar(dock, style="Bar.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(theme.PAD_M, theme.PAD_M))

        summary = ttk.Frame(dock, style="Raised.TFrame", padding=(0, 2))
        summary.grid(row=2, column=0, sticky="ew", pady=(0, theme.PAD_M))
        for column, (key, caption) in enumerate(
            (
                ("total_images", "IMAGES"),
                ("unique_images", "UNIQUE"),
                ("duplicates_removed", "DUPLICATES"),
                ("issues", "ISSUES"),
            )
        ):
            summary.columnconfigure(column, weight=1)
            value = ttk.Label(summary, text="—", style="Metric.TLabel", anchor="center")
            value.grid(row=0, column=column, sticky="ew")
            ttk.Label(
                summary, text=caption, style="MetricCaption.TLabel", anchor="center"
            ).grid(row=1, column=column, sticky="ew")
            self.metrics[key] = value

        self.log = ScrolledText(
            dock,
            height=6,
            state="disabled",
            wrap="word",
            relief="flat",
            borderwidth=0,
            background=theme.LOG_BG,
            foreground=theme.TEXT,
            insertbackground=theme.TEXT,
            selectbackground=theme.ACCENT_LO,
            font=(theme.MONO, 9),
            padx=theme.PAD_M + theme.PAD_S,
            pady=theme.PAD_M + 2,
        )
        self.log.grid(row=3, column=0, sticky="nsew")
        for tag, colour in (
            ("info", theme.MUTED),
            ("success", theme.SUCCESS),
            ("warning", theme.WARNING),
            ("error", theme.DANGER),
            ("output", theme.TEXT),
        ):
            self.log.tag_configure(tag, foreground=colour)
        return dock

    def toggle_log(self) -> None:
        if self.log.winfo_manager():
            self.log.grid_remove()
            self.log_toggle.configure(text="Show log")
        else:
            self.log.grid()
            self.log_toggle.configure(text="Hide log")

    def log_line(self, text: str, tag: str = "info") -> None:
        self.log.configure(state="normal")
        stamp = time.strftime("%H:%M:%S")  # local time: this is a local desktop log
        self.log.insert("end", f"{stamp}  {text}\n", tag)
        excess = int(self.log.index("end-1c").split(".")[0]) - LOG_LIMIT
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.log_line("Activity log cleared.")

    def copy_log(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log.get("1.0", "end-1c"))
        self.log_line("Activity log copied to the clipboard.", "success")

    def set_status(self, text: str, kind: str = "idle") -> None:
        self.status.set(text)
        self.status_label.configure(foreground=theme.STATUS_COLOURS[kind])

    # ------------------------------------------------------- validation
    def _parse_all(self) -> tuple[dict[str, Any], dict[str, str]]:
        values: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for spec in FIELDS:
            if spec.boolean:
                values[spec.name] = self.value(spec.name)
                continue
            try:
                values[spec.name] = spec.parse(self.value(spec.name))
                errors[spec.name] = ""
            except ValueError as exc:
                errors[spec.name] = str(exc)
        return values, errors

    def _validate(self) -> None:
        values, errors = self._parse_all()

        # Collapse per-field errors onto the line each control actually shows.
        display: dict[str, str] = {}
        for spec in FIELDS:
            if spec.boolean:
                continue
            key = spec.group or spec.name
            if errors[spec.name] and not display.get(key):
                display[key] = errors[spec.name]
        for group in GROUPS:
            members_ok = all(not errors[name] for name in group.members)
            if group.rule and members_ok and not display.get(group.name):
                display[group.name] = group.rule(values)

        # A missing source only blocks Build, so it is reported separately from
        # the form-wide errors that block every command.
        source = values["source"]
        self.source_ok = bool(source) and Path(source).is_dir()
        if source and not self.source_ok:
            display["source"] = "This folder does not exist."
        self.errors = {key: message for key, message in display.items() if message}
        self.values = values

        for spec in FIELDS:
            if spec.boolean:
                continue
            invalid = bool(self.errors.get(spec.group or spec.name))
            for entry in self.entries.get(spec.name, []):
                entry.configure(style="Invalid.TEntry" if invalid else "Input.TEntry")
        for key, labels in self.messages.items():
            message = self.errors.get(key, "")
            for label in labels:
                if message:
                    label.configure(text=message)
                    label.grid()
                else:
                    label.grid_remove()
        self._update_buttons()

    def _output_path(self) -> Path | None:
        raw = str(self.value("output")).strip()
        return Path(raw) if raw else None

    def _update_buttons(self) -> None:
        if self._running:
            for button in self.buttons:
                button.configure(state="disabled")
            self.open_output_button.configure(state="disabled")
            self.open_report_button.configure(state="disabled")
            return
        for button in self.buttons:
            button.configure(state="normal")

        # "source" is excluded: Verify, Statistics and Export never read it.
        form_ok = not (self.errors.keys() - {"source"})
        output = self._output_path()
        self.actions["build"].configure(
            state="normal"
            if form_ok and self.source_ok and output and not output.exists()
            else "disabled"
        )
        for name in ("verify", "stats", "export"):
            self.actions[name].configure(
                state="normal" if form_ok and output and output.is_dir() else "disabled"
            )
        self.open_output_button.configure(
            state="normal" if output and output.is_dir() else "disabled"
        )
        report = output / "reports" / "report.html" if output else None
        self.open_report_button.configure(
            state="normal" if report and report.exists() else "disabled"
        )

    def build_config(self) -> dict[str, Any]:
        """Assemble a full config from the form, raising ValueError if invalid."""
        config = copy.deepcopy(self.config_base)
        for spec in FIELDS:
            raw = self.value(spec.name)
            fields.assign(config, spec.path, raw if spec.boolean else spec.parse(raw))
        validate_config(config)
        return config

    # -------------------------------------------------------- execution
    def start(self, name: str) -> None:
        if self._running:
            return
        _, label, function, needs_output, _, _ = next(
            command for command in COMMANDS if command[0] == name
        )
        try:
            config = self.build_config()
            self._preflight(name, config, needs_output)
        except (ValueError, TypeError, KeyError) as exc:
            self.log_line(str(exc), "error")
            messagebox.showerror("Cannot run yet", str(exc))
            return
        if name == "build" and not self._confirm(config):
            return
        launch_vastai = name == "export" and self._confirm_vastai(config)
        self._run(name, label, function, config, launch_vastai)

    def _preflight(self, name: str, config: dict[str, Any], needs_output: bool) -> None:
        source = Path(config["source_dir"]) if config["source_dir"] else None
        output = Path(config["output_dir"])
        if name == "build":
            if not source or not source.is_dir():
                raise ValueError("Choose an existing source folder.")
            if output.exists():
                raise ValueError(
                    "That dataset folder already exists. Build always writes a new "
                    "folder, so pick a name that is not in use."
                )
            resolved_source, resolved_output = source.resolve(), output.resolve()
            if (
                resolved_output == resolved_source
                or resolved_source in resolved_output.parents
                or resolved_output in resolved_source.parents
            ):
                raise ValueError(
                    "The source and dataset folders must be separate; neither may "
                    "contain the other."
                )
        elif needs_output and not output.is_dir():
            raise ValueError(f"{COMMAND_LABELS[name]} needs an existing folder.")

    def _run(
        self,
        name: str,
        label: str,
        function: Callable[..., int],
        config: dict[str, Any],
        launch_vastai: bool = False,
    ) -> None:
        self._running = True
        self._update_buttons()
        self._started_at = time.monotonic()
        self._progress_state = (label, 0, 0)
        self._shown_progress = None
        self._phase_started_at = self._started_at
        self._phase_start_done = 0
        self._logged_phase = ""
        self._logged_progress = -1
        self.set_status(f"{label}…", "busy")
        self.log_line(f"{label} started.", "info")
        input_path = config["source_dir"] if name == "build" else config["output_dir"]
        self.log_line(f"Input: {input_path}", "info")
        if name == "build":
            self.log_line(
                f"Output: {config['output_dir']} · workers: {config['workers'] or 'auto'}",
                "info",
            )
        elif name == "export":
            train = config["train"]
            compression = config["compress"]
            self.log_line(
                f"Training: {train['model']} · {train['epochs']} epochs · "
                f"imgsz {train['imgsz']} · batch {train['batch']} · device {train['device']}",
                "info",
            )
            self.log_line(
                "Compression: "
                + (
                    f"{compression['max_size']}px / JPEG {compression['jpeg_quality']}"
                    if compression["enabled"]
                    else "off"
                ),
                "info",
            )

        def worker() -> None:
            stream = LineStream(self._output)
            try:
                # ponytail: process-wide stdout swap. Safe because commands are
                # serialised behind self._running and nothing else here prints.
                # Upgrade path is a logger passed into the builder.
                with contextlib.redirect_stdout(stream):
                    result = function(config, self._report)
                if result == 0 and launch_vastai:
                    self._launch_vastai(config, stream)
                stream.flush()
                self._outcome = ("done", name, label, result)
            except Exception as exc:  # noqa: BLE001 - never strand the interface
                stream.flush()
                self._outcome = ("failed", label, str(exc) or exc.__class__.__name__)

        threading.Thread(target=worker, daemon=True).start()

    def _confirm_vastai(self, config: dict[str, Any]) -> bool:
        instance_id = config["vastai"]["instance_id"]
        if not instance_id:
            return False
        return messagebox.askyesno(
            "Start Vast.ai training?",
            f"Export, upload to running instance {instance_id}, and start training?\n\n"
            "The training job runs remotely after upload. Vast.ai billing continues "
            "until you stop or destroy the instance.",
        )

    @staticmethod
    def _launch_vastai(config: dict[str, Any], stream: LineStream) -> None:
        dataset = Path(config["output_dir"]).resolve()
        suffix = ".ps1" if sys.platform == "win32" else ".sh"
        launcher = dataset.parent / f"{dataset.name}_train{suffix}"
        instance_id = config["vastai"]["instance_id"]
        command = (
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher),
                instance_id,
            ]
            if sys.platform == "win32"
            else [str(launcher), instance_id]
        )
        completed = subprocess.run(
            command,
            cwd=launcher.parent,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        stream.write(completed.stdout)
        stream.write(completed.stderr)
        if completed.returncode:
            raise RuntimeError(
                f"Vast.ai launch failed with exit code {completed.returncode}."
            )

    def _report(self, phase: str, done: int, total: int) -> None:
        """Called from the worker thread; the poller renders it on the UI thread."""
        if phase != self._progress_state[0]:
            self._phase_started_at = time.monotonic()
            self._phase_start_done = max(0, done - 1)
        self._progress_state = (phase, done, total)

    def _remaining(self) -> str:
        _, done, total = self._progress_state
        progressed = done - self._phase_start_done
        phase_elapsed = time.monotonic() - self._phase_started_at
        if total <= done or progressed <= 0 or phase_elapsed < 0.5:
            return ""
        seconds = max(1, round(phase_elapsed * (total - done) / progressed))
        return str(timedelta(seconds=seconds))

    def _drain(self) -> None:
        while True:
            try:
                self.log_line(self._output.get_nowait(), "output")
            except queue.Empty:
                return

    def _poll(self) -> None:
        """The only place worker results touch Tk, since Tk is not thread-safe.

        Background threads publish to plain attributes and a queue; everything
        is rendered here, on the main thread.
        """
        self._drain()
        while True:
            try:
                self._show_estimate(*self._estimate_results.get_nowait())
            except queue.Empty:
                break
        if self._running:
            self._render_progress()
            elapsed = f"{time.monotonic() - self._started_at:.0f}s elapsed"
            remaining = self._remaining()
            self.elapsed.set(
                f"{elapsed} · ~{remaining} left" if remaining else elapsed
            )
        outcome, self._outcome = self._outcome, None
        if outcome is not None:
            self._render_progress()
            self._drain()
            kind, *rest = outcome
            if kind == "done":
                self._finished(*rest)
            else:
                self._failed(*rest)
        self.root.after(POLL_MS, self._poll)

    def _render_progress(self) -> None:
        state = self._progress_state
        if state == self._shown_progress:
            return
        self._shown_progress = state
        phase, done, total = state
        if phase != self._logged_phase:
            self._logged_phase = phase
            self._logged_progress = 0
            self.log_line(
                f"{phase} started" + (f" ({total:,} items)." if total else "."),
                "info",
            )
        if total > 0:
            if self._bar_mode != "determinate":
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self._bar_mode = "determinate"
            self.progress.configure(maximum=total, value=done)
            self.set_status(f"{phase} {done:,}/{total:,}", "busy")
            milestone = min(10, done * 10 // total)
            if milestone > self._logged_progress:
                self._logged_progress = milestone
                remaining = self._remaining()
                self.log_line(
                    f"{phase}: {done:,}/{total:,} ({done / total:.0%})"
                    + (f" · ~{remaining} left" if remaining else ""),
                    "output",
                )
        else:
            if self._bar_mode != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
                self._bar_mode = "indeterminate"
            self.set_status(f"{phase}…", "busy")

    def _stop_bar(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._bar_mode = ""
        self._running = False
        self.elapsed.set("")

    def _finished(self, name: str, label: str, result: int) -> None:
        took = time.monotonic() - self._started_at
        self._stop_bar()
        ok = result == 0
        self.set_status(
            f"{label} complete" if ok else f"{label} finished with issues",
            "success" if ok else "warning",
        )
        self.log_line(
            f"{label} finished in {took:.1f}s (exit code {result}).",
            "success" if ok else "warning",
        )
        if name == "build" and ok:
            self._check_artifacts()
        self._load_summary()
        self._validate()

    def _failed(self, label: str, message: str) -> None:
        self._stop_bar()
        self.set_status(f"{label} failed", "error")
        self.log_line(message, "error")
        self._validate()
        messagebox.showerror(f"{label} failed", message)

    def _check_artifacts(self) -> None:
        output = self._output_path()
        if output is None:
            return
        missing = [
            artifact
            for artifact in REQUIRED_OUTPUT_ARTIFACTS
            if not (output / artifact).exists()
        ]
        if missing:
            self.set_status("Build finished, but output is incomplete", "warning")
            self.log_line(f"Missing expected files: {', '.join(missing)}", "error")
        else:
            self.log_line(
                "Output check passed: images, labels, config and reports all present.",
                "success",
            )

    def _load_summary(self) -> None:
        output = self._output_path()
        report = output / "reports" / "stats.json" if output else None
        if not report or not report.exists():
            return
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.log_line(f"Could not read the build summary: {exc}", "warning")
            return
        issues = sum(
            int(data.get(key, 0))
            for key in (
                "corrupt_images",
                "images_missing_labels",
                "invalid_labels",
                "labels_missing_images",
                "duplicate_filenames",
            )
        )
        for key, value in (
            ("total_images", data.get("total_images", "—")),
            ("unique_images", data.get("unique_images", data.get("valid_images", "—"))),
            ("duplicates_removed", data.get("duplicates_removed", 0)),
            ("issues", issues),
        ):
            self.metrics[key].configure(text=f"{value:,}" if isinstance(value, int) else str(value))

    # ---------------------------------------------------------- dialogs
    def _confirm(self, config: dict[str, Any]) -> bool:
        dialog = Toplevel(self.root)
        dialog.title("Confirm build")
        dialog.configure(background=theme.SURFACE)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, style="Card.TFrame", padding=theme.PAD_XL)
        body.grid(sticky="nsew")
        ttk.Label(body, text="Ready to build?", style="HeaderTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            body,
            text="A new dataset folder will be created with these settings.",
            style="CardHint.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, theme.PAD_L))
        details = (
            ("Source", config["source_dir"]),
            ("Dataset", config["output_dir"]),
            ("Classes", ", ".join(config["classes"])),
            (
                "Splits",
                "  ".join(
                    f"{key} {value:.0%}" for key, value in config["splits"].items()
                ),
            ),
            ("Seed", str(config["seed"])),
            (
                "Empty labels",
                "Created when missing"
                if config["create_empty_labels"]
                else "Not created",
            ),
        )
        for row, (label, value) in enumerate(details, start=2):
            ttk.Label(body, text=label, style="GroupLabel.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, theme.PAD_L), pady=3
            )
            ttk.Label(
                body, text=value, style="CardHint.TLabel", wraplength=430, justify="left"
            ).grid(row=row, column=1, sticky="w", pady=3)

        result = BooleanVar(value=False)

        def accept() -> None:
            result.set(True)
            dialog.destroy()

        actions = ttk.Frame(body, style="Card.TFrame")
        actions.grid(row=len(details) + 2, column=0, columnspan=2, sticky="e", pady=(theme.PAD_L, 0))
        ttk.Button(actions, text="Cancel", command=dialog.destroy, style="Secondary.TButton").pack(
            side="left", padx=(0, theme.PAD_M)
        )
        confirm = ttk.Button(actions, text="Build dataset", command=accept, style="Primary.TButton")
        confirm.pack(side="left")
        dialog.bind("<Escape>", lambda _: dialog.destroy())
        dialog.bind("<Return>", lambda _: accept())
        self._centre(dialog)
        confirm.focus_set()
        dialog.grab_set()
        dialog.wait_window()
        return result.get()

    def _centre(self, window: Toplevel) -> None:
        window.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - window.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - window.winfo_height()) // 3
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    # ---------------------------------------------------------- pickers
    def _pick_existing(self, name: str) -> None:
        title = (
            "Choose the folder of source images"
            if name == "source"
            else "Choose an existing dataset folder"
        )
        chosen = filedialog.askdirectory(mustexist=True, title=title)
        if chosen:
            self.vars[name].set(chosen)

    def _pick_new_output(self) -> None:
        parent = filedialog.askdirectory(
            mustexist=True, title="Choose where to create the dataset"
        )
        if parent:
            candidate = self._new_output_name(parent)
            if candidate:
                self.vars["output"].set(str(candidate))

    def _pick_model(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose base model weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if chosen:
            self.vars["train_model"].set(chosen)

    def _new_output_name(self, parent: str | Path) -> Path | None:
        name = simpledialog.askstring(
            "New dataset folder", "Folder name:", parent=self.root
        )
        if not name or not name.strip():
            return None
        candidate = Path(parent) / name.strip()
        if candidate.name != name.strip():
            messagebox.showerror(
                "Invalid folder name",
                "Enter a single folder name, without path separators.",
            )
            return None
        return candidate

    def _enable_drop_targets(self) -> None:
        if not hasattr(self.root, "drop_target_register"):
            return
        try:
            from tkinterdnd2 import DND_FILES  # type: ignore[import-not-found]
        except (ImportError, OSError, RuntimeError):
            return
        for name in ("source", "output"):
            for entry in self.entries.get(name, []):
                with contextlib.suppress(Exception):
                    entry.drop_target_register(DND_FILES)
                    entry.dnd_bind(
                        "<<Drop>>", lambda event, key=name: self._receive_drop(event, key)
                    )

    def _receive_drop(self, event: Any, name: str) -> str:
        paths = self.root.tk.splitlist(event.data)
        if paths:
            path = Path(paths[0])
            if path.is_dir():
                self.vars[name].set(str(path))
            else:
                self.log_line("Drop a folder, not an individual file.", "warning")
        return "break"

    # --------------------------------------------------------- estimate
    def _update_estimate(self) -> None:
        self._estimate_token += 1
        token = self._estimate_token
        raw = str(self.value("source")).strip()
        source = Path(raw) if raw else None
        if not source or not source.is_dir():
            self.estimate.set("Choose a source folder to see an estimate.")
            return
        try:
            allowed = set(fields.suffixes(str(self.value("extensions"))))
        except ValueError:
            self.estimate.set("Fix the extensions list to see an estimate.")
            return
        self.estimate.set("Counting source images…")

        def count() -> None:
            files = total_bytes = 0
            try:
                for path in source.rglob("*"):
                    if path.is_file() and path.suffix.lower() in allowed:
                        files += 1
                        total_bytes += path.stat().st_size
            except OSError:
                pass
            self._estimate_results.put((token, files, total_bytes))

        threading.Thread(target=count, daemon=True).start()

    def _show_estimate(self, token: int, files: int, total_bytes: int) -> None:
        if token != self._estimate_token:
            return
        if not files:
            self.estimate.set("No images with those extensions were found here.")
            return
        # Rough throughput guess; the progress bar reports the real rate.
        seconds = max(5.0, files / 35, total_bytes / (75 * 1024 * 1024))
        duration = (
            f"{round(seconds)}s"
            if seconds < 60
            else f"{seconds // 60:.0f}m {round(seconds % 60)}s"
        )
        size = (
            f"{total_bytes / 1024**2:.0f} MB"
            if total_bytes < 1024**3
            else f"{total_bytes / 1024**3:.1f} GB"
        )
        self.estimate.set(f"{files:,} images  ·  {size}  ·  roughly {duration}")

    # ----------------------------------------------------------- config
    def load_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            config = load_config(path)
            self.config_base = copy.deepcopy(config)
            for spec in FIELDS:
                value = fields.lookup(config, spec.path)
                self.vars[spec.name].set(value if spec.boolean else spec.fmt(value))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.log_line(f"Could not load that configuration: {exc}", "error")
            messagebox.showerror("Could not load configuration", str(exc))
            return
        self.log_line(f"Loaded configuration from {path}", "success")

    def save_config(self) -> None:
        try:
            config = self.build_config()
        except (ValueError, TypeError) as exc:
            self.log_line(str(exc), "error")
            messagebox.showerror("Invalid settings", str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml", filetypes=[("YAML", "*.yaml *.yml")]
        )
        if not path:
            return
        try:
            Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        except OSError as exc:
            self.log_line(f"Could not save that configuration: {exc}", "error")
            messagebox.showerror("Could not save configuration", str(exc))
            return
        self.log_line(f"Saved configuration to {path}", "success")

    def randomize_seed(self) -> None:
        self.vars["seed"].set(str(secrets.randbelow(2_147_483_647) + 1))
        self.log_line("Generated a new random seed.", "success")

    def open_output(self) -> None:
        output = self._output_path()
        if output and output.is_dir():
            webbrowser.open(output.resolve().as_uri())

    def open_report(self) -> None:
        output = self._output_path()
        report = output / "reports" / "report.html" if output else None
        if report and report.exists():
            webbrowser.open(report.resolve().as_uri())

    # -------------------------------------------------------- shortcuts
    def _bind_shortcuts(self) -> None:
        for name, _, _, _, sequence, _ in COMMANDS:
            self.root.bind(sequence, lambda _event, target=name: self.start(target))
        bindings = {
            "<Control-o>": self.open_output,
            "<Control-Shift-O>": self.load_config,
            "<Control-s>": self.save_config,
            "<Control-l>": self.clear_log,
            "<F5>": self._update_estimate,
        }
        for sequence, handler in bindings.items():
            self.root.bind(sequence, lambda _event, run=handler: run())


def _enable_dpi_awareness() -> None:
    """Render crisply on scaled Windows displays instead of being bitmap-stretched."""
    if os.name != "nt":
        return
    import ctypes

    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # per-monitor aware


def main() -> int:
    _enable_dpi_awareness()
    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore[import-not-found]

        root = TkinterDnD.Tk()
    except (ImportError, OSError, RuntimeError):
        root = Tk()
    DatasetBuilderGui(root)
    root.mainloop()
    return 0
