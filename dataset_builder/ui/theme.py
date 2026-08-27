"""Dark colour palette and ttk styles for the dataset-builder interface."""

from __future__ import annotations

from tkinter import ttk

# Ink-and-violet palette: a near-neutral slate base so the accent colour is the
# only saturated thing on screen, which keeps long forms readable.
BG = "#0f1117"
SURFACE = "#171a23"
RAISED = "#1e222d"
BORDER = "#2a2f3d"
TEXT = "#e6e9f0"
MUTED = "#8d94a5"
FAINT = "#5d6577"
ACCENT = "#7c5cff"
ACCENT_HI = "#9a80ff"
ACCENT_LO = "#5b3fd6"
SUCCESS = "#3ecf8e"
WARNING = "#f5b544"
DANGER = "#ff5c5c"
LOG_BG = "#0b0d12"
INVALID_BG = "#2b1620"

UI = "Segoe UI"
MONO = "Cascadia Mono"

# 8pt spacing scale; every pad in the app is a multiple of these.
PAD_S, PAD_M, PAD_L, PAD_XL = 4, 8, 16, 24

STATUS_COLOURS = {
    "idle": MUTED,
    "busy": ACCENT_HI,
    "success": SUCCESS,
    "warning": WARNING,
    "error": DANGER,
}


def apply(root) -> ttk.Style:
    """Install every named style the interface uses and return the Style."""
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=BG)

    style.configure("App.TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("Raised.TFrame", background=RAISED)
    style.configure("Header.TFrame", background=SURFACE)
    style.configure("Rule.TFrame", background=BORDER)

    _labels(style)
    _inputs(style)
    _buttons(style)
    _misc(style)
    return style


def _labels(style: ttk.Style) -> None:
    text = (
        ("Title.TLabel", BG, TEXT, (UI, 20, "bold")),
        ("Subtitle.TLabel", BG, MUTED, (UI, 10)),
        ("HeaderTitle.TLabel", SURFACE, TEXT, (UI, 17, "bold")),
        ("HeaderSubtitle.TLabel", SURFACE, MUTED, (UI, 10)),
        ("CardTitle.TLabel", SURFACE, TEXT, (UI, 11, "bold")),
        ("CardHint.TLabel", SURFACE, MUTED, (UI, 9)),
        ("FieldLabel.TLabel", SURFACE, TEXT, (UI, 10)),
        ("GroupLabel.TLabel", SURFACE, MUTED, (UI, 8, "bold")),
        ("Hint.TLabel", SURFACE, FAINT, (UI, 8)),
        ("Error.TLabel", SURFACE, DANGER, (UI, 8)),
        ("Estimate.TLabel", SURFACE, ACCENT_HI, (UI, 9, "bold")),
        ("Status.TLabel", SURFACE, MUTED, (UI, 10, "bold")),
        ("Elapsed.TLabel", SURFACE, FAINT, (UI, 9)),
        ("Metric.TLabel", RAISED, TEXT, (UI, 15, "bold")),
        ("MetricCaption.TLabel", RAISED, MUTED, (UI, 7, "bold")),
    )
    for name, background, foreground, font in text:
        style.configure(name, background=background, foreground=foreground, font=font)
    style.configure(
        "Badge.TLabel",
        background=RAISED,
        foreground=ACCENT_HI,
        font=(UI, 8, "bold"),
        padding=(PAD_M, PAD_S),
    )


def _inputs(style: ttk.Style) -> None:
    for name, border in (("Input.TEntry", BORDER), ("Invalid.TEntry", DANGER)):
        style.configure(
            name,
            fieldbackground=INVALID_BG if border == DANGER else RAISED,
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            borderwidth=1,
            padding=(PAD_M, 7),
        )
        style.map(
            name,
            bordercolor=[("focus", ACCENT)],
            lightcolor=[("focus", ACCENT)],
            darkcolor=[("focus", ACCENT)],
        )
    style.configure(
        "Switch.TCheckbutton",
        background=SURFACE,
        foreground=MUTED,
        indicatorbackground=RAISED,
        indicatorforeground=TEXT,
        bordercolor=BORDER,
        lightcolor=RAISED,
        darkcolor=RAISED,
        focuscolor=SURFACE,
        font=(UI, 9),
        padding=(0, 3),
    )
    style.map(
        "Switch.TCheckbutton",
        background=[("active", SURFACE)],
        foreground=[("active", TEXT), ("selected", TEXT)],
        indicatorbackground=[("selected", ACCENT), ("active", BORDER)],
        indicatorforeground=[("selected", "#ffffff")],
        bordercolor=[("selected", ACCENT), ("active", ACCENT_LO)],
    )


def _buttons(style: ttk.Style) -> None:
    buttons = (
        ("Primary.TButton", ACCENT, "#ffffff", (UI, 10, "bold"), (PAD_L, 9), ACCENT_HI),
        ("Secondary.TButton", RAISED, TEXT, (UI, 10), (PAD_M + PAD_S, 9), BORDER),
        ("Quiet.TButton", SURFACE, MUTED, (UI, 9), (PAD_M, PAD_S + 1), RAISED),
        ("Ghost.TButton", RAISED, MUTED, (UI, 9), (PAD_M, PAD_S + 1), BORDER),
    )
    for name, background, foreground, font, padding, active in buttons:
        style.configure(
            name,
            background=background,
            foreground=foreground,
            font=font,
            padding=padding,
            borderwidth=0,
            focuscolor=background,
        )
        style.map(
            name,
            background=[("active", active), ("disabled", background)],
            foreground=[("disabled", FAINT), ("active", TEXT)],
        )
    # Segmented page navigation.
    style.configure(
        "Nav.Toolbutton",
        background=BG,
        foreground=MUTED,
        font=(UI, 10, "bold"),
        padding=(PAD_L, PAD_M),
        borderwidth=0,
        focuscolor=BG,
        anchor="center",
    )
    style.map(
        "Nav.Toolbutton",
        background=[("selected", SURFACE), ("active", RAISED)],
        foreground=[("selected", TEXT), ("active", TEXT)],
    )


def _misc(style: ttk.Style) -> None:
    style.configure(
        "Bar.Horizontal.TProgressbar",
        troughcolor=RAISED,
        background=ACCENT,
        bordercolor=RAISED,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
        thickness=6,
    )
    style.configure("App.TPanedwindow", background=BG)
    style.configure("App.TPanedwindow.Sash", sashthickness=6, gripcount=0, background=BORDER)
    style.configure("Sash", sashthickness=6, gripcount=0, background=BORDER)
