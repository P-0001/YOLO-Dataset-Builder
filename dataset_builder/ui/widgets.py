"""Small reusable widgets and helpers for the dataset-builder interface."""

from __future__ import annotations

import io
import queue
from tkinter import ttk

from . import theme


class Card(ttk.Frame):
    """A titled panel that hands out its own row numbers.

    Callers never write literal grid rows, which is what previously let two
    tabs collide on the same row indexes.
    """

    def __init__(self, parent: ttk.Widget, title: str, hint: str = "") -> None:
        super().__init__(
            parent, style="Card.TFrame", padding=(theme.PAD_M + 4, theme.PAD_M)
        )
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        self._row = 1
        if hint:
            ttk.Label(
                self,
                text=hint,
                style="CardHint.TLabel",
                wraplength=420,
                justify="left",
            ).grid(
                row=1, column=0, columnspan=4, sticky="w", pady=(2, 0)
            )
            self._row = 2
        ttk.Frame(self, style="Rule.TFrame", height=1).grid(
            row=self._row, column=0, columnspan=4, sticky="ew", pady=(theme.PAD_M + 2, 2)
        )
        self._row += 1

    def next_row(self) -> int:
        row = self._row
        self._row += 1
        return row


class LineStream(io.TextIOBase):
    """Collects text written by the builder and queues it a line at a time."""

    def __init__(self, sink: queue.Queue[str]) -> None:
        self._sink = sink
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._sink.put(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._sink.put(self._buffer.strip())
        self._buffer = ""
