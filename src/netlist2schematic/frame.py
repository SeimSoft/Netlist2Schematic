"""Sheet frame: a dynamic border around the drawing, KiCad style.

The outer border has a grid: letters A, B, C ... down the left and right edge, numbers
1, 2, 3 ... along the top and bottom edge. Bottom right is the title block (circuit,
file path, date); bottom left shows the circuit's own block symbol (as a parent sheet
draws it, one pin per port). The frame grows with the drawing and always snaps to whole
grid cells.

Coordinates are grid units with the cell centre convention of the drawing (an integer
point is a cell centre), so frame edges are placed half a cell before the integers.
"""

from __future__ import annotations

import math
from datetime import date as _date

from dataclasses import dataclass

from .graphics import INK, Line, Prim, Text, bbox, transform
from .symbols import Symbol

FRAME_COLOR = "#6b7280"
COLUMN = 16  # cells per grid column (numbers along the top)
ROW = 12  # cells per grid row (letters down the side)
PAD = 4  # cells between the drawing and the inner border
MARGIN = 1  # cells of grid band between outer and inner border
TITLE_W = 40
TITLE_H = 7
TEXT = 0.42
SYMBOL_TITLE = 0.5


def _row_label(index: int) -> str:
    """A ... Z, AA, AB ... like spreadsheet rows."""
    label = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        label = chr(65 + rest) + label
    return label


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


@dataclass(frozen=True)
class Sheet:
    """Frame prims plus where the circuit's own symbol goes (integer grid point, so its
    pins sit on the grid like every part's)."""

    prims: list[Prim]
    symbol: Symbol | None
    origin: tuple[int, int]
    circuit: str


def symbol_prims(symbol: Symbol, origin: tuple[int, int], circuit: str) -> list[Prim]:
    """The own symbol at ``origin`` with the circuit name above it."""
    ox, oy = origin
    x0, y0, x1, _ = bbox(list(symbol.graphics))
    prims = [transform(p, 0, ox, oy) for p in symbol.graphics]
    prims.append(Text(ox + (x0 + x1) / 2, oy + y0 - 0.7, circuit, size=SYMBOL_TITLE))
    return prims


def sheet_frame(
    bounds: tuple[float, float, float, float],
    symbol: Symbol | None,
    circuit: str,
    source: str = "",
    today: str | None = None,
) -> Sheet:
    """Frame around ``bounds`` (grid units, from ``bbox`` of the drawing) with room for
    the own ``symbol`` bottom left."""
    today = today or _date.today().isoformat()
    bx0, by0, bx1, by1 = bounds
    sx0, sy0, sx1, sy1 = bbox(list(symbol.graphics)) if symbol is not None else (0.0, 0.0, 0.0, 0.0)
    sym_w, sym_h = (sx1 - sx0, sy1 - sy0 + 1.2) if symbol is not None else (0.0, 0.0)  # + title
    strip = max(TITLE_H + 2, sym_h + 3)
    left = bx0 - 0.5 - PAD
    top = by0 - 0.5 - PAD
    right = max(bx1 + 0.5 + PAD, left + TITLE_W + sym_w + 6)
    bottom = by1 + 0.5 + PAD + strip
    # Snap outward to whole grid cells.
    x0 = math.floor(left / COLUMN) * COLUMN
    y0 = math.floor(top / ROW) * ROW
    x1 = math.ceil(right / COLUMN) * COLUMN
    y1 = math.ceil(bottom / ROW) * ROW
    cols = (x1 - x0) // COLUMN
    rows = (y1 - y0) // ROW

    def p(e: float) -> float:  # edge coordinate -> drawing coordinate
        return e - 0.5

    prims: list[Prim] = []
    color = FRAME_COLOR

    def rect(a: float, b: float, c: float, d: float, width: float = 1.0) -> None:
        prims.append(Line(p(a), p(b), p(c), p(b), width=width, color=color))
        prims.append(Line(p(c), p(b), p(c), p(d), width=width, color=color))
        prims.append(Line(p(c), p(d), p(a), p(d), width=width, color=color))
        prims.append(Line(p(a), p(d), p(a), p(b), width=width, color=color))

    rect(x0, y0, x1, y1, width=1.6)
    rect(x0 + MARGIN, y0 + MARGIN, x1 - MARGIN, y1 - MARGIN)
    # Grid ticks and labels in the margin band.
    for k in range(cols):
        cx = x0 + k * COLUMN
        if k:
            prims.append(Line(p(cx), p(y0), p(cx), p(y0 + MARGIN), color=color))
            prims.append(Line(p(cx), p(y1 - MARGIN), p(cx), p(y1), color=color))
        centre = cx + COLUMN / 2
        prims.append(Text(p(centre), p(y0 + MARGIN / 2), str(k + 1), size=0.5, color=color))
        prims.append(Text(p(centre), p(y1 - MARGIN / 2), str(k + 1), size=0.5, color=color))
    for k in range(rows):
        cy = y0 + k * ROW
        if k:
            prims.append(Line(p(x0), p(cy), p(x0 + MARGIN), p(cy), color=color))
            prims.append(Line(p(x1 - MARGIN), p(cy), p(x1), p(cy), color=color))
        centre = cy + ROW / 2
        prims.append(Text(p(x0 + MARGIN / 2), p(centre), _row_label(k), size=0.5, color=color))
        prims.append(Text(p(x1 - MARGIN / 2), p(centre), _row_label(k), size=0.5, color=color))

    # Title block: bottom right, four rows.
    tx0, ty0 = x1 - MARGIN - TITLE_W, y1 - MARGIN - TITLE_H
    rect(tx0, ty0, x1 - MARGIN, y1 - MARGIN)
    row_h = TITLE_H / 4
    for k, (label, value) in enumerate(
        (("Schaltung", circuit), ("Datei", _shorten(source or circuit, 48)), ("Datum", today), ("Werkzeug", "netlist2schematic"))
    ):
        y = ty0 + (k + 0.5) * row_h
        if k:
            prims.append(Line(p(tx0), p(ty0 + k * row_h), p(x1 - MARGIN), p(ty0 + k * row_h), color=color))
        prims.append(Text(p(tx0 + 1), p(y), label, anchor="start", size=TEXT, color=color))
        prims.append(Text(p(tx0 + 9), p(y), value, anchor="start", size=TEXT, color=INK))

    # Own symbol: bottom left, its outline 1.5 cells inside the inner border.
    origin = (math.ceil(x0 + MARGIN + 1 - sx0), math.floor(y1 - MARGIN - 1.5 - sy1))
    return Sheet(prims, symbol, origin, circuit)


def frame_graphics(
    bounds: tuple[float, float, float, float],
    symbol: Symbol | None,
    circuit: str,
    source: str = "",
    today: str | None = None,
) -> list[Prim]:
    """All frame prims including the own symbol (for the SVG export)."""
    sheet = sheet_frame(bounds, symbol, circuit, source, today)
    if symbol is None:
        return sheet.prims
    return sheet.prims + symbol_prims(symbol, sheet.origin, circuit)


def frame_bounds(prims: list[Prim]) -> tuple[float, float, float, float]:
    return bbox(prims)
