"""Renderer-independent drawing primitives and the SVG writer.

All coordinates are in grid units: the centre of grid cell (x, y) is the point
(x, y). The Qt view and the SVG export draw the same primitive lists, so the
UI and the reports always show the same schematic.
"""

from __future__ import annotations

import math

import html
from dataclasses import dataclass, replace

INK = "#1f2937"
WIRE = "#1f2937"
ACCENT = "#b91c1c"


@dataclass(frozen=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 1.0
    color: str = INK
    dashed: bool = False


@dataclass(frozen=True)
class Poly:
    points: tuple[tuple[float, float], ...]
    closed: bool = False
    fill: bool = False
    width: float = 1.0
    color: str = INK
    opacity: float = 1.0  # fill opacity (group backgrounds)
    dashed: bool = False


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float
    fill: bool = False
    width: float = 1.0
    color: str = INK


@dataclass(frozen=True)
class Text:
    x: float
    y: float
    text: str
    anchor: str = "middle"  # start | middle | end
    size: float = 0.4
    color: str = INK
    title: str = ""  # full text for tooltips when ``text`` is shortened
    angle: float = 0.0  # degrees, SVG convention (positive = clockwise; -90 reads bottom-to-top)
    follow: bool = False  # turns with its symbol (pin names in IC boxes); kept readable (0 or -90 degrees)


Prim = Line | Poly | Circle | Text

CHAR_WIDTH = 0.6  # average glyph width relative to the font size


ORIENTATIONS = 8  # 0-3: quarter turns, 4-7: mirrored (x -> -x) then quarter turns


def rotate_point(x: float, y: float, orientation: int) -> tuple[float, float]:
    """Apply an orientation: optional horizontal mirror (orientation >= 4), then quarter turns."""
    if orientation % ORIENTATIONS >= 4:
        x = -x
    for _ in range(orientation % 4):
        x, y = -y, x
    return x, y


_FLIP = {"start": "end", "end": "start", "middle": "middle"}


def transform(prim: Prim, orientation: int, dx: float, dy: float) -> Prim:
    """Orient around the origin, then translate. Text stays upright (mirroring flips its anchor)."""

    def tp(x: float, y: float) -> tuple[float, float]:
        rx, ry = rotate_point(x, y, orientation)
        return rx + dx, ry + dy

    if isinstance(prim, Line):
        x1, y1 = tp(prim.x1, prim.y1)
        x2, y2 = tp(prim.x2, prim.y2)
        return replace(prim, x1=x1, y1=y1, x2=x2, y2=y2)
    if isinstance(prim, Poly):
        return replace(prim, points=tuple(tp(x, y) for x, y in prim.points))
    if isinstance(prim, Circle):
        x, y = tp(prim.x, prim.y)
        return replace(prim, x=x, y=y)
    x, y = tp(prim.x, prim.y)
    if prim.follow:
        return replace(prim, x=x, y=y, **_followed(prim, orientation))
    anchor = _FLIP[prim.anchor] if orientation % ORIENTATIONS >= 4 else prim.anchor
    return replace(prim, x=x, y=y, anchor=anchor)


def _followed(prim: Text, orientation: int) -> dict:
    """Angle and anchor of a text that turns with its symbol: the reading direction is
    transformed like a vector, then flipped if it would read right-to-left or downwards
    (the anchor swaps with it, so the text still grows away from the same point)."""
    rad = math.radians(prim.angle)
    dx, dy = rotate_point(round(math.cos(rad)), round(math.sin(rad)), orientation)
    anchor = prim.anchor
    if (dx, dy) in ((-1, 0), (0, 1)):  # right-to-left or top-to-bottom: read it the other way
        dx, dy = -dx, -dy
        anchor = _FLIP[anchor]
    return {"angle": 0.0 if dy == 0 else -90.0, "anchor": anchor}


def text_width(text: str, size: float) -> float:
    return len(text) * size * CHAR_WIDTH


def text_extent(p: Text) -> tuple[tuple[float, float], tuple[float, float]]:
    """((x0, x1), (y0, y1)) of a text, honouring anchor and a +-90 degree angle."""
    w = text_width(p.text, p.size)
    a0 = {"start": 0.0, "middle": -w / 2, "end": -w}[p.anchor]
    if abs(p.angle) == 90:
        # Along the text runs vertically; angle -90 reads bottom-to-top.
        sign = -1 if p.angle < 0 else 1
        y_a, y_b = p.y + sign * a0, p.y + sign * (a0 + w)
        return (p.x - p.size / 2, p.x + p.size / 2), (min(y_a, y_b), max(y_a, y_b))
    return (p.x + a0, p.x + a0 + w), (p.y - p.size / 2, p.y + p.size / 2)


def bbox(prims: list[Prim]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for p in prims:
        if isinstance(p, Line):
            xs += [p.x1, p.x2]
            ys += [p.y1, p.y2]
        elif isinstance(p, Poly):
            xs += [x for x, _ in p.points]
            ys += [y for _, y in p.points]
        elif isinstance(p, Circle):
            xs += [p.x - p.r, p.x + p.r]
            ys += [p.y - p.r, p.y + p.r]
        else:
            xs_t, ys_t = text_extent(p)
            xs += list(xs_t)
            ys += list(ys_t)
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    return min(xs), min(ys), max(xs), max(ys)


def to_svg(prims: list[Prim], scale: float = 24.0, margin: float = 1.5, title: str = "") -> str:
    """Standalone SVG document with a white background."""
    x0, y0, x1, y1 = bbox(prims)
    x0 -= margin
    y0 -= margin
    w = (x1 - x0 + margin) * scale
    h = (y1 - y0 + margin) * scale

    def X(v: float) -> str:
        return f"{(v - x0) * scale:.1f}"

    def Y(v: float) -> str:
        return f"{(v - y0) * scale:.1f}"

    stroke = scale / 16  # line width 1.0 in primitive units
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.1f} {h:.1f}" font-family="Helvetica, Arial, sans-serif">',
    ]
    if title:
        out.append(f"<title>{html.escape(title)}</title>")
    out.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="#ffffff"/>')
    for p in prims:
        if isinstance(p, Line):
            dash = f' stroke-dasharray="{stroke * 3:.1f} {stroke * 2:.1f}"' if p.dashed else ""
            out.append(
                f'<line x1="{X(p.x1)}" y1="{Y(p.y1)}" x2="{X(p.x2)}" y2="{Y(p.y2)}" '
                f'stroke="{p.color}" stroke-width="{stroke * p.width:.2f}" stroke-linecap="round"{dash}/>'
            )
        elif isinstance(p, Poly):
            pts = " ".join(f"{X(x)},{Y(y)}" for x, y in p.points)
            tag = "polygon" if p.closed else "polyline"
            fill = p.color if p.fill else "none"
            extra = f' fill-opacity="{p.opacity:g}"' if p.fill and p.opacity < 1 else ""
            stroke_color = "none" if p.width == 0 else p.color
            if p.dashed:
                extra += f' stroke-dasharray="{stroke * 3:.1f} {stroke * 2:.1f}"'
            out.append(
                f'<{tag} points="{pts}" fill="{fill}" stroke="{stroke_color}" '
                f'stroke-width="{stroke * p.width:.2f}" stroke-linejoin="round"{extra}/>'
            )
        elif isinstance(p, Circle):
            fill = p.color if p.fill else "none"
            out.append(
                f'<circle cx="{X(p.x)}" cy="{Y(p.y)}" r="{p.r * scale:.1f}" fill="{fill}" '
                f'stroke="{p.color}" stroke-width="{stroke * p.width:.2f}"/>'
            )
        else:
            tip = f"<title>{html.escape(p.title)}</title>" if p.title else ""
            rot = f' transform="rotate({p.angle:g} {X(p.x)} {Y(p.y)})"' if p.angle else ""
            out.append(
                f'<text x="{X(p.x)}" y="{Y(p.y)}" font-size="{p.size * scale:.1f}" fill="{p.color}" '
                f'text-anchor="{p.anchor}" dominant-baseline="central"{rot}>{tip}{html.escape(p.text)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)
