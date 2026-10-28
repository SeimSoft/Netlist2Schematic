"""Schematic symbols on the integer grid.

Every symbol has
- ``body``: occupied cells relative to its anchor (nothing else may sit there),
- ``pins``: one cell per terminal, just outside the body, where wires attach,
- ``pin_dirs``: unit vector pointing away from the body at each pin,
- ``graphics``: drawing primitives in the same coordinates (pin cell centres are
  integer points).

Standard symbols: resistor (IEEE zigzag or IEC box), capacitor, inductor,
diode, independent and controlled sources, BJT, MOSFET, switch, transformer.
Everything else (subcircuit instances, Python devices, unknown letters) becomes
an IC box. If the port names follow the naming convention (``_i``/``_ai`` inputs,
``_o``/``_ao`` outputs, upper case supply rails) the pins are grouped by function
(``directional_ic``: inputs left, outputs right, supplies on top, grounds at the
bottom); otherwise DIP style: ceil(n/2) rows, pin 1 top left, numbering counter
clockwise, pin numbers outside, port names inside.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .graphics import Circle, Line, Poly, Prim, Text, text_width
from .netlist import active_low

KIND_PASSIVE, KIND_SOURCE, KIND_ACTIVE, KIND_IC, KIND_PORT = range(5)
NUM_KINDS = 5

RESISTOR_STYLE = os.environ.get("N2S_RESISTOR", "ieee")  # "ieee" zigzag or "iec" box
LABEL_SIZE = 0.4
PIN_NUMBER_SIZE = 0.3
PIN_NAME_SIZE = 0.32


@dataclass(frozen=True)
class Symbol:
    kind: int
    body: tuple[tuple[int, int], ...]
    pins: tuple[tuple[int, int], ...]
    pin_nodes: tuple[str, ...]
    graphics: tuple[Prim, ...]
    pin_dirs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if not self.pin_dirs:
            object.__setattr__(self, "pin_dirs", tuple(_outward(self.body, p) for p in self.pins))


def _outward(body, pin) -> tuple[int, int]:
    xs = [c[0] for c in body]
    ys = [c[1] for c in body]
    if pin[0] < min(xs):
        return (-1, 0)
    if pin[0] > max(xs):
        return (1, 0)
    if pin[1] < min(ys):
        return (0, -1)
    return (0, 1)


def _rect(x0: int, x1: int, y0: int, y1: int) -> tuple[tuple[int, int], ...]:
    return tuple((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def _box(x0, y0, x1, y1, **kw) -> Poly:
    return Poly(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), closed=True, **kw)


def _arrow_head(x1, y1, x2, y2, size=0.22) -> Poly:
    """Filled triangle at (x2, y2) pointing from (x1, y1) to (x2, y2)."""
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    bx, by = x2 - ux * size, y2 - uy * size
    px, py = -uy * size * 0.55, ux * size * 0.55
    return Poly(((x2, y2), (bx + px, by + py), (bx - px, by - py)), closed=True, fill=True)


def _humps(x0: float, x1: float, y: float, up: bool = True, n: int = 4) -> list[tuple[float, float]]:
    """Inductor humps along a horizontal line."""
    pts = []
    step = (x1 - x0) / n
    r = step / 2
    for k in range(n):
        cx = x0 + k * step + r
        for i in range(9):
            t = math.pi * i / 8
            pts.append((cx - r * math.cos(t), y - (r if up else -r) * math.sin(t)))
    return pts


def _vhumps(y0: float, y1: float, x: float, right: bool, n: int = 4) -> list[tuple[float, float]]:
    pts = []
    step = (y1 - y0) / n
    r = step / 2
    for k in range(n):
        cy = y0 + k * step + r
        for i in range(9):
            t = math.pi * i / 8
            pts.append((x + (r if right else -r) * math.sin(t), cy - r * math.cos(t)))
    return pts


# --------------------------------------------------------------- two terminal
def _two(kind: int, nodes, prims) -> Symbol:
    a, b = (tuple(nodes) + ("", ""))[:2]
    return Symbol(kind=kind, body=_rect(-1, 1, 0, 0), pins=((-2, 0), (2, 0)), pin_nodes=(a, b), graphics=tuple(prims))


def resistor(nodes) -> Symbol:
    if RESISTOR_STYLE == "iec":
        prims = [Line(-2, 0, -0.9, 0), _box(-0.9, -0.3, 0.9, 0.3), Line(0.9, 0, 2, 0)]
    else:
        zig = [(-2, 0), (-1, 0)] + [(-1 + (2 * i - 1) / 6, 0.3 * (-1) ** i) for i in range(1, 7)] + [(1, 0), (2, 0)]
        prims = [Poly(tuple(zig))]
    return _two(KIND_PASSIVE, nodes, prims)


def capacitor(nodes) -> Symbol:
    prims = [
        Line(-2, 0, -0.18, 0),
        Line(-0.18, -0.6, -0.18, 0.6, width=2),
        Line(0.18, -0.6, 0.18, 0.6, width=2),
        Line(0.18, 0, 2, 0),
    ]
    return _two(KIND_PASSIVE, nodes, prims)


def inductor(nodes) -> Symbol:
    prims = [Line(-2, 0, -1, 0), Poly(tuple(_humps(-1, 1, 0))), Line(1, 0, 2, 0)]
    return _two(KIND_PASSIVE, nodes, prims)


def diode(nodes) -> Symbol:
    prims = [
        Line(-2, 0, -0.45, 0),
        Poly(((-0.45, -0.45), (-0.45, 0.45), (0.45, 0)), closed=True),
        Line(0.45, -0.45, 0.45, 0.45, width=1.5),
        Line(0.45, 0, 2, 0),
    ]
    return _two(KIND_PASSIVE, nodes, prims)


def _edge_diode(ax: float, ay: float, kx: float, ky: float, size: float = 0.3) -> list[Prim]:
    """Diode drawn on the straight line from anode (ax, ay) to cathode (kx, ky)."""
    length = math.hypot(kx - ax, ky - ay)
    ux, uy = (kx - ax) / length, (ky - ay) / length
    nx, ny = -uy, ux
    mx, my = (ax + kx) / 2, (ay + ky) / 2
    bx, by = mx - size * ux, my - size * uy  # triangle base
    tx, ty = mx + size * ux, my + size * uy  # tip and bar
    return [
        Line(ax, ay, bx, by),
        Poly(((bx - size * nx, by - size * ny), (bx + size * nx, by + size * ny), (tx, ty)), closed=True),
        Line(tx - size * nx, ty - size * ny, tx + size * nx, ty + size * ny, width=1.5),
        Line(tx, ty, kx, ky),
    ]


def bridge_rectifier(ac1: str, plus: str, ac2: str, minus: str) -> Symbol:
    """Full bridge as one diamond: AC pins left/right, + on top, - at the bottom."""
    v = 1.5  # diamond corners
    left, top, right, bottom = (-v, 0.0), (0.0, -v), (v, 0.0), (0.0, v)
    prims: list[Prim] = []
    for anode, cathode in ((left, top), (right, top), (bottom, left), (bottom, right)):
        prims += _edge_diode(*anode, *cathode)
    prims += [Line(-2, 0, -v, 0), Line(2, 0, v, 0), Line(0, -2, 0, -v), Line(0, 2, 0, v)]
    prims += [
        Text(-0.85, 0, "~", size=0.34),
        Text(0.85, 0, "~", size=0.34),
        Text(0, -0.8, "+", size=0.34),
        Text(0, 0.8, "−", size=0.34),
    ]
    return Symbol(
        kind=KIND_PASSIVE,
        body=_rect(-1, 1, -1, 1),
        pins=((-2, 0), (0, -2), (2, 0), (0, 2)),
        pin_nodes=(ac1, plus, ac2, minus),
        graphics=tuple(prims),
    )


def _source_marks(voltage: bool) -> list[Prim]:
    if voltage:
        return [Text(-0.42, 0, "+", size=0.42), Text(0.42, 0, "−", size=0.42)]
    return [Line(-0.45, 0, 0.3, 0), _arrow_head(-0.45, 0, 0.5, 0)]


def source(nodes, voltage: bool, controlled: bool) -> Symbol:
    if controlled:
        outline: Prim = Poly(((-0.8, 0), (0, -0.8), (0.8, 0), (0, 0.8)), closed=True)
    else:
        outline = Circle(0, 0, 0.8)
    prims = [Line(-2, 0, -0.8, 0), outline, Line(0.8, 0, 2, 0)] + _source_marks(voltage)
    return _two(KIND_SOURCE, nodes, prims)


def controlled_four(nodes, voltage: bool) -> Symbol:
    """E/G sources: output n+/n- left/right, control nc+/nc- below."""
    n = (tuple(nodes) + ("",) * 4)[:4]
    prims = [
        Line(-2, 0, -0.8, 0),
        Poly(((-0.8, 0), (0, -0.8), (0.8, 0), (0, 0.8)), closed=True),
        Line(0.8, 0, 2, 0),
        Line(-1, 2, -1, 1.35, dashed=True),
        Line(1, 2, 1, 1.35, dashed=True),
        Line(-1, 1.35, -0.4, 0.55, dashed=True),
        Line(1, 1.35, 0.4, 0.55, dashed=True),
        Text(-1.35, 1.55, "+", size=0.32),
        Text(1.35, 1.55, "−", size=0.32),
    ] + _source_marks(voltage)
    return Symbol(KIND_SOURCE, _rect(-1, 1, -1, 1), ((-2, 0), (2, 0), (-1, 2), (1, 2)), n, tuple(prims))


def switch(nodes) -> Symbol:
    prims = [
        Line(-2, 0, -0.6, 0),
        Circle(-0.6, 0, 0.1, fill=True),
        Line(-0.6, 0, 0.6, -0.55),
        Circle(0.6, 0, 0.1, fill=True),
        Line(0.6, 0, 2, 0),
    ]
    if len(nodes) >= 4:
        prims += [
            Line(-1, 2, -1, 1.2, dashed=True),
            Line(1, 2, 1, 1.2, dashed=True),
            Line(-1, 1.2, 1, 1.2, dashed=True),
            Line(0, 1.2, 0, -0.25, dashed=True),
        ]
        n = (tuple(nodes) + ("",) * 4)[:4]
        return Symbol(KIND_ACTIVE, _rect(-1, 1, -1, 1), ((-2, 0), (2, 0), (-1, 2), (1, 2)), n, tuple(prims))
    return _two(KIND_ACTIVE, nodes, prims)


# ------------------------------------------------------------------- active
def bjt(nodes, pnp: bool) -> Symbol:
    c, b, e = (tuple(nodes) + ("", "", ""))[:3]
    prims: list[Prim] = [
        Line(-2, 0, -0.4, 0),
        Line(-0.4, -0.7, -0.4, 0.7, width=2),
        Line(-0.4, -0.35, 1, -1.1),
        Line(1, -1.1, 1, -2),
        Line(-0.4, 0.35, 1, 1.1),
        Line(1, 1.1, 1, 2),
        Circle(0.25, 0, 1.05),
    ]
    if pnp:
        prims.append(_arrow_head(1, 1.1, -0.25, 0.43))
    else:
        prims.append(_arrow_head(-0.4, 0.35, 0.85, 1.02))
    return Symbol(KIND_ACTIVE, _rect(-1, 1, -1, 1), ((1, -2), (-2, 0), (1, 2)), (c, b, e), tuple(prims))


def mosfet(nodes, pmos: bool) -> Symbol:
    d, g, s = (tuple(nodes) + ("", "", ""))[:3]
    prims: list[Prim] = [
        Line(-2, 0, -0.6, 0),
        Line(-0.6, -0.65, -0.6, 0.65, width=1.5),
        Line(-0.3, -0.85, -0.3, -0.35, width=2),
        Line(-0.3, -0.25, -0.3, 0.25, width=2),
        Line(-0.3, 0.35, -0.3, 0.85, width=2),
        Line(-0.3, -0.6, 1, -0.6),
        Line(1, -0.6, 1, -2),
        Line(-0.3, 0.6, 1, 0.6),
        Line(1, 0.6, 1, 2),
    ]
    pins = [(1, -2), (-2, 0), (1, 2)]
    names = [d, g, s]
    if len(nodes) >= 4:
        prims.append(Line(-0.3, 0, 2, 0))
        pins.append((2, 0))
        names.append(nodes[3])
        tip = ((0.6, 0), (-0.25, 0)) if not pmos else ((-0.25, 0), (0.6, 0))
    else:
        prims.append(Line(-0.3, 0, 1, 0))
        prims.append(Line(1, 0, 1, 0.6))
        tip = ((0.6, 0), (-0.25, 0)) if not pmos else ((-0.25, 0), (0.6, 0))
    prims.append(_arrow_head(tip[0][0], tip[0][1], tip[1][0], tip[1][1]))
    return Symbol(KIND_ACTIVE, _rect(-1, 1, -1, 1), tuple(pins), tuple(names), tuple(prims))


@dataclass(frozen=True)
class Winding:
    """One coil of a transformer: nodes along the coil (more than two = taps) and
    phase dots as (segment index, at segment start)."""

    nodes: tuple[str, ...]
    dots: tuple[tuple[int, bool], ...] = ()


SEGMENT = 2  # grid cells per coil segment (between two taps)


def transformer(windings: list[Winding]) -> Symbol:
    """Transformer with any number of windings; the first is drawn left (primary),
    the others stacked on the right. Taps (e.g. a centre tap) get their own pins."""
    left, right = windings[:1], windings[1:]

    def height(ws: list[Winding]) -> int:
        return sum(SEGMENT * (len(w.nodes) - 1) for w in ws) + max(0, len(ws) - 1)

    total = max(height(left), height(right), 2)
    top = -(total // 2)
    prims: list[Prim] = []
    pins: list[tuple[int, int]] = []
    nodes: list[str] = []

    def draw(ws: list[Winding], side: int) -> None:
        y = top + (total - height(ws)) // 2
        coil_x = side * 0.6
        for w in ws:
            ys = [y + SEGMENT * i for i in range(len(w.nodes))]
            for node, py in zip(w.nodes, ys):
                pins.append((side * 2, py))
                nodes.append(node)
                prims.append(Line(side * 2, py, coil_x, py))
            for seg in range(len(w.nodes) - 1):
                prims.append(Poly(tuple(_vhumps(ys[seg], ys[seg + 1], coil_x, right=side < 0, n=3))))
            for seg, at_start in w.dots:
                dy = 0.35 if at_start else -0.35
                py = ys[seg] if at_start else ys[seg + 1]
                prims.append(Circle(side * 1.0, py + dy, 0.08, fill=True))
            y = ys[-1] + 1

    draw(left, -1)
    draw(right, 1)
    prims.append(Line(-0.08, top, -0.08, top + total, width=1.5))
    prims.append(Line(0.08, top, 0.08, top + total, width=1.5))
    return Symbol(KIND_PASSIVE, _rect(-1, 1, top, top + total), tuple(pins), tuple(nodes), tuple(prims))


# ----------------------------------------------------------------------- IC
def ic(pin_nodes: list[str], pin_names: list[str], block: bool = False) -> Symbol:
    """DIP-style box. ``pin_nodes[i]``/``pin_names[i]`` belong to package pin i+1 ('' = not connected).

    ``block``: a subcircuit of the design (can be opened), drawn as a plain rectangle
    without the pin 1 dot and package pin numbers; the pins keep the same places.
    """
    n = max(len(pin_nodes), 1)
    rows = math.ceil(n / 2)
    top = -((rows - 1) // 2)
    left = [(i, top + i) for i in range(rows)]  # pin index, row y
    right = [(rows + j, top + rows - 1 - j) for j in range(n - rows)]

    name_l = max((len(pin_names[i]) for i, _ in left), default=0)
    name_r = max((len(pin_names[i]) for i, _ in right), default=0)
    inner = (name_l + name_r) * PIN_NAME_SIZE * 0.55 + 1.0
    hw = max(1, math.ceil(inner / 2))
    edge = hw + 0.5
    bottom = top + rows - 1

    prims: list[Prim] = [_box(-edge, top - 0.5, edge, bottom + 0.5, width=1.5)]
    if not block:
        prims.append(pin1_dot(-edge, top - 0.5))
    pins: list[tuple[int, int]] = [(0, 0)] * n
    for side, items in ((-1, left), (1, right)):
        px = side * (hw + 1)
        for i, y in items:
            pins[i] = (px, y)
            prims.append(Line(px, y, side * edge, y))
            if not block:
                prims.append(Text((px + side * edge) / 2, y - 0.25, str(i + 1), size=PIN_NUMBER_SIZE))
            if pin_names[i]:
                anchor = "start" if side < 0 else "end"
                prims.append(Text(side * edge - side * 0.15, y, pin_names[i], anchor=anchor, size=PIN_NAME_SIZE, follow=True))
    nodes = tuple(pin_nodes) + ("",) * (n - len(pin_nodes))
    return Symbol(KIND_IC, _rect(-hw, hw, top, bottom), tuple(pins), nodes, tuple(prims))


def pin1_dot(left: float, top: float) -> Circle:
    """Filled dot in the top left corner of an IC box (the pin 1 mark of a package); design
    subcircuits (blocks) have none, so the two are told apart at a glance."""
    return Circle(left + 0.38, top + 0.38, 0.13, fill=True)


SIDES = ("left", "right", "top", "bottom")


def directional_ic(pin_nodes: list[str], pin_names: list[str], sides: list[str | None], block: bool = False) -> Symbol:
    """IC box with the pins grouped by function: inputs left, outputs right, supplies on
    top, grounds at the bottom (``sides[i]``; None = whichever of left / right has fewer).

    Pins keep their package order within a side; package pin numbers sit at the leads
    (not for ``block``). Top / bottom pins are spaced so their supply or ground symbols
    (net name) and pin names do not overlap.
    """
    n = len(pin_nodes)
    groups: dict[str, list[int]] = {s: [] for s in SIDES}
    for i, side in enumerate(sides):
        if side in groups:
            groups[side].append(i)
    for i, side in enumerate(sides):
        if side not in groups:
            groups["left" if len(groups["left"]) <= len(groups["right"]) else "right"].append(i)
    for side in ("left", "right"):
        groups[side].sort()

    def spacing(side: str) -> int:
        # supply names are centred above their pin; ground names start right of the symbol
        margin = 0.6 if side == "top" else 1.3
        widest = max((max(text_width(pin_names[i], PIN_NAME_SIZE), text_width(pin_nodes[i], POWER_TEXT_SIZE))
                      for i in groups[side]), default=0.0)
        return max(2, math.ceil(widest + margin))

    rows = max(len(groups["left"]), len(groups["right"]), 1)
    name_l = max((len(pin_names[i]) for i in groups["left"]), default=0)
    name_r = max((len(pin_names[i]) for i in groups["right"]), default=0)
    inner = (name_l + name_r) * PIN_NAME_SIZE * 0.55 + 1.0
    span = {s: (len(groups[s]) - 1) * spacing(s) for s in ("top", "bottom")}
    hw = max(1, math.ceil(inner / 2), math.ceil(max(span.values(), default=0) / 2) + 1)
    edge = hw + 0.5
    # one row under the top edge / above the bottom edge for the top / bottom pin names
    first = 1 if groups["top"] else 0
    top, bottom = 0, first + rows - 1 + (1 if groups["bottom"] else 0)
    shift = -((bottom - top) // 2)  # centre the box on the anchor
    top, bottom, first = top + shift, bottom + shift, first + shift

    prims: list[Prim] = [_box(-edge, top - 0.5, edge, bottom + 0.5, width=1.5)]
    if not block:
        prims.append(pin1_dot(-edge, top - 0.5))
    pins: list[tuple[int, int]] = [(0, 0)] * n
    for side, sx in (("left", -1), ("right", 1)):
        items = groups[side]
        y0 = first + (rows - len(items)) // 2
        px = sx * (hw + 1)
        for k, i in enumerate(items):
            y = y0 + k
            pins[i] = (px, y)
            if active_low(pin_names[i]):
                # inversion bubble at the body edge for active-low pins (_ni / _no)
                prims.append(Line(px, y, sx * (edge + 0.24), y))
                prims.append(Circle(sx * (edge + 0.12), y, 0.12))
            else:
                prims.append(Line(px, y, sx * edge, y))
            if not block:
                prims.append(Text((px + sx * edge) / 2, y - 0.25, str(i + 1), size=PIN_NUMBER_SIZE))
            if pin_names[i]:
                prims.append(Text(sx * edge - sx * 0.15, y, pin_names[i], anchor="start" if sx < 0 else "end", size=PIN_NAME_SIZE, follow=True))
    for side, sy in (("top", -1), ("bottom", 1)):
        items = groups[side]
        step = spacing(side)
        x0 = -span[side] // 2
        y_edge = (top - 0.5) if sy < 0 else (bottom + 0.5)
        py = top - 1 if sy < 0 else bottom + 1
        for k, i in enumerate(items):
            x = x0 + k * step
            pins[i] = (x, py)
            prims.append(Line(x, py, x, y_edge))
            if not block:
                prims.append(Text(x + 0.12, (py + y_edge) / 2, str(i + 1), anchor="start", size=PIN_NUMBER_SIZE))
            if pin_names[i]:
                prims.append(Text(x, y_edge - sy * 0.35, pin_names[i], size=PIN_NAME_SIZE, follow=True))
    return Symbol(KIND_IC, _rect(-hw, hw, top, bottom), tuple(pins), tuple(pin_nodes), tuple(prims))


# -------------------------------------------------------------------- ports
def power_port(net: str) -> Symbol:
    """Supply flag, upright: lead from below, bar on top, net name above the bar."""
    prims = [Line(0, 1, 0, 0), Line(-0.6, 0, 0.6, 0, width=2), Text(0, -0.45, net, size=LABEL_SIZE)]
    return Symbol(KIND_PORT, _rect(-1, 1, -1, 0), ((0, 1),), (net,), tuple(prims))


def signal_port(net: str) -> Symbol:
    w = text_width(net, LABEL_SIZE) + 0.6
    hw = max(1, math.ceil(w / 2))
    x0, x1 = -hw - 0.4, hw + 0.4
    prims = [
        Poly(((x0, -0.38), (x1 - 0.4, -0.38), (x1, 0), (x1 - 0.4, 0.38), (x0, 0.38)), closed=True),
        Line(x1, 0, hw + 1, 0),
        Text((x0 + x1 - 0.4) / 2, 0, net, size=LABEL_SIZE),
    ]
    return Symbol(KIND_PORT, _rect(-hw, hw, 0, 0), ((hw + 1, 0),), (net,), tuple(prims))


# --------------------------------------------------------------- annotation
GROUND_TEXT_SIZE = 0.34


def ground_glyph(px: float, py: float, dx: int, dy: int, name: str = "", name_left: bool = False) -> list[Prim]:
    """Compact ground symbol at (px, py) pointing along (dx, dy); stays within half a cell.

    ``name`` (other grounds than the main one, e.g. hvn, vee) is written beside the bars,
    to the right or with ``name_left`` to the left (away from the part a stub comes from).
    """
    qx, qy = -dy, dx
    prims: list[Prim] = [Line(px, py, px + 0.2 * dx, py + 0.2 * dy)]
    for dist, half in ((0.2, 0.3), (0.32, 0.19), (0.44, 0.08)):
        cx, cy = px + dist * dx, py + dist * dy
        prims.append(Line(cx - half * qx, cy - half * qy, cx + half * qx, cy + half * qy, width=1.3))
    if name:
        if dy:
            side = -1 if name_left else 1
            prims.append(Text(px + side * 0.36, py + 0.36 * dy, name, anchor="end" if name_left else "start", size=GROUND_TEXT_SIZE))
        else:
            prims.append(Text(px + 0.32 * dx, py + 0.5, name, size=GROUND_TEXT_SIZE))
    return prims


def power_glyph(px: float, py: float, dx: int, dy: int, net: str) -> list[Prim]:
    """Supply symbol attached to a pin at (px, py), pointing along (dx, dy): short lead,
    bar across, net name beyond the bar."""
    qx, qy = -dy, dx
    bx, by = px + 0.3 * dx, py + 0.3 * dy
    prims: list[Prim] = [
        Line(px, py, bx, by),
        Line(bx - 0.35 * qx, by - 0.35 * qy, bx + 0.35 * qx, by + 0.35 * qy, width=2),
    ]
    if dy:
        prims.append(Text(px, py + 0.62 * dy, net, size=POWER_TEXT_SIZE))
    else:
        prims.append(Text(px + 0.45 * dx, py, net, anchor="start" if dx > 0 else "end", size=POWER_TEXT_SIZE))
    return prims


def power_glyph_length(net: str, horizontal: bool) -> float:
    """How far the supply symbol reaches from its pin along the pin direction."""
    return 0.45 + text_width(net, POWER_TEXT_SIZE) if horizontal else 0.8


PASSIVE_LEAD = 0.2
PASSIVE_BODY = 1.0


def passive_glyph(kind: str, px: float, py: float, dx: int, dy: int, label: tuple[str, ...]) -> list[Prim]:
    """Two-terminal part (R, C or L) hanging on a pin at (px, py), pointing along (dx, dy).

    The far end is left open for a ground or supply glyph. Name and value sit beside the body.
    """
    qx, qy = -dy, dx
    s0 = PASSIVE_LEAD
    s1 = PASSIVE_LEAD + PASSIVE_BODY

    def at(along: float, across: float = 0.0) -> tuple[float, float]:
        return px + along * dx + across * qx, py + along * dy + across * qy

    prims: list[Prim] = [Line(px, py, *at(s0))]
    if kind == "R":
        prims += [
            Line(*at(s0), *at(s0 + 0.2)),
            Poly((at(s0 + 0.2, 0.14), at(s1 - 0.2, 0.14), at(s1 - 0.2, -0.14), at(s0 + 0.2, -0.14)), closed=True),
            Line(*at(s1 - 0.2), *at(s1)),
        ]
    elif kind == "C":
        mid = s0 + PASSIVE_BODY / 2
        prims += [
            Line(*at(s0), *at(mid - 0.06)),
            Line(*at(mid - 0.06, 0.3), *at(mid - 0.06, -0.3), width=2),
            Line(*at(mid + 0.06, 0.3), *at(mid + 0.06, -0.3), width=2),
            Line(*at(mid + 0.06), *at(s1)),
        ]
    else:
        n = 40
        wave = [at(s0 + 0.2 + t * (PASSIVE_BODY - 0.4), 0.14 * math.sin(4 * math.pi * t)) for t in (i / n for i in range(n + 1))]
        prims += [Line(*at(s0), *at(s0 + 0.2)), Poly(tuple(wave)), Line(*at(s1 - 0.2), *at(s1))]
    text_across = 0.4
    for i, line in enumerate(l for l in label if l):
        x, y = at(s0 + PASSIVE_BODY / 2, text_across + 0.22 * i)
        prims.append(Text(x, y, line, size=0.3))
    return prims


def passive_reach(supply: str | None, horizontal: bool) -> float:
    """Length of a pin-attached part plus its end (ground, or the named supply), along the pin."""
    end = power_glyph_length(supply, horizontal) if supply else 0.5
    return PASSIVE_LEAD + PASSIVE_BODY + end


POWER_TEXT_SIZE = 0.34
NET_LABEL_SIZE = 0.32
NET_LABEL_COLOR = "#b45309"


def net_label_length(net: str) -> float:
    """Length of a net label along the pin direction, in grid units (tip to flat end)."""
    return text_width(net, NET_LABEL_SIZE) + 0.75


def net_label(px: float, py: float, dx: int, dy: int, net: str) -> list[Prim]:
    """Connection-by-name label at a pin: a flag pointing at the pin with the net name.

    Horizontal pins get horizontal text, vertical pins text reading bottom-to-top.
    """
    qx, qy = -dy, dx
    length = net_label_length(net)
    half = 0.27

    def at(along: float, across: float) -> tuple[float, float]:
        return px + along * dx + across * qx, py + along * dy + across * qy

    tip, start, end = 0.15, 0.45, 0.15 + length
    shape = Poly((at(tip, 0), at(start, half), at(end, half), at(end, -half), at(start, -half)), closed=True, color=NET_LABEL_COLOR)
    cx, cy = at((start + end) / 2, 0)
    text = Text(cx, cy, net, size=NET_LABEL_SIZE, color=NET_LABEL_COLOR, angle=-90 if dy else 0)
    return [shape, text]
