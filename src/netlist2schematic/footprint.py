"""KiCad footprints of the board parts: 2D drawing and 3D render.

The footprint of a symbol comes from the netlist's ``*@parts`` header (KiCad
"Library:Name"). ``load`` reads the ``.kicad_mod`` from the KiCad library, ``graphics``
turns pads, silkscreen, fabrication and courtyard outlines into drawing primitives
(millimetres, y down; pads labelled with their number and, if known, their net).
``render_3d`` places the footprint alone on a small board and lets ``kicad-cli pcb
render`` draw it with its 3D model (KiCad library model, or a project model given by
the caller); the images are cached by content.

Paths: ``N2S_KICAD_FOOTPRINTS`` (footprint libraries) and ``N2S_KICAD_CLI`` override
the defaults of a macOS / Linux KiCad installation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .graphics import Circle, Line, Poly, Prim, Text

_FP_DIRS = (
    "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints",
    "/usr/share/kicad/footprints",
    "/usr/local/share/kicad/footprints",
)
_CLI = ("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",)

COLORS = {
    "F.SilkS": "#e5e7eb",
    "F.Fab": "#9ca3af",
    "F.CrtYd": "#d946ef",
    "pad": "#d4a017",
    "pad_tht": "#c2853d",
    "hole": "#111827",
    "pad_text": "#111827",
    "net_text": "#38bdf8",
}
LAYERS = ("F.CrtYd", "F.Fab", "F.SilkS")  # drawn in this order (silkscreen on top)
STROKE = 16  # Line width units per mm (graphics: width 1.0 = 1/16 cell)


def footprint_dir() -> Path | None:
    env = os.environ.get("N2S_KICAD_FOOTPRINTS")
    for candidate in ([env] if env else []) + list(_FP_DIRS):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def kicad_cli() -> str | None:
    env = os.environ.get("N2S_KICAD_CLI")
    for candidate in ([env] if env else []) + list(_CLI):
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("kicad-cli")


# ------------------------------------------------------------------ s-expressions
_TOKEN = re.compile(r'\s*(\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+)')


def parse_sexpr(text: str) -> list:
    """Nested lists of atoms (strings without quotes)."""
    stack: list[list] = [[]]
    for m in _TOKEN.finditer(text):
        tok = m.group(1)
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        elif tok.startswith('"'):
            stack[-1].append(tok[1:-1].replace('\\"', '"'))
        else:
            stack[-1].append(tok)
    return stack[0][0] if stack[0] else []


def _find(node: list, key: str) -> list | None:
    return next((c for c in node[1:] if isinstance(c, list) and c and c[0] == key), None)


def _all(node: list, key: str) -> list[list]:
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == key]


def _xy(node: list | None, default=(0.0, 0.0)) -> tuple[float, ...]:
    return tuple(float(v) for v in node[1:]) if node else default


# ----------------------------------------------------------------------- model
@dataclass
class Pad:
    number: str
    kind: str  # smd | thru_hole | np_thru_hole | connect
    shape: str  # rect | roundrect | circle | oval | trapezoid | custom
    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0
    drill: float = 0.0


@dataclass
class Footprint:
    lib_id: str
    text: str  # the .kicad_mod source
    pads: list[Pad] = field(default_factory=list)
    shapes: list[tuple[str, str, list]] = field(default_factory=list)  # (kind, layer, node)
    description: str = ""
    model: str = ""  # 3D model path of the library footprint ("" = none)

    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = [], []
        for p in self.pads:
            r = math.hypot(p.w, p.h) / 2
            xs += [p.x - r, p.x + r]
            ys += [p.y - r, p.y + r]
        for _, _, node in self.shapes:
            for key in ("start", "end", "center", "mid"):
                v = _find(node, key)
                if v:
                    xs.append(float(v[1]))
                    ys.append(float(v[2]))
        if not xs:
            return -1.0, -1.0, 1.0, 1.0
        return min(xs), min(ys), max(xs), max(ys)


def library_file(lib_id: str) -> Path | None:
    root = footprint_dir()
    if root is None or ":" not in lib_id:
        return None
    lib, name = lib_id.split(":", 1)
    path = root / f"{lib}.pretty" / f"{name}.kicad_mod"
    return path if path.exists() else None


def load(lib_id: str) -> Footprint | None:
    path = library_file(lib_id)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    tree = parse_sexpr(text)
    fp = Footprint(lib_id, text)
    descr = _find(tree, "descr")
    fp.description = descr[1] if descr and len(descr) > 1 else ""
    model = _find(tree, "model")
    fp.model = model[1] if model and len(model) > 1 else ""
    for pad in _all(tree, "pad"):
        at = _xy(_find(pad, "at"))
        size = _xy(_find(pad, "size"), (0.0, 0.0))
        drill = _find(pad, "drill")
        diameter = next((float(v) for v in (drill or [])[1:] if not isinstance(v, list) and _is_number(v)), 0.0)
        fp.pads.append(
            Pad(
                number=str(pad[1]),
                kind=str(pad[2]),
                shape=str(pad[3]),
                x=at[0],
                y=at[1],
                w=size[0],
                h=size[1] if len(size) > 1 else size[0],
                angle=at[2] if len(at) > 2 else 0.0,
                drill=diameter,
            )
        )
    for kind in ("fp_line", "fp_rect", "fp_circle", "fp_arc", "fp_poly"):
        for node in _all(tree, kind):
            layer = _find(node, "layer")
            if layer and layer[1] in LAYERS:
                fp.shapes.append((kind, layer[1], node))
    return fp


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


# -------------------------------------------------------------------- drawing
def _width(node: list) -> float:
    stroke = _find(node, "stroke")
    w = _find(stroke, "width") if stroke else _find(node, "width")
    return float(w[1]) if w else 0.12


def _rotated(points, cx: float, cy: float, angle: float):
    a = math.radians(-angle)  # KiCad: positive angle = counter-clockwise on screen
    c, s = math.cos(a), math.sin(a)
    return tuple((cx + x * c - y * s, cy + x * s + y * c) for x, y in points)


def pad_outline(p: Pad) -> tuple[tuple[float, float], ...]:
    hw, hh = p.w / 2, p.h / 2
    if p.shape in ("circle", "oval"):
        r = min(hw, hh)
        pts = []
        # stadium: two half circles joined by straight sides
        if hw >= hh:
            for cx, start in ((hw - r, -90), (-(hw - r), 90)):
                pts += [(cx + r * math.cos(math.radians(start + k * 15)), r * math.sin(math.radians(start + k * 15))) for k in range(13)]
        else:
            for cy, start in ((hh - r, 0), (-(hh - r), 180)):
                pts += [(r * math.cos(math.radians(start + k * 15)), cy + r * math.sin(math.radians(start + k * 15))) for k in range(13)]
        return _rotated(pts, p.x, p.y, p.angle)
    if p.shape == "roundrect":
        r = min(hw, hh) * 0.25
        pts = []
        for cx, cy, start in ((hw - r, hh - r, 0), (-(hw - r), hh - r, 90), (-(hw - r), -(hh - r), 180), (hw - r, -(hh - r), 270)):
            pts += [(cx + r * math.cos(math.radians(start + k * 30)), cy + r * math.sin(math.radians(start + k * 30))) for k in range(4)]
        return _rotated(pts, p.x, p.y, p.angle)
    return _rotated(((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)), p.x, p.y, p.angle)


def _arc_points(start, mid, end, steps: int = 24):
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return [start, end]
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(y - uy, x - ux) for x, y in (start, mid, end))

    def between(a, b, c):  # is b on the ccw way from a to c?
        return (b - a) % (2 * math.pi) <= (c - a) % (2 * math.pi)

    span = (a3 - a1) % (2 * math.pi)
    if not between(a1, a2, a3):
        span -= 2 * math.pi
    return [(ux + r * math.cos(a1 + span * k / steps), uy + r * math.sin(a1 + span * k / steps)) for k in range(steps + 1)]


def graphics(fp: Footprint, nets: dict[str, str] | None = None) -> list[Prim]:
    """Primitives in millimetres: outlines (courtyard, fab, silkscreen), pads with their
    number and the net of the pad (``nets``: pad number -> net name)."""
    nets = nets or {}
    prims: list[Prim] = []
    for layer in LAYERS:
        color = COLORS[layer]
        for kind, lay, node in fp.shapes:
            if lay != layer:
                continue
            w = _width(node) * STROKE
            fill = _find(node, "fill")
            filled = bool(fill and fill[1] in ("yes", "solid"))
            if kind == "fp_line":
                (x1, y1), (x2, y2) = _xy(_find(node, "start")), _xy(_find(node, "end"))
                prims.append(Line(x1, y1, x2, y2, width=w, color=color))
            elif kind == "fp_rect":
                (x1, y1), (x2, y2) = _xy(_find(node, "start")), _xy(_find(node, "end"))
                prims.append(Poly(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), closed=True, fill=filled, width=w, color=color))
            elif kind == "fp_circle":
                (cx, cy), (ex, ey) = _xy(_find(node, "center")), _xy(_find(node, "end"))
                prims.append(Circle(cx, cy, math.hypot(ex - cx, ey - cy), fill=filled, width=w, color=color))
            elif kind == "fp_arc":
                pts = _arc_points(_xy(_find(node, "start")), _xy(_find(node, "mid")), _xy(_find(node, "end")))
                prims.append(Poly(tuple(pts), width=w, color=color))
            elif kind == "fp_poly":
                pts = tuple(_xy(xy) for xy in _all(_find(node, "pts") or ["pts"], "xy"))
                if pts:
                    prims.append(Poly(pts, closed=True, fill=filled, width=w, color=color))
    for p in fp.pads:
        if p.kind == "np_thru_hole":
            prims.append(Circle(p.x, p.y, max(p.drill, p.w) / 2, width=0.1 * STROKE, color=COLORS["F.Fab"]))
            continue
        color = COLORS["pad_tht"] if p.kind == "thru_hole" else COLORS["pad"]
        prims.append(Poly(pad_outline(p), closed=True, fill=True, width=0, color=color))
        if p.drill:
            prims.append(Circle(p.x, p.y, p.drill / 2, fill=True, width=0, color=COLORS["hole"]))
        size = max(0.25, min(p.w, p.h) * 0.55)
        if p.number:
            prims.append(Text(p.x, p.y, p.number, size=size, color=COLORS["pad_text"] if not p.drill else COLORS["F.SilkS"]))
        net = nets.get(p.number)
        if net:
            prims.append(Text(p.x, p.y + min(p.w, p.h) / 2 + 0.25, net, size=max(0.25, size * 0.6), color=COLORS["net_text"]))
    return prims


# ----------------------------------------------------------------------- 3D
VIEWS = {  # name -> kicad-cli --rotate "x,y,z" (degrees)
    "Iso": "-45,0,45",
    "Oben": "0,0,0",
    "Vorne": "-90,0,0",
    "Seite": "-90,0,90",
    "Unten": "180,0,0",
}
MARGIN = 1.5  # mm of board around the footprint


def board_text(fp: Footprint, ref: str = "", model: Path | None = None) -> str:
    """A minimal board holding only this footprint (centred), for kicad-cli render.
    ``model`` replaces (or adds) the 3D model, e.g. a project model for a footprint
    without a library model."""
    body = fp.text.replace('(footprint "', f'(footprint "{fp.lib_id.split(":", 1)[0]}:', 1)
    body = body.replace('(layer "F.Cu")', '(layer "F.Cu")\n\t(at 0 0)', 1)
    if ref:
        body = body.replace('"REF**"', f'"{ref}"')
    if model is not None:
        block = f'(model "{model}" (offset (xyz 0 0 0)) (scale (xyz 1 1 1)) (rotate (xyz 0 0 0)))'
        if "(model " in body:
            body = re.sub(r'\(model "[^"]*"', f'(model "{model}"', body, count=1)
        else:
            body = body.rstrip()[:-1] + block + "\n)"
    x0, y0, x1, y1 = fp.bbox()
    x0, y0, x1, y1 = x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN
    layers = (
        '(0 "F.Cu" signal) (2 "B.Cu" signal) (9 "F.Adhes" user "F.Adhesive") (11 "B.Adhes" user "B.Adhesive") '
        '(13 "F.Paste" user) (15 "B.Paste" user) (5 "F.SilkS" user "F.Silkscreen") (7 "B.SilkS" user "B.Silkscreen") '
        '(1 "F.Mask" user) (3 "B.Mask" user) (25 "Edge.Cuts" user) (31 "F.CrtYd" user "F.Courtyard") '
        '(29 "B.CrtYd" user "B.Courtyard") (35 "F.Fab" user) (33 "B.Fab" user)'
    )
    return (
        f'(kicad_pcb (version 20241229) (generator "netlist2schematic") (generator_version "9.0")\n'
        f" (general (thickness 1.6) (legacy_teardrops no))\n (paper \"A4\")\n (layers {layers})\n"
        f' (setup (pad_to_mask_clearance 0))\n (net 0 "")\n {body}\n'
        f" (gr_rect (start {x0:.3f} {y0:.3f}) (end {x1:.3f} {y1:.3f}) (stroke (width 0.1) (type default)) "
        f'(fill no) (layer "Edge.Cuts"))\n)\n'
    )


def render_3d(fp: Footprint, cache_dir: Path, view: str = "Iso", ref: str = "", model: Path | None = None,
              size: tuple[int, int] = (640, 480), zoom: float = 0.6) -> Path | None:
    """PNG of the footprint with its 3D model (cached); None without kicad-cli. ``view`` is a
    name of VIEWS or an "x,y,z" rotation; ``zoom`` < 1 leaves room for tall parts."""
    cli = kicad_cli()
    if cli is None:
        return None
    board = board_text(fp, ref, model)
    key = hashlib.sha1(f"{board}|{view}|{size}|{zoom}".encode()).hexdigest()[:16]
    out = cache_dir / f"{key}.png"
    if out.exists():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pcb = Path(tmp) / "part.kicad_pcb"
        pcb.write_text(board)
        cmd = [cli, "pcb", "render", "--output", str(out), "--width", str(size[0]), "--height", str(size[1]),
               "--rotate", VIEWS.get(view, view), "--zoom", f"{zoom:g}", "--quality", "basic",
               "--background", "transparent", str(pcb)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"kicad-cli render failed: {result.stderr.strip() or result.stdout.strip()}")
    return out


def has_3d_model(fp: Footprint, model: Path | None = None) -> bool:
    """A library model whose file exists (any of .step / .wrl), or a project model."""
    if model is not None:
        return model.exists()
    if not fp.model:
        return False
    root = footprint_dir()
    if root is None:
        return True  # cannot check: let kicad-cli try
    models = root.parent / "3dmodels"
    rel = re.sub(r"^\$\{[^}]+\}/", "", fp.model)
    path = models / rel
    return any(path.with_suffix(s).exists() for s in (".step", ".stp", ".wrl"))
