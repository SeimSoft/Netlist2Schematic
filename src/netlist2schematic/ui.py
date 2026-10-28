"""Qt (PySide6) schematic editor: proposal, manual correction, training data.

Workflow: open a netlist, a ``sizing.py`` or a ``.jinja2`` template; the policy
proposes a placement (or the approved one from ``schematic.json`` is loaded);
move, rotate and mirror parts until it reads well; "Send training data" stores
the placement in the training database and as the approved schematic.

View (dark, Virtuoso-like): the sheet is unbounded. Mouse wheel / pinch /
Ctrl+scroll zoom at the cursor; trackpad scroll, right or middle mouse drag, or
Space+drag pan; F fits the drawing. Drag on empty space selects several parts;
dragging a selected part moves the whole selection. R rotates, M mirrors.
Click a wire or net label to select its net; 9 highlights the selected nets in
a random colour, 0 clears all highlights. The dock shows the properties of the
selected part or net.

History: the slider replays the decisions one at a time: placements, moves of
placed parts by the policy (revisions) and manual corrections; it shows the
policy's probability map for each of its decisions. The sheet always shows the
replayed state. Every manual move, rotation or mirror is recorded as a step and
discards the steps after it (nothing is recomputed). At the end of the history,
▶ asks the policy for exactly one more step; "Rest vorschlagen" completes the
drawing from the slider position.

Window: the toolbar holds the actions as icon buttons (name and shortcut in the
tooltip), the netlist and the properties are docks. Every sheet is a tab; a
double click on a subcircuit block (a circuit of the design, drawn as a plain
rectangle; vendor models stay IC boxes) opens it in its own tab. The hierarchy
dock (right) shows the cell tree of the opened source; selecting an element there
shows the sheet of its parent subcircuit (opening a tab if needed) and selects
and centres the element on it.

Training data: "Gewichte löschen" removes the network weights (the next proposal
uses only the prior); "Trainingsdaten…" lists the stored examples, select rows and
delete them via the context menu.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import qtawesome as qta
from PySide6.QtCore import QEvent, QPointF, QRectF, QRegularExpression, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import footprint as FP
from . import paths, policy, trainer
from .api import HierarchyNode, default_circuit, hierarchy, saved_placement, top_subckts
from .diagram import Diagram, build_diagram, own_symbol
from .env import placement_order
from .graphics import ACCENT, INK, Circle, Line, Poly, Prim, Text, bbox, text_extent
from .metrics import evaluate
from .netlist import Netlist, is_ground, parse_netlist
from .placement import Placement, group_conflicts, group_fits, nearest_fit
from .frame import sheet_frame, symbol_prims
from .render import NET_NAME_COLOR, GroupBox, component_graphics, fly_lines, group_boxes, net_graphics, net_islands, schematic_graphics, schematic_svg
from .routing import route
from .sources import Source, load_source, save_placement
from .symbols import NET_LABEL_COLOR
from .trainer import HistoryStep
from .training_db import TrainingDB

CELL = 24.0  # scene pixels per grid cell
SCENE_HALF = 2_000_000.0  # scene extent in pixels: practically unbounded

# Dark theme: drawing colours (used by the SVG export) mapped to screen colours.
BACKGROUND = "#000000"
GRID_MINOR = "#141814"
GRID_MAJOR = "#243024"
THEME = {
    INK: "#4ade80",  # symbols and wires
    NET_NAME_COLOR: "#38bdf8",
    NET_LABEL_COLOR: "#fbbf24",
    ACCENT: "#ef4444",
}
SELECTED = "#ffffff"
RELATED = "#d1fae5"  # nets of the selected part(s)
CURRENT = "#facc15"  # part placed in the step shown by the history slider

KIND_NAMES = {
    "R": "Widerstand",
    "C": "Kondensator",
    "L": "Spule",
    "D": "Diode",
    "V": "Spannungsquelle",
    "I": "Stromquelle",
    "B": "Verhaltensquelle",
    "E": "Spannungsgesteuerte Spannungsquelle",
    "G": "Spannungsgesteuerte Stromquelle",
    "F": "Stromgesteuerte Stromquelle",
    "H": "Stromgesteuerte Spannungsquelle",
    "Q": "Bipolartransistor",
    "M": "MOSFET",
    "S": "Schalter",
    "W": "Schalter",
    "X": "Subcircuit / IC",
    "N": "Python-Modell",
    "K": "Übertrager",
    "P": "Port",
}


def _pt(x: float, y: float) -> QPointF:
    return QPointF((x + 0.5) * CELL, (y + 0.5) * CELL)


def draw_prims(painter: QPainter, prims: list[Prim], color: str | None = None, width: float = 1.0) -> None:
    """Draw primitives in grid coordinates (cell centre = integer point) in the dark theme."""
    for p in prims:
        c = QColor(color or THEME.get(p.color, p.color))
        if isinstance(p, Text):
            font = QFont("Helvetica")
            font.setPixelSize(max(1, round(p.size * CELL)))
            painter.setFont(font)
            painter.setPen(QPen(c))
            metrics = QFontMetricsF(font)
            text_w = metrics.horizontalAdvance(p.text)
            dx = {"start": 0.0, "middle": -text_w / 2, "end": -text_w}[p.anchor]
            painter.save()
            painter.translate(_pt(p.x, p.y))
            if p.angle:
                painter.rotate(p.angle)
            painter.drawText(QPointF(dx, (metrics.ascent() - metrics.descent()) / 2), p.text)
            painter.restore()
            continue
        pen = QPen(c, p.width * width * CELL / 16)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if getattr(p, "dashed", False):
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen if p.width > 0 else Qt.PenStyle.NoPen)
        fill = QColor(c)
        if isinstance(p, Poly) and p.opacity < 1:
            fill.setAlphaF(p.opacity)
        painter.setBrush(QBrush(fill) if getattr(p, "fill", False) else Qt.BrushStyle.NoBrush)
        if isinstance(p, Line):
            painter.drawLine(_pt(p.x1, p.y1), _pt(p.x2, p.y2))
        elif isinstance(p, Poly):
            path = QPainterPath(_pt(*p.points[0]))
            for x, y in p.points[1:]:
                path.lineTo(_pt(x, y))
            if p.closed:
                path.closeSubpath()
            painter.drawPath(path)
        elif isinstance(p, Circle):
            painter.drawEllipse(_pt(p.x, p.y), p.r * CELL, p.r * CELL)


def _rect_of(prims: list[Prim]) -> QRectF:
    if not prims:
        return QRectF()
    x0, y0, x1, y1 = bbox(prims)
    return QRectF((x0 + 0.5) * CELL - 4, (y0 + 0.5) * CELL - 4, (x1 - x0) * CELL + 8, (y1 - y0) * CELL + 8)


class ComponentItem(QGraphicsItem):
    """A placed part. Its position is the anchor cell; it snaps to the grid."""

    def __init__(self, sheet: "SheetTab", ci: int, rotation: int):
        super().__init__()
        self.sheet = sheet
        self.ci = ci
        self.component = sheet.diagram.components[ci]
        self.rotation_steps = rotation
        self.invalid = False
        self.current = False
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        tip = self.component.title
        if self.component.subckt:
            tip += "\nDoppelklick: Subcircuit in neuem Tab öffnen"
        self.setToolTip(tip)
        self.setZValue(2)
        self._update_geometry()

    def _update_geometry(self) -> None:
        self.prepareGeometryChange()
        # Net labels are drawn by their net so they can be selected and highlighted.
        self.prims = component_graphics(self.component, self.rotation_steps, net_labels=False)
        self._rect = _rect_of(self.prims)
        blocked, pins = self.component.relative_cells(self.rotation_steps)
        self.cells = blocked + pins
        xs = [c[0] for c in blocked + pins]
        ys = [c[1] for c in blocked + pins]
        self._hit = QRectF(min(xs) * CELL, min(ys) * CELL, (max(xs) - min(xs) + 1) * CELL, (max(ys) - min(ys) + 1) * CELL)

    def set_rotation(self, rotation: int) -> None:
        self.rotation_steps = rotation
        self._update_geometry()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect.united(self._hit)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self._hit)
        return path

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isSelected():
            # Reserved area (body, labels, symbols): nothing else may be placed there.
            painter.setPen(QPen(QColor(255, 255, 255, 90), 0, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 255, 255, 25))
            for cx, cy in self.cells:
                painter.drawRect(QRectF(cx * CELL, cy * CELL, CELL, CELL))
        color = THEME[ACCENT] if self.invalid else (SELECTED if self.isSelected() else (CURRENT if self.current else None))
        draw_prims(painter, self.prims, color)

    def cell(self) -> tuple[int, int]:
        return round(self.pos().x() / CELL), round(self.pos().y() / CELL)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return QPointF(round(value.x() / CELL) * CELL, round(value.y() / CELL) * CELL)
        return super().itemChange(change, value)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.sheet.preview_move()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.sheet.drop_selection()

    def mouseDoubleClickEvent(self, event):
        if self.component.subckt:
            self.sheet.win.open_subcircuit(self.sheet.source, self.sheet.netlist, self.component.subckt)
            return
        super().mouseDoubleClickEvent(event)


class NetItem(QGraphicsItem):
    """One net: wires, junctions, name and pin labels. Selectable and highlightable."""

    def __init__(self, sheet: "SheetTab", net: str, prims: list[Prim]):
        super().__init__()
        self.sheet = sheet
        self.net = net
        self.prims = prims
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setToolTip(net)
        self.setZValue(2.5)  # above parts, but only the thin net geometry is clickable
        self._rect = _rect_of(prims)
        raw = QPainterPath()
        for p in prims:
            if isinstance(p, Line):
                raw.moveTo(_pt(p.x1, p.y1))
                raw.lineTo(_pt(p.x2, p.y2))
        stroker = QPainterPathStroker()
        stroker.setWidth(CELL * 0.35)
        self._shape = stroker.createStroke(raw)
        for p in prims:
            if isinstance(p, Poly) and p.closed:
                poly = QPainterPath(_pt(*p.points[0]))
                for x, y in p.points[1:]:
                    poly.lineTo(_pt(x, y))
                poly.closeSubpath()
                self._shape.addPath(poly)

    def boundingRect(self) -> QRectF:
        return self._rect

    def shape(self) -> QPainterPath:
        return self._shape

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        highlight = self.sheet.highlights.get(self.net)
        if self.isSelected():
            draw_prims(painter, self.prims, SELECTED, width=2.0)
        elif highlight:
            draw_prims(painter, self.prims, highlight, width=2.2)
        elif self.net in self.sheet.related_nets:
            draw_prims(painter, self.prims, RELATED, width=1.7)
        else:
            draw_prims(painter, self.prims)


class OwnSymbolItem(QGraphicsItem):
    """The circuit's own block symbol in the bottom left corner of the frame.

    Clicking a pin (or its name) selects that port's net exactly like clicking one of its
    wires (Shift / Cmd adds to the selection); the pin is drawn white while its net is
    selected, in the highlight colour while it is highlighted.
    """

    PICK = 1.4  # cells: how close to a pin a click must be

    def __init__(self, sheet: "SheetTab", symbol, origin: tuple[int, int], circuit: str):
        super().__init__()
        self.sheet = sheet
        self.symbol = symbol
        self.prims = symbol_prims(symbol, (0, 0), circuit)
        self._rect = _rect_of(self.prims)
        self.origin = origin
        self.setPos(origin[0] * CELL, origin[1] * CELL)
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{circuit}: Pin anklicken wählt das Netz des Ports")

    def boundingRect(self) -> QRectF:
        return self._rect

    def _lead(self, pi: int) -> Line:
        """The pin's lead: from the pin to the box edge (one cell inwards)."""
        (px, py), (dx, dy) = self.symbol.pins[pi], self.symbol.pin_dirs[pi]
        return Line(px, py, px - dx, py - dy, width=2.4)

    def pin_at(self, pos: QPointF) -> int | None:
        """Index of the pin nearest to a local position (within PICK cells), if any."""
        x, y = pos.x() / CELL - 0.5, pos.y() / CELL - 0.5  # inverse of _pt
        best, best_d = None, self.PICK
        for pi, ((px, py), (dx, dy)) in enumerate(zip(self.symbol.pins, self.symbol.pin_dirs)):
            # Distance to the lead and the pin name just inside the box.
            ex, ey = px - 2 * dx, py - 2 * dy
            t = max(0.0, min(1.0, ((x - px) * (ex - px) + (y - py) * (ey - py)) / max((ex - px) ** 2 + (ey - py) ** 2, 1e-9)))
            d = math.hypot(x - (px + t * (ex - px)), y - (py + t * (ey - py)))
            if d < best_d:
                best, best_d = pi, d
        return best

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_prims(painter, self.prims)
        selected = set(self.sheet._selected_nets())
        for pi, net in enumerate(self.symbol.pin_nodes):
            color = SELECTED if net in selected else self.sheet.highlights.get(net)
            if color:
                draw_prims(painter, [self._lead(pi)], color)

    def pin_point(self, net: str) -> tuple[float, float] | None:
        """Grid position of the pin of ``net`` (for fly lines)."""
        if net not in self.symbol.pin_nodes:
            return None
        px, py = self.symbol.pins[list(self.symbol.pin_nodes).index(net)]
        return float(self.origin[0] + px), float(self.origin[1] + py)

    def hoverMoveEvent(self, event):
        pi = self.pin_at(event.pos())
        if pi is None:
            self.unsetCursor()
            self.setToolTip("")
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(f"Port {self.symbol.pin_nodes[pi]}: klicken wählt das Netz")

    def mousePressEvent(self, event):
        pi = self.pin_at(event.pos())
        if pi is None or event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        additive = bool(event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        self.sheet.select_net(self.symbol.pin_nodes[pi], additive)
        event.accept()


FLY_COLOR = QColor(255, 255, 255, 110)  # fly lines of selected nets (semi-transparent)


class FlyLineItem(QGraphicsItem):
    """Curved, semi-transparent lines joining the separate places of a selected net (labels
    at pins, port labels, the own symbol's pin), like Virtuoso's flight lines. Drawn with
    a cosmetic pen, so they stay visible however far the view is zoomed out."""

    BOW = 0.22  # control point offset relative to the line length

    def __init__(self, lines: list[tuple[tuple[float, float], tuple[float, float]]], color: QColor):
        super().__init__()
        self.color = color
        self.path = QPainterPath()
        for (ax, ay), (bx, by) in lines:
            a, b = _pt(ax, ay), _pt(bx, by)
            mid = (a + b) / 2
            d = b - a
            length = math.hypot(d.x(), d.y()) or 1.0
            normal = QPointF(-d.y() / length, d.x() / length)
            self.path.moveTo(a)
            self.path.quadTo(mid + normal * (self.BOW * length), b)
        self._rect = self.path.boundingRect().adjusted(-4, -4, 4, 4)
        self.setZValue(3)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.color, 2.0)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path)


class FrameItem(QGraphicsItem):
    """The sheet frame (grid, title block, port list). Behind the parts, never selectable."""

    def __init__(self, prims: list[Prim]):
        super().__init__()
        self.prims = prims
        self._rect = _rect_of(prims)
        self.setZValue(-1)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None):
        draw_prims(painter, self.prims)


class GroupItem(QGraphicsItem):
    """A functional group: coloured background behind its parts, outline and title.

    Clicking the title or the outline selects the group (its description shows in the
    properties dock); the inside stays free for rubber-band selection of the parts."""

    def __init__(self, sheet: "SheetTab", box: GroupBox):
        super().__init__()
        self.sheet = sheet
        self.box = box
        self.gid = box.gid
        self.prims = list(box.prims)
        self._rect = _rect_of(self.prims)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(0.2)  # above the frame, below heat map and parts
        self.setToolTip(f"{box.title}\n{box.description}" if box.description else box.title)
        x0, y0, x1, y1 = box.rect
        outline = QPainterPath()
        outline.addRect(QRectF(_pt(x0, y0), _pt(x1, y1)))
        stroker = QPainterPathStroker()
        stroker.setWidth(CELL * 0.5)
        self._shape = stroker.createStroke(outline)
        title = next(p for p in self.prims if isinstance(p, Text))
        (tx0, tx1), (ty0, ty1) = text_extent(title)
        self._shape.addRect(QRectF(_pt(tx0, ty0), _pt(tx1, ty1)).adjusted(-4, -4, 4, 4))

    def boundingRect(self) -> QRectF:
        return self._rect

    def shape(self) -> QPainterPath:
        return self._shape

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_prims(painter, self.prims)
        if self.isSelected():
            draw_prims(painter, self.prims[1:], SELECTED, width=1.6)


class HeatItem(QGraphicsItem):
    """Policy probability per anchor cell for the step shown by the history slider."""

    def __init__(self):
        super().__init__()
        self.heat = None
        self.offset = (0, 0)
        self._rect = QRectF()
        self.setZValue(0.5)

    def set_heat(self, heat, offset: tuple[int, int]) -> None:
        self.prepareGeometryChange()
        self.heat = heat
        self.offset = offset
        if heat is None:
            self._rect = QRectF()
        else:
            w, h = heat.shape
            self._rect = QRectF(-offset[0] * CELL, -offset[1] * CELL, w * CELL, h * CELL)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None):
        if self.heat is None:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        ox, oy = self.offset
        for (gx, gy), value in np.ndenumerate(self.heat):
            if value < 0.02:
                continue
            painter.setBrush(QColor(249, 115, 22, int(25 + 150 * value)))
            painter.drawRect(QRectF((gx - ox) * CELL, (gy - oy) * CELL, CELL, CELL))


class SchematicScene(QGraphicsScene):
    """Unbounded black sheet: the grid is drawn wherever the view looks."""

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor(BACKGROUND))
        pixels = painter.worldTransform().m11() * CELL
        step = 1 if pixels >= 10 else 5 if pixels >= 2 else 25
        minor = QPen(QColor(GRID_MINOR), 0)
        major = QPen(QColor(GRID_MAJOR), 0)
        x0 = int(rect.left() // CELL) // step * step
        y0 = int(rect.top() // CELL) // step * step
        for x in range(x0, int(rect.right() // CELL) + 1, step):
            painter.setPen(major if x % (5 * step) == 0 else minor)
            painter.drawLine(QPointF((x + 0.5) * CELL, rect.top()), QPointF((x + 0.5) * CELL, rect.bottom()))
        for y in range(y0, int(rect.bottom() // CELL) + 1, step):
            painter.setPen(major if y % (5 * step) == 0 else minor)
            painter.drawLine(QPointF(rect.left(), (y + 0.5) * CELL), QPointF(rect.right(), (y + 0.5) * CELL))


class SchematicView(QGraphicsView):
    """Zoom at the cursor (wheel, pinch, Ctrl/Cmd+scroll); pan with trackpad scroll,
    right or middle mouse drag, or Space+drag.

    The view keeps its centre as an exact scene point (``_centre``) and computes zoom
    and pan from it. Qt's scroll bars only position the content in whole pixels and are
    set from ``_centre`` after every change, so that rounding error never accumulates.
    (The scroll bars must stay enabled: with them switched off Qt cannot move the view.)
    """

    MIN_SCALE, MAX_SCALE = 0.08, 8.0
    GESTURE_PAUSE = 0.4  # seconds without zoom steps after which a new zoom gesture starts

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setBackgroundBrush(QColor(BACKGROUND))
        self._centre = QPointF()  # scene point at the centre of the viewport (exact)
        self._syncing = False  # True while we move the scroll bars ourselves
        self._scroll = (0, 0)  # scroll positions we set last
        self._last_zoom = 0.0  # when the last zoom step happened (a gesture ends after a pause)
        self._pan_from: QPointF | None = None
        self._space = False
        self.empty_click = None  # callback for a left click on empty space

    # ------------------------------------------------------------ viewport
    def _scale(self) -> float:
        return self.transform().m11()

    def _view_centre(self) -> QPointF:
        return QPointF(self.viewport().rect().center())

    def _apply(self, scale: float, centre: QPointF) -> None:
        """Show ``centre`` (scene point) at the viewport centre with the given scale."""
        self._centre = QPointF(centre)
        self._syncing = True
        ratio = scale / self._scale()
        self.scale(ratio, ratio)  # relative: resetTransform() left the view's centring stale
        self.centerOn(centre)
        self._scroll = self._scroll_values()
        self._syncing = False

    def _scroll_values(self) -> tuple[int, int]:
        return self.horizontalScrollBar().value(), self.verticalScrollBar().value()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        # Only a scroll that Qt made itself (e.g. auto-scroll while selecting) changes the
        # centre; re-reading it after our own whole-pixel positioning would add up errors.
        if not self._syncing and self._scroll_values() != self._scroll:
            self._centre = self.mapToScene(self._view_centre().toPoint())
            self._scroll = self._scroll_values()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply(self._scale(), self._centre)

    # ---------------------------------------------------------- zoom and pan
    def zoom_at(self, factor: float, pos: QPointF) -> None:
        """Zoom by ``factor`` keeping the scene point under viewport position ``pos`` fixed."""
        scale = self._scale()
        factor = max(self.MIN_SCALE / scale, min(self.MAX_SCALE / scale, factor))
        if factor == 1.0:
            return
        now = time.monotonic()
        if now - self._last_zoom > self.GESTURE_PAUSE or self._scroll_values() != self._scroll:
            # The view was moved by something else (fit, resize, a pan, or Qt catching up with
            # its scroll ranges): start from the centre it really shows.
            self._centre = self.mapToScene(self._view_centre().toPoint())
        self._last_zoom = now
        offset = (pos - self._view_centre()) * (1.0 / scale)  # scene offset of the pointer from the centre
        anchor = self._centre + offset
        new_scale = scale * factor
        self._apply(new_scale, anchor - offset * (scale / new_scale))

    def pan(self, dx: float, dy: float) -> None:
        """Move the content by ``dx``/``dy`` viewport pixels (content follows the pointer)."""
        scale = self._scale()
        self._apply(scale, self._centre - QPointF(dx, dy) * (1.0 / scale))

    def fit(self) -> None:
        rect = QRectF()
        for item in self.scene().items():
            if isinstance(item, (ComponentItem, FrameItem)) and item.isVisible():
                rect = rect.united(item.sceneBoundingRect())
        if rect.isEmpty():
            rect = QRectF(-10 * CELL, -10 * CELL, 20 * CELL, 20 * CELL)
        rect = rect.adjusted(-CELL, -CELL, CELL, CELL)
        view = self.viewport().rect()
        scale = min(view.width() / rect.width(), view.height() / rect.height())
        self._apply(scale, rect.center())

    # ----------------------------------------------------------------- input
    def wheelEvent(self, event):
        # A mouse wheel has no scroll phase (macOS still reports a pixel delta for it);
        # a trackpad two-finger scroll does, and pans. Ctrl/Cmd+scroll always zooms.
        trackpad = event.phase() != Qt.ScrollPhase.NoScrollPhase
        zoom_mod = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        if trackpad and not zoom_mod:
            delta = event.pixelDelta()
            self.pan(delta.x(), delta.y())
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom_at(1.15**steps, event.position())

    def viewportEvent(self, event):
        if event.type() == QEvent.Type.NativeGesture and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom_at(1.0 + event.value(), QPointF(self.viewport().mapFromGlobal(QCursor.pos())))
            return True
        return super().viewportEvent(event)

    def mousePressEvent(self, event):
        pan_button = event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton)
        if pan_button or (self._space and event.button() == Qt.MouseButton.LeftButton):
            self._pan_from = event.position()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        super().mousePressEvent(event)
        additive = event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        if event.button() == Qt.MouseButton.LeftButton and not additive and self.itemAt(event.position().toPoint()) is None:
            if self.empty_click is not None:
                self.empty_click()

    def mouseMoveEvent(self, event):
        if self._pan_from is not None:
            delta = event.position() - self._pan_from
            self._pan_from = event.position()
            self.pan(delta.x(), delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan_from is not None:
            self._pan_from = None
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().unsetCursor()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.viewport().unsetCursor()
            return
        super().keyReleaseEvent(event)


class SheetTab(QWidget):
    """One schematic sheet (a tab): a subcircuit with its own drawing, history and proposal."""

    def __init__(self, win: "MainWindow", source: Source, netlist: Netlist):
        super().__init__()
        self.win = win
        self.source = source
        self.netlist = netlist
        self.diagram: Diagram | None = None
        self.placement: Placement = {}
        self.items: dict[int, ComponentItem] = {}
        self.net_items: dict[str, NetItem] = {}
        self.highlights: dict[str, str] = {}
        self.group_items: dict[str, GroupItem] = {}
        self.pending_focus: str | None = None  # element to select once it is placed (proposal still building)
        self.routing = None
        self.frame_item: FrameItem | None = None
        self.symbol_item: OwnSymbolItem | None = None
        self.fly_items: list[FlyLineItem] = []
        # Nets selected without a wire to select (supply / ground, via the own symbol) and nets
        # of pads selected in the footprint dock: both get fly lines like selected nets.
        self.extra_nets: list[str] = []
        self.pad_nets: list[str] = []
        self.related_nets: set[str] = set()  # every net of the selected parts (fly lines, emphasis)
        self._selecting = False
        self.history: list[HistoryStep] = []
        self.rollout: trainer.Rollout | None = None  # live policy state for step-wise proposals
        # Proposals are built one step per event-loop turn, so the circuit grows on screen.
        self.building = False
        self.build_timer = QTimer(self)
        self.build_timer.setSingleShot(True)
        self.build_timer.timeout.connect(self._build_step)
        self.route_cost = 0.0  # seconds the last routing took
        self.routed_at = 0.0  # when it finished

        self.scene = SchematicScene(self)
        self.scene.setSceneRect(-SCENE_HALF, -SCENE_HALF, 2 * SCENE_HALF, 2 * SCENE_HALF)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.view = SchematicView(self.scene)
        self.view.empty_click = self.clear_extra_nets
        self.heat = HeatItem()
        self.scene.addItem(self.heat)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.show_step)
        self.slider.sliderPressed.connect(self.stop_build)
        self.step_label = QLabel("Verlauf")
        self.step_label.setMinimumWidth(280)
        self.btn_prev = QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setToolTip("Vorheriger Schritt (B)")
        self.btn_prev.clicked.connect(self.previous_step)
        self.btn_next = QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setToolTip("Nächster Schritt (N); am Ende sagt das Netz genau einen weiteren Schritt vorher")
        self.btn_next.clicked.connect(self.next_step)
        self.btn_continue = QPushButton("Rest vorschlagen")
        self.btn_continue.clicked.connect(self.toggle_build)

        history = QHBoxLayout()
        history.setContentsMargins(6, 2, 6, 4)
        history.addWidget(self.btn_prev)
        history.addWidget(self.slider, 1)
        history.addWidget(self.btn_next)
        history.addWidget(self.step_label)
        history.addWidget(self.btn_continue)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addLayout(history)

    @property
    def circuit(self) -> str:
        return self.diagram.circuit if self.diagram is not None else ""

    def _message(self, text: str) -> None:
        self.win._message(text)

    def select_circuit(self, name: str) -> None:
        if not name:
            return
        self.stop_build()
        self.pending_focus = None
        # Editing happens on an unbounded sheet; the policy fits it into a grid itself.
        diagram = build_diagram(self.netlist, name, self.source.pin_maps, self.source.subcircuits)
        self.diagram = replace(diagram, bounded=False)
        self.highlights.clear()
        saved = saved_placement(self.diagram, self.source)
        if saved is not None:
            self.set_history([HistoryStep(ci, saved[ci], None, kind="given") for ci in placement_order(self.diagram)])
            self._message(f"{name}: freigegebene Platzierung aus schematic.json geladen.")
            self.view.fit()
        else:
            self.suggest()

    def show_example(self, diagram: Diagram, history: list[HistoryStep]) -> None:
        self.stop_build()
        self.diagram = replace(diagram, bounded=False)
        self.highlights.clear()
        self.set_history(history, step=1)
        self.view.fit()

    def suggest(self) -> None:
        """Propose a placement from scratch, step by step on screen."""
        if self.diagram is None:
            return
        self.set_history([])
        self.start_build()

    def start_build(self) -> None:
        """Let the policy add steps at the end of the history until all parts are placed."""
        self.building = True
        self.btn_continue.setText("Anhalten (Esc)")
        self._message("Vorschlag wird aufgebaut … (Esc hält an)")
        self.build_timer.start(0)

    def stop_build(self) -> None:
        if not self.building:
            return
        self.building = False
        self.build_timer.stop()
        self.btn_continue.setText("Rest vorschlagen")
        self._message("Aufbau angehalten. ▶ (N) für einzelne Schritte, „Rest vorschlagen“ für alle.")

    def finish_build(self) -> None:
        """Run the remaining build steps at once (scripts and tests)."""
        while self.building:
            self._build_step()

    def toggle_build(self) -> None:
        if self.building:
            self.stop_build()
        else:
            self.regenerate_future()

    def _build_step(self) -> None:
        if not self.building:
            return
        first = not self.history
        if not self.next_step(quiet=True):
            self.building = False
            self.btn_continue.setText("Rest vorschlagen")
            self.draw_wires()
            self.view.fit()
            if self.pending_focus is not None:
                self.focus(self.pending_focus)  # chosen in the hierarchy while the proposal was building
            if self.diagram is not None and len(self.placement) == len(self.diagram.components):
                self._message("Vorschlag erzeugt. Bauteile verschieben und mit „Send training data“ speichern.")
            return
        item = self.items[self.history[-1].component]
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        if first or not visible.contains(item.sceneBoundingRect()):
            self.view.fit()
        self.build_timer.start(0)

    # -------------------------------------------------------------- history
    def set_history(self, history: list[HistoryStep], step: int | None = None) -> None:
        """Replace the history; the final placement is its full replay."""
        self.history = history
        self.rollout = None  # rebuilt from the end state when the next step is requested
        self.placement = self.state_at(len(history))
        self.redraw()
        self.slider.blockSignals(True)
        self.slider.setMaximum(len(history))
        self.slider.setValue(len(history) if step is None else step)
        self.slider.blockSignals(False)
        self.show_step(self.slider.value())

    def state_at(self, k: int) -> Placement:
        """Placement after replaying the first ``k`` history steps."""
        state: Placement = {}
        for step in self.history[:k]:
            state[step.component] = step.position
        return state

    def at_end(self) -> bool:
        return self.slider.value() >= len(self.history)

    def show_step(self, k: int) -> None:
        """Show the replayed state after ``k`` steps; highlight step k and its probability map."""
        if self.diagram is None:
            return
        state = self.state_at(k)
        current = self.history[k - 1] if 0 < k <= len(self.history) else None
        for ci, item in self.items.items():
            item.setVisible(ci in state)
            if ci in state:
                x, y, o = state[ci]
                if item.rotation_steps != o:
                    item.set_rotation(o)
                item.setPos(x * CELL, y * CELL)
            item.current = current is not None and ci == current.component and k < len(self.history)
            item.invalid = False
            item.update()
        if current is not None and current.heat is not None and k < len(self.history):
            self.heat.set_heat(current.heat, current.offset)
        else:
            self.heat.set_heat(None, (0, 0))
        if current is None:
            self.step_label.setText(f"Schritt 0/{len(self.history)}")
        else:
            comp = self.diagram.components[current.component]
            note = {
                "place": "",
                "move": " (vom Netz verschoben)",
                "manual": " (korrigiert)",
                "given": " (vorgegeben)",
                "example": " (Trainingsbeispiel; Heatmap = aktuelles Modell)",
            }[current.kind]
            self.step_label.setText(f"Schritt {k}/{len(self.history)}: {comp.name}{note}")
        self.draw_wires()
        self.update_properties()

    def visible_placement(self) -> Placement:
        return self.state_at(self.slider.value())

    def apply_edit(self, changes: Placement) -> None:
        """Record a manual correction at the slider position.

        The changes become history steps right after the current one; the steps after
        them are discarded. Nothing is recomputed: ▶ asks for the next step.
        """
        self.stop_build()
        k = self.slider.value()
        state = self.state_at(k)
        corrections = [
            HistoryStep(ci, pos, None, kind="manual", previous=state.get(ci))
            for ci, pos in changes.items()
            if state.get(ci) != pos
        ]
        dropped = len(self.history) - k
        self.set_history(self.history[:k] + corrections)
        if dropped:
            self._message(f"Korrektur gespeichert, {dropped} spätere Schritte verworfen. ▶ (N) für den nächsten Schritt.")

    def previous_step(self) -> None:
        self.stop_build()
        self.slider.setValue(self.slider.value() - 1)

    def next_step(self, quiet: bool = False) -> bool:
        """Advance the slider; at the end of the history, predict exactly one new step.

        Returns False when there is no further step.
        """
        if self.diagram is None:
            return False
        if self.slider.value() < len(self.history):
            self.slider.setValue(self.slider.value() + 1)
            return True
        try:
            if self.rollout is None:
                self.rollout = trainer.Rollout(self.diagram, self.win.params, self.state_at(len(self.history)))
            step = self.rollout.next()
        except RuntimeError as exc:
            self._message(f"Kein weiterer Schritt möglich: {exc}")
            return False
        if step is None:
            if not quiet:
                self._message("Alle Bauteile sind platziert.")
            return False
        self.history.append(step)
        self.placement[step.component] = step.position
        if step.component not in self.items:
            x, y, o = step.position
            item = ComponentItem(self, step.component, o)
            item.setPos(x * CELL, y * CELL)
            self.scene.addItem(item)
            self.items[step.component] = item
        self.slider.blockSignals(True)
        self.slider.setMaximum(len(self.history))
        self.slider.setValue(len(self.history))
        self.slider.blockSignals(False)
        self.show_step(len(self.history))
        return True

    def regenerate_future(self) -> None:
        """Keep the steps up to the slider; the policy adds the remaining steps one by one."""
        if self.diagram is None:
            return
        self.set_history(self.history[: self.slider.value()])
        self.start_build()

    # -------------------------------------------------------------- drawing
    def redraw(self) -> None:
        selected = {item.ci for item in self.items.values() if item.isSelected()}
        for item in self.items.values():
            self.scene.removeItem(item)
        self.items.clear()
        if self.diagram is None:
            return
        for ci, (x, y, r) in self.placement.items():
            item = ComponentItem(self, ci, r)
            item.setPos(x * CELL, y * CELL)
            self.scene.addItem(item)
            item.setSelected(ci in selected)  # keep the selection across edits (e.g. pressing R twice)
            self.items[ci] = item
        self.draw_wires()
        if self.pending_focus is not None:
            self.focus(self.pending_focus)

    def draw_wires(self) -> None:
        # While a proposal builds up, routing (all nets again) would dominate each step:
        # route only after twice its last cost has passed; the end of the build routes again.
        if self.building and time.monotonic() - self.routed_at < 2 * self.route_cost:
            return
        start = time.monotonic()
        selected = {n for n, item in self.net_items.items() if item.isSelected()}
        for item in self.net_items.values():
            self.scene.removeItem(item)
        self.net_items.clear()
        if self.diagram is None or not self.placement:
            self.win._refresh_status()
            return
        shown = self.visible_placement()
        self.routing = route(self.diagram, shown)
        nets = net_graphics(self.diagram, shown, self.routing, named_labels=True)
        for net, prims in nets.items():
            item = NetItem(self, net, prims)
            self.scene.addItem(item)
            item.setSelected(net in selected)
            self.net_items[net] = item
        self.draw_frame(schematic_graphics(self.diagram, shown, self.routing))
        self.draw_groups(shown)
        self.routed_at = time.monotonic()
        self.route_cost = self.routed_at - start
        self.win._refresh_status()

    def draw_groups(self, shown: Placement) -> None:
        """Group backgrounds around their placed parts; while a proposal builds up, groups
        without placed parts show their planned region (coarse placement) dashed."""
        selected = {g for g, item in self.group_items.items() if item.isSelected()}
        for item in self.group_items.values():
            self.scene.removeItem(item)
        self.group_items.clear()
        planned = self.rollout.regions if self.rollout is not None and self.building else None
        for box in group_boxes(self.diagram, shown, planned):
            item = GroupItem(self, box)
            self.scene.addItem(item)
            item.setSelected(box.gid in selected)
            self.group_items[box.gid] = item

    def draw_frame(self, drawing: list[Prim]) -> None:
        """Frame around the drawing: it grows and shrinks with the parts."""
        for item in (self.frame_item, self.symbol_item):
            if item is not None:
                self.scene.removeItem(item)
        source = str(self.source.path) if self.source is not None else ""
        sheet = sheet_frame(bbox(drawing), own_symbol(self.diagram.ports), self.diagram.circuit, source)
        self.frame_item = FrameItem(sheet.prims)
        self.scene.addItem(self.frame_item)
        self.symbol_item = None
        if sheet.symbol is not None:
            self.symbol_item = OwnSymbolItem(self, sheet.symbol, sheet.origin, sheet.circuit)
            self.scene.addItem(self.symbol_item)
        self.update_fly_lines()

    def _selected(self) -> list[ComponentItem]:
        return [i for i in self.scene.selectedItems() if isinstance(i, ComponentItem) and i.isVisible()]

    def select_all(self) -> None:
        for item in self.items.values():
            if item.isVisible():
                item.setSelected(True)

    def _moved(self, items: list[ComponentItem], dx: int = 0, dy: int = 0) -> Placement:
        return {i.ci: (i.cell()[0] + dx, i.cell()[1] + dy, i.rotation_steps) for i in items}

    def preview_move(self) -> None:
        """Colour the dragged selection red while it would not fit."""
        if self.diagram is None:
            return
        items = self._selected()
        ok = group_fits(self.diagram, self.visible_placement(), self._moved(items))
        for item in items:
            if item.invalid == ok:
                item.invalid = not ok
                item.update()

    def drop_selection(self) -> None:
        """Accept the move, else snap the selection to the nearest free offset, else revert."""
        if self.diagram is None:
            return
        items = self._selected()
        state = self.visible_placement()
        if not items or all(state[i.ci][:2] == i.cell() for i in items):
            return
        target = nearest_fit(self.diagram, state, self._moved(items), radius=4)
        if target is None:
            self._message("Kein freier Platz in der Nähe – Auswahl zurückgesetzt.")
            self.show_step(self.slider.value())
            return
        self.apply_edit(target)

    def rotate_selected(self) -> None:
        # Quarter turn within the same mirror state.
        self._reorient(lambda o: (o // 4) * 4 + (o + 1) % 4, "Drehung")

    def mirror_selected(self) -> None:
        self._reorient(lambda o: o ^ 4, "Spiegeln")

    def _reorient(self, change, what: str) -> None:
        """Rotate or mirror the selection in place, else at the nearest free position."""
        if self.diagram is None:
            return
        items = self._selected()
        if not items:
            return
        state = self.visible_placement()
        moved = {}
        for item in items:
            x, y, o = state[item.ci]
            allowed = self.diagram.components[item.ci].rotations
            new = change(o)
            # Skip orientations the part does not allow (e.g. transformers only upright).
            for _ in range(8):
                if new in allowed:
                    break
                new = change(new)
            moved[item.ci] = (x, y, new if new in allowed else o)
        if all(moved[ci] == state[ci] for ci in moved):
            self._message(f"{what}: für dieses Bauteil nicht erlaubt.")
            return
        target = nearest_fit(self.diagram, state, moved, radius=3)
        if target is None:
            blockers = group_conflicts(self.diagram, state, moved)
            names = sorted(self.diagram.components[c].name for c in blockers if c is not None)
            self._message(f"{what} nicht möglich: kollidiert mit {', '.join(names) or 'Rastergrenze'}.")
            return
        if target != moved:
            self._message(f"{what}: Bauteil leicht versetzt, damit es passt.")
        self.apply_edit(target)

    # ----------------------------------------------------------- highlights
    def focus(self, element: str) -> bool:
        """Select and centre the part that draws netlist element ``element``.

        While a proposal still builds up the part may not be placed yet: it is selected
        as soon as it appears. Returns whether it is shown now."""
        ci = self.diagram.component_of(element) if self.diagram is not None else None
        item = self.items.get(ci) if ci is not None else None
        if item is None:
            self.pending_focus = element if ci is not None else None
            if ci is None:
                self._message(f"{element}: kein eigenes Symbol in {self.circuit}.")
            return False
        self.pending_focus = None
        self.scene.clearSelection()
        for net_item in self.net_items.values():
            net_item.setSelected(False)
        item.setSelected(True)
        self.view.centerOn(item.mapToScene(item.boundingRect().center()))
        return True

    def _selected_nets(self) -> list[str]:
        """Selected nets: wires / labels clicked on the sheet plus nets without wires selected
        through the own symbol (supply, ground)."""
        nets = [i.net for i in self.scene.selectedItems() if isinstance(i, NetItem)]
        return nets + [n for n in self.extra_nets if n not in nets]

    def _selection_changed(self) -> None:
        if not self._selecting:
            self.extra_nets = []  # a click on the sheet replaces a symbol-only selection
        self.update_properties()
        self.update_fly_lines()

    def clear_extra_nets(self) -> None:
        """Click on empty sheet space: drop a selection the scene does not know about."""
        if self.extra_nets:
            self.extra_nets = []
            self.update_properties()
            self.update_fly_lines()

    def set_pad_nets(self, nets: list[str]) -> None:
        """Nets of the pads selected in the footprint dock (fly lines only, the part stays selected)."""
        self.pad_nets = [n for n in dict.fromkeys(nets) if n]
        self.update_fly_lines()

    def highlight_selected_nets(self) -> None:
        nets = self._selected_nets()
        if not nets:
            self._message("Erst ein Netz anklicken (Draht oder Netz-Label).")
            return
        for net in nets:
            self.highlights[net] = QColor.fromHsv(random.randrange(360), 220, 255).name()
        self.scene.clearSelection()
        self.scene.update()

    def select_net(self, net: str, additive: bool = False) -> None:
        """Select a net as if one of its wires had been clicked (own symbol's pins). Supply
        and ground nets have no wire: they are selected all the same and their fly lines
        lead to every supply / ground symbol of the net."""
        self._selecting = True
        try:
            if not additive:
                self.scene.clearSelection()
                self.extra_nets = []
            item = self.net_items.get(net)
            if item is not None:
                item.setSelected(True)
            elif net not in self.extra_nets:
                self.extra_nets.append(net)
        finally:
            self._selecting = False
        self.update_properties()
        self.update_fly_lines()

    def _parts_nets(self) -> list[str]:
        """Every net of the selected parts, supply and ground included, in pin order."""
        nets: list[str] = []
        for item in self._selected():
            nets += [n for n in item.component.symbol.pin_nodes if n]
        return list(dict.fromkeys(nets))

    def update_fly_lines(self) -> None:
        """Fly lines for every selected net and every net of the selected parts: between its
        islands (wired parts, labelled pins, port labels, own symbol pin), minimum spanning
        tree, curved. The nets of selected parts are drawn emphasised as well."""
        before = self.related_nets
        self.related_nets = set(self._parts_nets()) if self.diagram is not None else set()
        for net in before | self.related_nets:
            if net in self.net_items:
                self.net_items[net].update()
        for item in self.fly_items:
            self.scene.removeItem(item)
        self.fly_items = []
        if self.diagram is not None and self.routing is not None:
            shown = self.visible_placement()
            for net in dict.fromkeys(self._selected_nets() + self.pad_nets + list(self.related_nets)):
                point = self.symbol_item.pin_point(net) if self.symbol_item is not None else None
                islands = net_islands(self.diagram, shown, self.routing, net, [point] if point else None)
                lines = fly_lines(islands)
                if lines:
                    color = QColor(self.highlights[net]) if net in self.highlights else QColor(FLY_COLOR)
                    color.setAlpha(FLY_COLOR.alpha())
                    item = FlyLineItem(lines, color)
                    self.scene.addItem(item)
                    self.fly_items.append(item)
        if self.symbol_item is not None:
            self.symbol_item.update()

    def clear_highlights(self) -> None:
        self.highlights.clear()
        self.scene.update()

    # ----------------------------------------------------------- properties
    def update_properties(self) -> None:
        if self is not self.win.sheet:
            return  # the dock shows the current tab only
        rows: list[tuple[str, str]] = []
        parts = self._selected()
        nets = self._selected_nets()
        groups = [i for i in self.scene.selectedItems() if isinstance(i, GroupItem)]
        if len(groups) == 1 and not parts and not nets and self.diagram is not None:
            box = groups[0].box
            members = [self.diagram.components[ci].name for ci in self.diagram.group_members(box.gid)]
            rows += [("Gruppe", box.title), ("Beschreibung", box.description or "–"),
                     ("Bauteile", ", ".join(m for m in members if not m.startswith("port:")))]
            if box.planned:
                rows.append(("Status", "geplante Region (Grobplatzierung)"))
        elif len(parts) == 1 and not nets and self.diagram is not None:
            item = parts[0]
            comp = item.component
            x, y, o = self.visible_placement()[item.ci]
            rows += [
                ("Name", comp.name),
                ("Typ", KIND_NAMES.get(comp.kind, comp.kind)),
                ("Beschreibung", comp.title),
                ("Position", f"{x}, {y}"),
                ("Orientierung", f"{(o % 4) * 90}°" + (", gespiegelt" if o >= 4 else "")),
            ]
            if comp.subckt:
                rows.append(("Subcircuit", f"{comp.subckt} (Doppelklick öffnet ihn in einem Tab)"))
            group = self.diagram.group(comp.group) if comp.group else None
            if group is not None:
                rows += [("Gruppe", group.title), ("Gruppenbeschreibung", group.description or "–")]
            parts = self.diagram.parts_of(item.ci)
            for part in parts:
                rows += [("Bauteil", f"{part.ref} {part.value}".strip()), ("Footprint", part.footprint)]
                if part.datasheet:
                    rows.append(("Datenblatt", part.datasheet))
            note = "Subcircuit-Block: Footprints im geöffneten Subcircuit." if comp.subckt else "Kein Board-Bauteil (Modell)."
            self.win.show_footprints(parts, self.diagram.pad_nets, note)
            steps = [str(i + 1) for i, h in enumerate(self.history) if h.component == item.ci]
            if steps:
                rows.append(("Verlaufsschritte", ", ".join(steps)))
            for pi, node in enumerate(comp.symbol.pin_nodes):
                kind = "Masse" if node and is_ground(node) else "per Name" if node in self.diagram.named_nets else ""
                rows.append((f"Pin {pi + 1}", f"{node or 'NC'}" + (f" ({kind})" if kind else "")))
        elif len(nets) == 1 and not parts and self.diagram is not None:
            net = nets[0]
            mode = "per Name" if net in self.diagram.named_nets else "Draht"
            if net in self.diagram.power_nets:
                mode = "Versorgung (Symbol an jedem Pin)"
            elif self.diagram.is_ground(net):
                mode = "Masse (Symbol an jedem Pin)"
            if self.routing is not None and net in self.routing.unrouted:
                mode = "nicht geroutet (Labels)"
            rows += [("Netz", net), ("Verbindung", mode)]
            for ci, pi in self.diagram.nets.get(net, []):
                comp = self.diagram.components[ci]
                rows.append(("Anschluss", f"{comp.name} Pin {pi + 1}"))
            if net in self.highlights:
                rows.append(("Hervorhebung", self.highlights[net]))
        elif parts or nets:
            rows.append(("Auswahl", f"{len(parts)} Bauteile, {len(nets)} Netze"))
        self.win.properties.setRowCount(len(rows))
        for r, (key, value) in enumerate(rows):
            self.win.properties.setItem(r, 0, QTableWidgetItem(key))
            cell = QTableWidgetItem(value)
            cell.setToolTip(value)
            self.win.properties.setItem(r, 1, cell)
        self.win.properties.resizeRowsToContents()  # descriptions wrap over several lines

    def export_svg(self) -> None:
        if self.diagram is None or len(self.placement) < len(self.diagram.components):
            return
        default = f"{self.diagram.circuit}.svg"
        if self.source is not None and self.source.circuit_dir is not None:
            default = str(self.source.circuit_dir / default)
        path, _ = QFileDialog.getSaveFileName(self, "SVG exportieren", default, "SVG (*.svg)")
        if path:
            source = str(self.source.path) if self.source is not None else ""
            Path(path).write_text(schematic_svg(self.diagram, self.placement, source=source))
            self._message(f"SVG gespeichert: {path}")



class MainWindow(QMainWindow):
    """Toolbar, tabs (one sheet per opened subcircuit), netlist and properties docks."""

    def __init__(self, db: TrainingDB, model: Path, source_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("netlist2schematic")
        self.resize(1500, 900)
        self.db = db
        self.model = model
        self.source: Source | None = None
        self.netlist: Netlist | None = None
        self.worker: TrainWorker | None = None
        self.params = policy.load_or_init(model)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

        # Toolbar: icon buttons (qtawesome) with a tooltip: name, description and shortcut.
        color = self.palette().color(QPalette.ColorRole.WindowText).name()

        def icon(name: str) -> QIcon:
            return qta.icon(name, color=color)

        def sheet_call(method: str):
            return lambda: getattr(self.sheet, method)() if self.sheet is not None else None

        entries = [
            ("open", "Öffnen", "Netzliste, sizing.py oder .jinja2 öffnen und die Schaltung anzeigen.", "Ctrl+O", icon("fa5s.folder-open"), self.open_dialog),
            ("reload", "Neu laden", "Quelle erneut ausführen; alle Tabs dieser Quelle aktualisieren.", "F5", icon("fa5s.sync-alt"), self.reload),
            None,
            ("suggest", "Vorschlag", "Das Netz baut einen Platzierungsvorschlag für diese Schaltung Schritt für Schritt auf.", None, icon("fa5s.magic"), sheet_call("suggest")),
            ("rotate", "Drehen", "Ausgewählte Bauteile um 90° drehen.", "R", icon("fa5s.redo"), sheet_call("rotate_selected")),
            ("mirror", "Spiegeln", "Ausgewählte Bauteile spiegeln.", "M", icon("fa5s.arrows-alt-h"), sheet_call("mirror_selected")),
            ("fit", "Einpassen", "Die ganze Schaltung in die Ansicht einpassen.", "F", icon("fa5s.expand-arrows-alt"), lambda: self.sheet and self.sheet.view.fit()),
            None,
            ("highlight", "Netz hervorheben", "Ausgewählte Netze in einer zufälligen Farbe hervorheben.", "9", icon("fa5s.highlighter"), sheet_call("highlight_selected_nets")),
            ("clear", "Hervorhebungen löschen", "Alle Netz-Hervorhebungen entfernen.", "0", icon("fa5s.eraser"), sheet_call("clear_highlights")),
            None,
            ("send", "Send training data", "Die fertige Platzierung als Trainingsbeispiel speichern (und als freigegebene Platzierung).", "Ctrl+S", icon("fa5s.save"), self.send_training_data),
            ("retrain", "Neu trainieren", "Das Netz mit allen Trainingsbeispielen neu trainieren.", None, icon("fa5s.brain"), self.retrain),
            ("delete_weights", "Gewichte löschen", "Die Netzgewichte löschen; Vorschläge nutzen dann nur die Grundheuristik.", None, icon("fa5s.trash-alt"), self.delete_weights),
            ("training_data", "Trainingsdaten…", "Liste der Trainingsbeispiele mit Varianten; öffnen oder löschen.", None, icon("fa5s.table"), self.show_training_data),
            None,
            ("export", "SVG exportieren…", "Die fertige Schaltung als SVG-Datei speichern.", None, icon("fa5s.file-export"), sheet_call("export_svg")),
        ]
        self.toolbar = QToolBar("Werkzeuge")
        self.toolbar.setObjectName("tools")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(22, 22))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(self.toolbar)
        self.actions: dict[str, QAction] = {}
        for entry in entries:
            if entry is None:
                self.toolbar.addSeparator()
                continue
            key, text, description, shortcut, button_icon, slot = entry
            action = QAction(button_icon, text, self)
            tip = f"{text}\n{description}"
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
                tip = f"{text} ({QKeySequence(shortcut).toString(QKeySequence.SequenceFormat.NativeText)})\n{description}"
            action.setToolTip(tip)
            action.setStatusTip(description)
            action.triggered.connect(slot)
            self.toolbar.addAction(action)
            self.actions[key] = action

        for text, key, slot in (
            ("Nächster Schritt", "N", sheet_call("next_step")),
            ("Vorheriger Schritt", "B", sheet_call("previous_step")),
            ("Aufbau anhalten", "Esc", sheet_call("stop_build")),
            ("Alles auswählen", QKeySequence.StandardKey.SelectAll, sheet_call("select_all")),
        ):
            action = QAction(text, self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(slot)
            self.addAction(action)

        # Netlist dock: circuit of the current tab and the netlist text.
        self.circuit_box = QComboBox()
        self.circuit_box.currentTextChanged.connect(self.select_circuit)
        self.netlist_view = QPlainTextEdit()
        self.netlist_view.setReadOnly(True)
        self.netlist_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.netlist_view.setFont(mono)
        netlist_panel = QWidget()
        netlist_layout = QVBoxLayout(netlist_panel)
        netlist_layout.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("Schaltung"))
        row.addWidget(self.circuit_box, 1)
        netlist_layout.addLayout(row)
        netlist_layout.addWidget(self.netlist_view, 1)
        self.netlist_dock = QDockWidget("Netzliste", self)
        self.netlist_dock.setObjectName("netlist")
        self.netlist_dock.setWidget(netlist_panel)
        self.netlist_dock.setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.netlist_dock)

        self.properties = QTableWidget(0, 2)
        self.properties.setHorizontalHeaderLabels(["Eigenschaft", "Wert"])
        self.properties.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.properties.verticalHeader().setVisible(False)
        self.properties.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.properties.setWordWrap(True)
        self.properties_dock = QDockWidget("Eigenschaften", self)
        self.properties_dock.setObjectName("properties")
        self.properties_dock.setWidget(self.properties)
        self.properties_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)

        # Hierarchy dock: cell tree of the source; selecting an element shows it in its parent's sheet.
        self.hierarchy = QTreeWidget()
        self.hierarchy.setHeaderLabels(["Instanz", "Typ"])
        self.hierarchy.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.hierarchy.setUniformRowHeights(True)
        self.hierarchy.itemSelectionChanged.connect(self._hierarchy_selected)
        self.hierarchy_dock = QDockWidget("Hierarchie", self)
        self.hierarchy_dock.setObjectName("hierarchy")
        self.hierarchy_dock.setWidget(self.hierarchy)
        self.hierarchy_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.hierarchy_dock)
        self.splitDockWidget(self.properties_dock, self.hierarchy_dock, Qt.Orientation.Vertical)

        # Footprint dock: KiCad footprint and 3D model of the selected symbol.
        self.footprint_panel = FootprintPanel()
        self.footprint_dock = QDockWidget("Footprint", self)
        self.footprint_dock.setObjectName("footprint")
        self.footprint_dock.setWidget(self.footprint_panel)
        self.footprint_panel.pads_selected.connect(lambda nets: self.sheet.set_pad_nets(nets) if self.sheet is not None else None)
        self.footprint_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.footprint_dock)
        self.tabifyDockWidget(self.hierarchy_dock, self.footprint_dock)
        self.hierarchy_dock.raise_()

        self.status_label = QLabel()
        self.progress_label = QLabel()
        self.statusBar().addWidget(self.progress_label)
        self.statusBar().addPermanentWidget(self.status_label)
        self._refresh_status()

        if source_path is not None:
            # Show the window first; loading (sizing code, proposal) runs in the event loop.
            QTimer.singleShot(0, lambda: self.load(source_path))

    # ------------------------------------------------------------------ tabs
    @property
    def sheet(self) -> SheetTab | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, SheetTab) else None

    def sheets(self) -> list[SheetTab]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def _add_sheet(self, source: Source, netlist: Netlist, title: str) -> SheetTab:
        sheet = SheetTab(self, source, netlist)
        index = self.tabs.addTab(sheet, title)
        self.tabs.setCurrentIndex(index)
        return sheet

    def open_subcircuit(self, source: Source, netlist: Netlist, name: str) -> SheetTab:
        """Show subcircuit ``name`` in its tab (opened on first use)."""
        for i, sheet in enumerate(self.sheets()):
            if sheet.source is source and sheet.circuit.lower() == name.lower():
                self.tabs.setCurrentIndex(i)
                return sheet
        sheet = self._add_sheet(source, netlist, name)
        sheet.select_circuit(netlist.get(name).name)
        self._tab_changed()
        return sheet

    def close_tab(self, index: int) -> None:
        sheet = self.tabs.widget(index)
        if isinstance(sheet, SheetTab):
            sheet.stop_build()
        self.tabs.removeTab(index)
        if sheet is not None:
            sheet.deleteLater()

    def _tab_changed(self, *_args) -> None:
        """Docks follow the current tab: its netlist, circuit and selection."""
        sheet = self.sheet
        self.circuit_box.blockSignals(True)
        self.circuit_box.clear()
        if sheet is not None:
            if self.netlist_view.toPlainText() != sheet.source.netlist:
                self.netlist_view.setPlainText(sheet.source.netlist)
            self.circuit_box.addItems(top_subckts(sheet.netlist) if sheet.source is self.source else [sheet.circuit])
            if self.circuit_box.findText(sheet.circuit) < 0:
                self.circuit_box.addItem(sheet.circuit)
            self.circuit_box.setCurrentText(sheet.circuit)
            self._show_subckt_text(sheet.circuit)
            self.setWindowTitle(f"netlist2schematic – {sheet.source.path.name} – {sheet.circuit}")
            sheet.update_properties()
        else:
            self.netlist_view.clear()
            self.properties.setRowCount(0)
        self.circuit_box.blockSignals(False)
        self._refresh_status()

    def show_footprints(self, parts: list, pad_nets: dict[str, dict[str, str]], note: str = "") -> None:
        """Footprint dock for the selected symbol (kept while nothing else is selected)."""
        key = (tuple((p.ref, p.footprint) for p in parts), note if not parts else "")
        if key == getattr(self, "_footprint_key", None):
            return
        self._footprint_key = key
        self.footprint_panel.show_parts(parts, pad_nets, note)

    # ------------------------------------------------------------ hierarchy
    def _fill_hierarchy(self) -> None:
        """Cell tree of the current source: its top subcircuits (the preferred one first)."""
        self.hierarchy.blockSignals(True)
        self.hierarchy.clear()
        if self.source is not None and self.netlist is not None:
            preferred = default_circuit(self.source, self.netlist)
            roots = [preferred] + [n for n in top_subckts(self.netlist) if n != preferred]
            for name in roots:
                item = self._hierarchy_item(hierarchy(self.netlist, name))
                self.hierarchy.addTopLevelItem(item)
                item.setExpanded(name == preferred)
        self.hierarchy.blockSignals(False)

    def _hierarchy_item(self, node: HierarchyNode) -> QTreeWidgetItem:
        item = QTreeWidgetItem([node.name, node.model])
        item.setData(0, Qt.ItemDataRole.UserRole, (node.parent, node.name, node.subckt))
        item.setToolTip(0, f"{node.parent}/{node.name}" if node.parent else node.name)
        for child in node.children:
            item.addChild(self._hierarchy_item(child))
        return item

    def _hierarchy_selected(self) -> None:
        items = self.hierarchy.selectedItems()
        if not items:
            return
        parent, name, subckt = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not parent:
            self.open_subcircuit(self.source, self.netlist, subckt)  # a top subcircuit: its own sheet
            return
        self.focus_element(parent, name)

    def focus_element(self, parent: str, element: str) -> None:
        """Show subcircuit ``parent`` (its tab, opened if needed) with ``element`` selected."""
        if self.source is None or self.netlist is None:
            return
        sheet = self.open_subcircuit(self.source, self.netlist, parent)
        sheet.focus(element)

    def _show_subckt_text(self, name: str) -> None:
        """Scroll the netlist dock to the definition of ``name``."""
        if not name:
            return
        document = self.netlist_view.document()
        pattern = QRegularExpression(rf"^\.subckt\s+{QRegularExpression.escape(name)}\b", QRegularExpression.PatternOption.CaseInsensitiveOption | QRegularExpression.PatternOption.MultilineOption)
        found = document.find(pattern)
        if not found.isNull():
            self.netlist_view.setTextCursor(found)
            self.netlist_view.centerCursor()

    # ------------------------------------------------------------------ data
    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Netzliste öffnen", "", "Netzliste oder Sizing (*.cir *.sp *.net *.py *.jinja2 *.txt);;Alle Dateien (*)"
        )
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        """Open a source: all tabs are replaced by one for its main circuit."""
        try:
            source = load_source(path)
        except Exception as exc:  # sizing code may raise anything
            self._message(f"Laden fehlgeschlagen: {exc}")
            return
        self.source = source
        self.netlist = parse_netlist(source.netlist)
        while self.tabs.count():
            self.close_tab(0)
        preferred = default_circuit(source, self.netlist)
        sheet = self._add_sheet(source, self.netlist, preferred)
        sheet.select_circuit(preferred)
        self._fill_hierarchy()
        self._tab_changed()

    def reload(self) -> None:
        """Run the source again; every tab of it shows its circuit from the new netlist."""
        if self.source is None:
            return
        try:
            source = load_source(self.source.path)
        except Exception as exc:  # sizing code may raise anything
            self._message(f"Neu laden fehlgeschlagen: {exc}")
            return
        old, self.source = self.source, source
        self.netlist = parse_netlist(source.netlist)
        names = {n.lower() for n in self.netlist.names}
        for i, sheet in reversed(list(enumerate(self.sheets()))):
            if sheet.source is not old:
                continue
            if sheet.circuit.lower() not in names:
                self.close_tab(i)
                continue
            sheet.source, sheet.netlist = source, self.netlist
            sheet.select_circuit(self.netlist.get(sheet.circuit).name)
        self._fill_hierarchy()
        self._tab_changed()

    def select_circuit(self, name: str) -> None:
        """Circuit chosen in the netlist dock: shown in the current tab."""
        sheet = self.sheet
        if not name or sheet is None or name == sheet.circuit:
            return
        sheet.select_circuit(name)
        self.tabs.setTabText(self.tabs.currentIndex(), name)
        self._tab_changed()

    def open_example(self, ex) -> None:
        """Open a training example with its placement history in a new tab."""
        diagram, history = trainer.example_history(ex, self.model)
        source = Source(path=Path(f"Trainingsbeispiel {ex.id}"), netlist=ex.netlist, circuit=ex.circuit, pin_maps=ex.pin_maps)
        sheet = self._add_sheet(source, parse_netlist(ex.netlist), f"#{ex.id} {ex.circuit}")
        sheet.show_example(diagram, history)
        self._tab_changed()
        self._message(f"Trainingsbeispiel {ex.id} geladen: mit dem Regler deine Schritte durchgehen, die Heatmap zeigt das aktuelle Modell.")

    # ------------------------------------------------------------- training
    def send_training_data(self) -> None:
        sheet = self.sheet
        complete = sheet is not None and sheet.diagram is not None and len(sheet.visible_placement()) == len(sheet.diagram.components)
        if not complete:
            self._message("Keine vollständige Platzierung vorhanden (Verlauf bis zum Ende stellen).")
            return
        named = {sheet.diagram.components[ci].name: pos for ci, pos in sheet.placement.items()}
        try:
            env, shifted = trainer.fitted_env(sheet.diagram, named)
            env.demo_steps(shifted)
        except ValueError as exc:
            self._message(f"Platzierung ungültig: {exc}")
            return
        example_id = self.db.add(sheet.diagram.circuit, sheet.source.netlist, named, source="manual", pin_maps=sheet.source.pin_maps)
        message = f"Beispiel {example_id} gespeichert ({self.db.count()} in der Datenbank)."
        if sheet.source.circuit_dir is not None:
            saved = save_placement(sheet.source.circuit_dir, sheet.diagram.circuit, named)
            message += f" Freigabe in {saved.name}."
        self._message(message)
        self._refresh_status()

    def retrain(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.set_busy(True)
        self.worker = TrainWorker(self.db, self.model)
        self.worker.progress.connect(lambda s, d, t, i: self.progress_label.setText(f"{s}: {d}/{t} {i}"))
        self.worker.done.connect(self._on_trained)
        self.worker.start()

    def delete_weights(self) -> None:
        if not Path(self.model).exists():
            self._message("Keine Gewichte vorhanden.")
            return
        answer = QMessageBox.question(self, "Gewichte löschen", f"Die Gewichte in {self.model} löschen?\nVorschläge nutzen danach nur die Grundheuristik, bis neu trainiert wird.")
        if answer == QMessageBox.StandardButton.Yes:
            Path(self.model).unlink()
            self.params = policy.load_or_init(self.model)
            for sheet in self.sheets():
                sheet.rollout = None
            self._message("Gewichte gelöscht.")

    def show_training_data(self) -> None:
        TrainingDataDialog(self.db, self).exec()
        self._refresh_status()

    def set_busy(self, busy: bool) -> None:
        for key in ("suggest", "send", "retrain", "rotate", "mirror", "delete_weights", "training_data"):
            self.actions[key].setEnabled(not busy)

    def _on_trained(self, message: str) -> None:
        self.set_busy(False)
        self.params = policy.load_or_init(self.model)
        for sheet in self.sheets():
            sheet.rollout = None
        self.progress_label.setText(message)
        self._message(message + " Mit „Vorschlag“ das neue Modell testen.")

    # --------------------------------------------------------------- status
    def _message(self, text: str) -> None:
        self.statusBar().showMessage(text, 10000)

    def _refresh_status(self) -> None:
        parts = [f"Trainingsbeispiele: {self.db.count()}"]
        sheet = self.sheet
        complete = sheet is not None and sheet.diagram is not None and len(sheet.visible_placement()) == len(sheet.diagram.components)
        if complete:
            reward, details = evaluate(sheet.diagram, sheet.placement, sheet.routing)
            parts.append(
                f"Score {reward:.1f} · Kreuzungen {int(details['crossing'])} · "
                f"Drahtlänge {int(details['length'])} · Knicke {int(details['bend'])} · "
                f"unverbunden {int(details['unrouted'])}"
            )
        self.status_label.setText("   |   ".join(parts))


class TrainingDataDialog(QDialog):
    """Table of stored training examples; select rows, right click -> Löschen."""

    COLUMNS = ["ID", "Schaltung", "Erstellt", "Quelle", "Bauteile", "Varianten", "Training"]

    def __init__(self, db: TrainingDB, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Trainingsdaten")
        self.resize(640, 420)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.info = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.info)
        self.reload()

    def reload(self) -> None:
        examples = self.db.examples()
        cfg = trainer.TrainConfig()  # the configuration "Neu trainieren" uses
        replaced = trainer.superseded(examples) if cfg.newest_per_netlist else {}
        self.examples = {ex.id: ex for ex in examples}
        self.table.setRowCount(len(examples))
        total = 0
        for row, ex in enumerate(examples):
            try:
                count = len(trainer.example_variants(ex, cfg)[2])
                total += 0 if ex.id in replaced else count
                derived = str(count)
            except (KeyError, ValueError):
                derived = "ungültig"
            used = f"ersetzt durch #{replaced[ex.id]}" if ex.id in replaced else "ja"
            values = [str(ex.id), ex.circuit, ex.created_at, ex.source, str(len(ex.placement)), derived, used]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.info.setText(
            f"{len(examples)} Beispiele ({len(examples) - len(replaced)} im Training), {total} abgeleitete Varianten "
            f"(je bis zu {cfg.augment_subsets} Teilschaltungen + {cfg.augment_shifts} Verschiebungen). "
            "Rechtsklick: Öffnen / Löschen. Mehrere Zeichnungen derselben Netzliste: nur die neueste wird trainiert."
        )

    def selected_ids(self) -> list[int]:
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        return [int(self.table.item(r, 0).text()) for r in sorted(rows)]

    def _context_menu(self, pos) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        menu = QMenu(self)
        open_action = menu.addAction("Öffnen")
        open_action.setEnabled(len(ids) == 1 and hasattr(self.parent(), "open_example"))
        delete_action = menu.addAction(f"Löschen ({len(ids)})")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen == open_action:
            self.open(ids[0])
        elif chosen == delete_action:
            self.delete(ids, confirm=True)

    def open(self, example_id: int) -> None:
        self.parent().open_example(self.examples[example_id])
        self.accept()

    def delete(self, ids: list[int], confirm: bool = True) -> None:
        if confirm:
            answer = QMessageBox.question(self, "Löschen", f"{len(ids)} Trainingsbeispiel(e) endgültig löschen?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.db.delete(ids)
        self.reload()


class PadItem(QGraphicsPathItem):
    """One pad of the footprint drawing: selectable (click, Shift/Cmd, rubber band); the
    selected pads' nets get fly lines on the sheet."""

    def __init__(self, path: QPainterPath, number: str, net: str):
        super().__init__(path)
        self.number, self.net = number, net
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPen(Qt.PenStyle.NoPen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(2)
        self.setToolTip(f"Pad {number}: {net}" if net else f"Pad {number} (nicht verbunden)")

    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(SELECTED), 2.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 60))
            painter.drawPath(self.path())


class PrimItem(QGraphicsItem):
    """Static primitives (footprint drawing), not interactive."""

    def __init__(self, prims: list[Prim]):
        super().__init__()
        self.prims = prims
        self._rect = _rect_of(prims)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_prims(painter, self.prims)


def project_model(lib_id: str) -> Path | None:
    """3D model the project generates for a footprint without library model (common/models3d.py)."""
    try:
        import importlib

        return importlib.import_module("src.project.common.models3d").model_file(lib_id)
    except Exception:  # not inside the project, or no such model
        return None


class RenderWorker(QThread):
    """Renders a footprint in 3D with kicad-cli (cached), off the UI thread."""

    done = Signal(int, str, str)  # request id, png path ("" on failure), message

    def __init__(self, request: int, fp: "FP.Footprint", view: str, ref: str, model: Path | None, zoom: float):
        super().__init__()
        self.args = (request, fp, view, ref, model, zoom)

    def run(self) -> None:
        request, fp, view, ref, model, zoom = self.args
        try:
            png = FP.render_3d(fp, paths.data_dir() / "footprint3d", view, ref, model, zoom=zoom)
            self.done.emit(request, str(png) if png else "", "" if png else "kicad-cli nicht gefunden")
        except Exception as exc:  # kicad-cli errors, timeouts
            self.done.emit(request, "", str(exc))


class FootprintPanel(QWidget):
    """Footprint of the selected symbol: KiCad footprint drawing (pads with their nets) and,
    if the footprint has a 3D model, a 3D render with selectable view. Pads can be selected
    (several with Shift/Cmd or a rubber band); ``pads_selected`` sends their nets."""

    pads_selected = Signal(list)

    def __init__(self):
        super().__init__()
        self.parts: list = []
        self.pad_nets: dict[str, dict[str, str]] = {}
        self.footprint: FP.Footprint | None = None
        self.model: Path | None = None
        self.view_name = "Iso"
        self.yaw = 45
        self.zoom = 0.6
        self.request = 0
        self.workers: list[RenderWorker] = []

        self.part_box = QComboBox()
        self.part_box.currentIndexChanged.connect(self._show_part)
        self.info = QLabel("Kein Bauteil ausgewählt.")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor("#0b0f14"))
        self.scene.selectionChanged.connect(self._pads_changed)
        self.view2d = QGraphicsView(self.scene)
        self.view2d.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view2d.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.view2d.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        self.view2d.setMinimumHeight(180)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(200)
        self.image.setStyleSheet("background: #0b0f14; color: #9ca3af;")
        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        for name in FP.VIEWS:
            b = QToolButton()
            b.setText(name)
            b.clicked.connect(lambda _=False, n=name: self.set_view(n))
            buttons.addWidget(b)
        for text, tip, slot in (("⟲", "Um 45° drehen", lambda: self.turn(-45)), ("⟳", "Um 45° drehen", lambda: self.turn(45)),
                                ("−", "Verkleinern", lambda: self.zoom_by(0.8)), ("+", "Vergrößern", lambda: self.zoom_by(1.25))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.part_box)
        layout.addWidget(self.info)
        layout.addWidget(QLabel("Footprint"))
        layout.addWidget(self.view2d, 1)
        layout.addWidget(QLabel("3D-Modell"))
        layout.addLayout(buttons)
        layout.addWidget(self.image, 1)
        self.part_box.hide()

    # ------------------------------------------------------------ content
    def show_parts(self, parts: list, pad_nets: dict[str, dict[str, str]], note: str = "") -> None:
        """Parts of the selected symbol (several when one instance is several board parts)."""
        self.parts, self.pad_nets = parts, pad_nets
        self.part_box.blockSignals(True)
        self.part_box.clear()
        for part in parts:
            self.part_box.addItem(f"{part.ref}: {part.footprint.split(':')[-1]}")
        self.part_box.setVisible(len(parts) > 1)
        self.part_box.blockSignals(False)
        if not parts:
            self.footprint = None
            self.info.setText(note or "Kein Bauteil ausgewählt.")
            self.scene.clear()
            self.image.clear()
            self.image.setText("")
            return
        self._show_part(0)

    def _show_part(self, index: int) -> None:
        if not 0 <= index < len(self.parts):
            return
        part = self.parts[index]
        lines = [f"<b>{part.ref}</b> {part.value}", part.footprint]
        if part.description:
            lines.append(part.description)
        if part.rating:
            lines.append(f"Rating: {part.rating}")
        if part.datasheet:
            lines.append(f'<a href="{part.datasheet}">Datenblatt</a>')
        self.info.setText("<br>".join(lines))
        self.info.setOpenExternalLinks(True)
        self.scene.clear()
        self.footprint = FP.load(part.footprint) if part.footprint else None
        if self.footprint is None:
            self.image.setText("Footprint nicht in der KiCad-Bibliothek gefunden.")
            return
        nets = self.pad_nets.get(part.ref, {})
        item = PrimItem(FP.graphics(self.footprint, nets))
        self.scene.addItem(item)
        for pad in self.footprint.pads:
            if pad.kind == "np_thru_hole":
                continue
            outline = FP.pad_outline(pad)
            path = QPainterPath(_pt(*outline[0]))
            for x, y in outline[1:]:
                path.lineTo(_pt(x, y))
            path.closeSubpath()
            self.scene.addItem(PadItem(path, pad.number, nets.get(pad.number, "")))
        self.scene.setSceneRect(item.boundingRect().adjusted(-CELL, -CELL, CELL, CELL))
        QTimer.singleShot(0, lambda: self.view2d.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))
        self.model = project_model(part.footprint)
        self.render3d()

    def selected_pads(self) -> list[PadItem]:
        return [i for i in self.scene.selectedItems() if isinstance(i, PadItem)]

    def _pads_changed(self) -> None:
        try:
            nets = [pad.net for pad in self.selected_pads()]
        except RuntimeError:  # scene being cleared
            nets = []
        self.pads_selected.emit(nets)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.scene.items():
            self.view2d.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ---------------------------------------------------------------- 3D
    def set_view(self, name: str) -> None:
        self.view_name = name
        self.yaw = 45 if name == "Iso" else 0
        self.render3d()

    def turn(self, degrees: int) -> None:
        self.yaw = (self.yaw + degrees) % 360
        self.render3d()

    def zoom_by(self, factor: float) -> None:
        self.zoom = min(3.0, max(0.2, self.zoom * factor))
        self.render3d()

    def rotation(self) -> str:
        x, y, z = (float(v) for v in FP.VIEWS[self.view_name].split(","))
        return f"{x:g},{y:g},{(z - (45 if self.view_name == 'Iso' else 0) + self.yaw) % 360:g}"

    def render3d(self) -> None:
        fp = self.footprint
        if fp is None:
            return
        if not FP.has_3d_model(fp, self.model):
            self.image.setPixmap(QPixmap())
            self.image.setText("Kein 3D-Modell vorhanden.")
            return
        self.request += 1
        self.image.setText("3D-Ansicht wird gerendert …")
        ref = self.parts[self.part_box.currentIndex()].ref if self.parts else ""
        worker = RenderWorker(self.request, fp, self.rotation(), ref, self.model, self.zoom)
        worker.done.connect(self._rendered)
        worker.finished.connect(lambda w=worker: self.workers.remove(w) if w in self.workers else None)
        self.workers.append(worker)
        worker.start()

    def _rendered(self, request: int, png: str, message: str) -> None:
        if request != self.request:
            return  # an older request (the selection changed meanwhile)
        if not png:
            self.image.setPixmap(QPixmap())
            self.image.setText(f"3D-Render fehlgeschlagen: {message}")
            return
        pixmap = QPixmap(png)
        self.image.setPixmap(pixmap.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

    def wait(self) -> None:
        """Finish pending renders (tests, closing)."""
        for worker in list(self.workers):
            worker.wait()


class TrainWorker(QThread):
    progress = Signal(str, int, int, str)
    done = Signal(str)

    def __init__(self, db: TrainingDB, model: Path):
        super().__init__()
        self.db = db
        self.model = model

    def run(self) -> None:
        try:
            history = trainer.train(
                self.db, self.model, trainer.TrainConfig(), progress=lambda s, d, t, i: self.progress.emit(s, d, t, i)
            )
            last = history["bc_loss"][-1] if history["bc_loss"] else float("nan")
            self.done.emit(f"Training fertig, letzter Verlust {last:.4f}.")
        except Exception as exc:  # surfaced in the UI
            self.done.emit(f"Training fehlgeschlagen: {exc}")


def run_ui(source_path: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(TrainingDB(paths.db_path()), paths.model_path(), source_path)
    window.show()
    return app.exec()
