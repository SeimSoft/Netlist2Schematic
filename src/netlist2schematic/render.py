"""Schematic drawing: components, ground symbols, wires, junctions, net names.

``component_graphics`` returns the primitives of one component relative to its
anchor (used by the Qt items). ``net_graphics`` groups everything that belongs
to one net (wires, junctions, net name, labels at pins) so a viewer can select
and highlight nets. ``schematic_graphics`` returns the whole sheet in grid
coordinates and ``schematic_svg`` turns it into an SVG document.

Nets between ICs are connected by name (labels at every pin). Nets the router
could not connect are shown the same way, so every connection stays readable.
"""

from __future__ import annotations

import math

from .diagram import REST, Component, Diagram, own_symbol
from .frame import frame_graphics
from .netlist import ground_name
from dataclasses import dataclass

from .graphics import WIRE, Circle, Line, Poly, Prim, Text, bbox, to_svg, transform
from .placement import Placement
from .routing import Routing, route
from .symbols import ground_glyph, net_label

NET_NAME_SIZE = 0.32
NET_NAME_COLOR = "#1d4ed8"


def ground_graphics(comp: Component, rotation: int) -> list[Prim]:
    prims: list[Prim] = []
    for pi, style in comp.ground_layout(rotation):
        (px, py), (dx, dy) = comp.pin(pi, rotation)
        name = ground_name(comp.symbol.pin_nodes[pi])
        if style == "stub":
            # Sideways pin: stub outwards, compact ground symbol hanging below its end.
            prims.append(Line(px, py, px + dx, py))
            prims += ground_glyph(px + dx, py, 0, 1, name, name_left=dx < 0)
        else:
            prims += ground_glyph(px, py, dx, dy, name)
    return prims


def component_graphics(comp: Component, rotation: int, net_labels: bool = True) -> list[Prim]:
    """Symbol, labels and ground symbols of one component, relative to its anchor.

    ``net_labels=False`` leaves out the connection-by-name labels (a viewer draws
    them with their net instead).
    """
    prims = [transform(p, rotation, 0, 0) for p in comp.symbol.graphics]
    prims += ground_graphics(comp, rotation) + comp.power_prims(rotation) + comp.attached_prims(rotation)
    if net_labels:
        prims += comp.net_label_prims(rotation)
    return prims + comp.label_prims(rotation)


def net_graphics(diagram: Diagram, placement: Placement, routing: Routing, named_labels: bool = False) -> dict[str, list[Prim]]:
    """Primitives per net: wires, junction dots, net name, and labels at pins.

    Labels at pins of unrouted nets are always included; those of nets connected
    by name only with ``named_labels`` (otherwise the components draw them).
    """
    out: dict[str, list[Prim]] = {}
    for net, paths in routing.paths.items():
        prims = out.setdefault(net, [])
        for path in paths:
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                prims.append(Line(ax, ay, bx, by, width=1.2, color=WIRE))
        for x, y in routing.junctions(net):
            prims.append(Circle(x, y, 0.13, fill=True, color=WIRE))
        if net not in routing.unrouted and diagram.port_component(net) is None:
            spot = routing.label_spot(net)
            if spot is not None:
                (ax, ay), (bx, by) = spot
                mx, my = (ax + bx) / 2, (ay + by) / 2
                if ay == by:
                    prims.append(Text(mx, my - 0.3, net, size=NET_NAME_SIZE, color=NET_NAME_COLOR))
                else:
                    prims.append(Text(mx + 0.2, my, net, anchor="start", size=NET_NAME_SIZE, color=NET_NAME_COLOR))
    labelled = set(routing.unrouted) | (set(diagram.named_nets) if named_labels else set())
    for net in labelled:
        for ci, pi in diagram.nets.get(net, []):
            if ci not in placement or diagram.components[ci].kind == "P" or diagram.is_power_pin(ci, pi):
                continue
            x, y, r = placement[ci]
            (ox, oy), (dx, dy) = diagram.components[ci].pin(pi, r)
            out.setdefault(net, []).extend(net_label(x + ox, y + oy, dx, dy, net))
    return out


def wire_graphics(diagram: Diagram, placement: Placement, routing: Routing) -> list[Prim]:
    return [p for prims in net_graphics(diagram, placement, routing).values() for p in prims]


GROUP_COLORS = ("#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5")
REST_COLOR = "#6b7280"
GROUP_PAD = 0.8  # cells between the parts and the group outline
GROUP_TITLE_SIZE = 0.55
GROUP_OPACITY = 0.10


@dataclass(frozen=True)
class GroupBox:
    """Outline of one functional group on the sheet: around its placed parts, or its planned
    region (coarse placement) while none of them is placed yet."""

    gid: str
    title: str
    description: str
    color: str
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1 in grid units
    planned: bool
    prims: tuple[Prim, ...]


def group_color(diagram: Diagram, gid: str) -> str:
    group = diagram.group(gid)
    if group is not None and group.color:
        return group.color
    if gid == REST:
        return REST_COLOR
    ids = [g.id for g in diagram.groups if g.id != REST]
    return GROUP_COLORS[ids.index(gid) % len(GROUP_COLORS)] if gid in ids else REST_COLOR


def group_boxes(
    diagram: Diagram, placement: Placement, planned: dict[str, tuple[int, int, int, int]] | None = None
) -> list[GroupBox]:
    """Coloured background, outline and title of every group (none without a header)."""
    boxes = []
    for group in diagram.groups:
        members = [ci for ci in diagram.group_members(group.id) if ci in placement]
        if members:
            rects = []
            for ci in members:
                x, y, r = placement[ci]
                bx0, by0, bx1, by1 = bbox(component_graphics(diagram.components[ci], r, net_labels=False))
                rects.append((bx0 + x, by0 + y, bx1 + x, by1 + y))
            x0 = min(r[0] for r in rects) - GROUP_PAD
            y0 = min(r[1] for r in rects) - GROUP_PAD
            x1 = max(r[2] for r in rects) + GROUP_PAD
            y1 = max(r[3] for r in rects) + GROUP_PAD
            is_plan = False
        elif planned and group.id in planned:
            gx0, gy0, gx1, gy1 = planned[group.id]
            x0, y0, x1, y1 = gx0 - 0.5, gy0 - 0.5, gx1 + 0.5, gy1 + 0.5
            is_plan = True
        else:
            continue
        color = group_color(diagram, group.id)
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        prims: tuple[Prim, ...] = (
            Poly(corners, closed=True, fill=True, color=color, opacity=GROUP_OPACITY, width=0),
            Poly(corners, closed=True, color=color, width=1.4, dashed=is_plan),
            Text(x0 + 0.2, y0 - 0.45, group.title, anchor="start", size=GROUP_TITLE_SIZE, color=color, title=group.description),
        )
        boxes.append(GroupBox(group.id, group.title, group.description, color, (x0, y0, x1, y1), is_plan, prims))
    return boxes


def schematic_graphics(diagram: Diagram, placement: Placement, routing: Routing | None = None) -> list[Prim]:
    if routing is None:
        routing = route(diagram, placement)
    prims: list[Prim] = []
    prims += wire_graphics(diagram, placement, routing)
    for ci, (x, y, r) in placement.items():
        prims += [transform(p, 0, x, y) for p in component_graphics(diagram.components[ci], r)]
    return prims


def schematic_svg(diagram: Diagram, placement: Placement, scale: float = 24.0, source: str = "") -> str:
    """SVG of the sheet inside its frame (grid, title block with ``source`` path, port list)."""
    groups = [p for box in group_boxes(diagram, placement) for p in box.prims]
    drawing = groups + schematic_graphics(diagram, placement)
    prims = frame_graphics(bbox(drawing), own_symbol(diagram.ports), diagram.circuit, source) + drawing
    return to_svg(prims, scale=scale, title=diagram.circuit)


def net_islands(
    diagram: Diagram,
    placement: Placement,
    routing: Routing,
    net: str,
    extra: list[tuple[float, float]] | None = None,
) -> list[list[tuple[float, float]]]:
    """The places a net appears on the sheet, grouped into islands that are already joined
    by wires: every placed pin of the net (labelled pins and port labels each on their own
    unless a wire reaches them) plus ``extra`` points (e.g. the own symbol's pin)."""
    parent: dict[tuple[float, float], tuple[float, float]] = {}

    def find(p):
        parent.setdefault(p, p)
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a, b):
        parent[find(a)] = find(b)

    pins: list[tuple[float, float]] = []
    for ci, pi in diagram.nets.get(net, []):
        if ci not in placement:
            continue
        x, y, r = placement[ci]
        (ox, oy), _ = diagram.components[ci].pin(pi, r)
        pins.append((x + ox, y + oy))
        find(pins[-1])
    for path in routing.paths.get(net, []):
        for a, b in zip(path, path[1:]):
            union(tuple(map(float, a)), tuple(map(float, b)))
    for p in extra or []:
        pins.append(p)
        find(p)
    islands: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for p in dict.fromkeys((float(x), float(y)) for x, y in pins):
        islands.setdefault(find(p), []).append(p)
    return list(islands.values())


def fly_lines(islands: list[list[tuple[float, float]]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Shortest set of connections joining all islands (minimum spanning tree, Prim), each
    between the closest two points of the islands it joins."""
    if len(islands) < 2:
        return []

    def closest(a, b):
        return min(((p, q) for p in a for q in b), key=lambda pq: math.dist(pq[0], pq[1]))

    joined = [0]
    lines = []
    while len(joined) < len(islands):
        best = None
        for i in joined:
            for j in range(len(islands)):
                if j in joined:
                    continue
                p, q = closest(islands[i], islands[j])
                if best is None or math.dist(p, q) < math.dist(*best[1]):
                    best = (j, (p, q))
        joined.append(best[0])
        lines.append(best[1])
    return lines
