"""Coarse placement of the functional groups (``*@groups`` header) before the detail.

Every group gets a rectangular region sized by the parts it holds (their reserved cells
plus room for wires). The regions are ordered by signal flow, the group with the most
input ports first, then always the group most connected to the ones already ordered
(the implicit REST group last), and packed left to right in rows. The detail placement
keeps each part inside its group's region (``PlacementEnv.regions``), so a group is
drawn as one block instead of being scattered over the sheet.
"""

from __future__ import annotations

import math

from .diagram import REST, Diagram
from .netlist import signal_direction
from .placement import relative_cells

Region = tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive cells)

SLACK = 2.3  # region area per reserved part area (wires, spacing)
MARGIN = 2  # cells around every part in the area estimate
GAP = 3  # cells between two regions
ASPECT = 1.5  # preferred width / height of a region and of the whole floorplan


def part_size(diagram: Diagram, ci: int) -> tuple[int, int]:
    """Width and height of everything a part reserves (body, pins, symbols, pin exits)."""
    comp = diagram.components[ci]
    o = comp.rotations[0]
    body, pins = relative_cells(diagram, ci, o)
    cells = body + pins + comp.exit_cells(o)
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def group_size(diagram: Diagram, gid: str) -> tuple[int, int]:
    sizes = [part_size(diagram, ci) for ci in diagram.group_members(gid)]
    if not sizes:
        return 0, 0
    area = SLACK * sum((w + MARGIN) * (h + MARGIN) for w, h in sizes)
    w = max(max(s[0] for s in sizes) + MARGIN, math.ceil(math.sqrt(area * ASPECT)))
    h = max(max(s[1] for s in sizes) + MARGIN, math.ceil(area / w))
    return w, h


def group_links(diagram: Diagram) -> dict[tuple[str, str], int]:
    """Number of signal nets (wired or by name) shared by every pair of groups."""
    links: dict[tuple[str, str], int] = {}
    for net, pins in diagram.nets.items():
        if diagram.is_ground(net) or net in diagram.power_nets:
            continue
        gids = sorted({diagram.components[ci].group for ci, _ in pins})
        for i, a in enumerate(gids):
            for b in gids[i + 1 :]:
                links[(a, b)] = links.get((a, b), 0) + 1
    return links


def flow_order(diagram: Diagram) -> list[str]:
    """Groups by signal flow: most input ports first, then the most connected one next."""
    gids = [g.id for g in diagram.groups if diagram.group_members(g.id)]
    if not gids:
        return []
    links = group_links(diagram)

    def link(a: str, b: str) -> int:
        return links.get((min(a, b), max(a, b)), 0)

    def inputs(gid: str) -> int:
        count = 0
        for ci in diagram.group_members(gid):
            comp = diagram.components[ci]
            names = [comp.title] if comp.kind == "P" else [n for n in comp.symbol.pin_nodes if n in diagram.ports]
            count += sum(1 for n in names if signal_direction(n) == "in")
        return count

    rest = [REST] if REST in gids else []
    left = [g for g in gids if g != REST]
    order: list[str] = []
    while left:
        if not order:
            best = max(left, key=lambda g: (inputs(g), sum(link(g, o) for o in left), -left.index(g)))
        else:
            best = max(left, key=lambda g: (sum(link(g, o) for o in order), link(g, order[-1]), -left.index(g)))
        order.append(best)
        left.remove(best)
    return order + rest


def _pack(order: list[str], sizes: dict[str, tuple[int, int]], row_width: int) -> dict[str, Region]:
    """Regions packed left to right in rows no wider than ``row_width`` (relative to 0, 0)."""
    regions: dict[str, Region] = {}
    x, y, row_h = 0, 0, 0
    for gid in order:
        w, h = sizes[gid]
        if x > 0 and x + w > row_width:
            x, y, row_h = 0, y + row_h + GAP, 0
        regions[gid] = (x, y, x + w - 1, y + h - 1)
        x += w + GAP
        row_h = max(row_h, h)
    return regions


def floorplan(diagram: Diagram, origin: tuple[int, int] = (0, 0)) -> dict[str, Region]:
    """Regions of all groups, packed in rows in signal-flow order, starting at ``origin``.

    The row width is chosen among the possible row breaks so that the whole floorplan
    comes closest to the sheet aspect ``ASPECT`` (landscape), not stacked into a column.
    """
    order = flow_order(diagram)
    sizes = {gid: group_size(diagram, gid) for gid in order}
    if not order:
        return {}
    widest = max(w for w, _ in sizes.values())
    candidates = {widest} | {sum(sizes[g][0] + GAP for g in order[:k]) - GAP for k in range(1, len(order) + 1)}

    def misfit(width: int) -> float:
        x0, y0, x1, y1 = extent(_pack(order, sizes, width))
        return abs(math.log(((x1 - x0 + 1) / (y1 - y0 + 1)) / ASPECT))

    best = min(sorted(c for c in candidates if c >= widest), key=misfit)
    return shifted(_pack(order, sizes, best), origin[0], origin[1])


def extent(regions: dict[str, Region]) -> Region:
    return (
        min(r[0] for r in regions.values()),
        min(r[1] for r in regions.values()),
        max(r[2] for r in regions.values()),
        max(r[3] for r in regions.values()),
    )


def shifted(regions: dict[str, Region], dx: int, dy: int) -> dict[str, Region]:
    return {g: (x0 + dx, y0 + dy, x1 + dx, y1 + dy) for g, (x0, y0, x1, y1) in regions.items()}


def aligned(diagram: Diagram, regions: dict[str, Region], placement: dict[int, tuple[int, int, int]]) -> dict[str, Region]:
    """The floorplan moved (as a whole) so already placed parts sit in their regions as
    well as possible: shift = mean offset of the placed parts from their region centres."""
    offsets = []
    for ci, (x, y, _) in placement.items():
        region = regions.get(diagram.components[ci].group)
        if region is not None:
            offsets.append((x - (region[0] + region[2]) / 2, y - (region[1] + region[3]) / 2))
    if not offsets:
        return regions
    dx = round(sum(o[0] for o in offsets) / len(offsets))
    dy = round(sum(o[1] for o in offsets) / len(offsets))
    return shifted(regions, dx, dy)
