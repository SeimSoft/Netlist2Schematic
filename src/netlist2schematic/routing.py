"""Orthogonal grid autorouter for schematic wires.

Each net is routed as a tree: the first pin is the seed, every further pin is
connected by a cheapest-path search to the tree built so far. Ground nets are
not routed (every ground pin carries its own ground symbol), and neither are
nets connected by name (labels at every pin).

Readability rules enforced by the search:
- bodies and pins of other nets are obstacles,
- a wire may cross a wire of another net only perpendicularly and straight
  through (no running along it, no bend or junction on the crossing),
- crossings and bends cost extra,
- a wire should leave a pin straight in the pin's direction and enter a pin from
  its direction (a bend right at a pin costs extra; still allowed when it is tight).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .diagram import Diagram
from .placement import Placement, body_cells, extent, pin_cells

CROSSING_COST = 6.0
BEND_COST = 0.5
PIN_EXIT_COST = 2.0  # leaving a pin sideways instead of straight out
PIN_ENTRY_COST = 2.0  # entering a pin from the side instead of head-on
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
NO_DIR = 4
H, V = "h", "v"


def _axis(d: int) -> str:
    return H if d in (0, 1) else V


Cell = tuple[int, int]


@dataclass
class Routing:
    paths: dict[str, list[list[Cell]]] = field(default_factory=dict)
    # cell -> net -> axes used there
    usage: dict[Cell, dict[str, set[str]]] = field(default_factory=dict)
    unrouted: list[str] = field(default_factory=list)
    pins: dict[str, set[Cell]] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return sum(len(p) - 1 for paths in self.paths.values() for p in paths)

    @property
    def bends(self) -> int:
        total = 0
        for paths in self.paths.values():
            for path in paths:
                for a, b, c in zip(path, path[1:], path[2:]):
                    if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                        total += 1
        return total

    @property
    def crossings(self) -> int:
        return sum(1 for nets in self.usage.values() if len(nets) > 1)

    def edges(self, net: str) -> set[tuple[Cell, Cell]]:
        result = set()
        for path in self.paths.get(net, []):
            for a, b in zip(path, path[1:]):
                result.add((min(a, b), max(a, b)))
        return result

    def junctions(self, net: str) -> list[Cell]:
        """Cells where three or more wire ends meet (a pin counts as one)."""
        degree: dict[Cell, int] = {}
        for a, b in self.edges(net):
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        for p in self.pins.get(net, set()):
            if p in degree:
                degree[p] += 1
        return sorted(c for c, d in degree.items() if d >= 3)

    def label_spot(self, net: str) -> tuple[Cell, Cell] | None:
        """Middle edge of the longest straight horizontal run (fallback: any run)."""
        best = None
        for path in self.paths.get(net, []):
            run = [path[0]]
            for a, b in zip(path, path[1:]):
                if len(run) > 1 and (run[-1][0] - run[-2][0], run[-1][1] - run[-2][1]) != (b[0] - a[0], b[1] - a[1]):
                    best = _better(best, run)
                    run = [a]
                run.append(b)
            best = _better(best, run)
        if best is None or len(best) < 2:
            return None
        mid = (len(best) - 1) // 2
        return best[mid], best[mid + 1]


def _better(best, run):
    if len(run) < 2:
        return best
    horizontal = run[0][1] == run[-1][1]
    score = len(run) + (100 if horizontal else 0)
    if best is None:
        return run
    best_h = best[0][1] == best[-1][1]
    return run if score > len(best) + (100 if best_h else 0) else best


def _search(
    start: Cell,
    goal: set[Cell],
    blocked: set[Cell],
    net: str,
    usage,
    area: tuple[int, int, int, int],
    pin_dirs: dict[Cell, tuple[int, int]] | None = None,
) -> list[Cell] | None:
    """Cheapest path from ``start`` to the tree ``goal``. ``pin_dirs`` maps the net's
    pin cells to their outward direction (straight exit/entry preferred)."""
    x0, y0, x1, y1 = area
    pin_dirs = pin_dirs or {}
    start_dir = DIRS.index(pin_dirs[start]) if start in pin_dirs else NO_DIR
    def foreign(cell: Cell) -> dict[str, set[str]]:
        return {n: a for n, a in usage.get(cell, {}).items() if n != net}

    goal = {g for g in goal if not foreign(g)}
    if start in goal:
        return [start]
    # Starting "already heading" out of the pin makes a straight exit free of bend cost.
    best: dict[tuple[Cell, int], float] = {(start, start_dir): 0.0}
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {(start, start_dir): None}
    heap = [(0.0, 0, start, start_dir)]
    counter = 0
    while heap:
        cost, _, cell, d = heapq.heappop(heap)
        if cost > best.get((cell, d), float("inf")):
            continue
        if cell in goal:
            path = []
            state: tuple[Cell, int] | None = (cell, d)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return path[::-1]
        on_crossing = bool(foreign(cell))
        for nd, (dx, dy) in enumerate(DIRS):
            if on_crossing and nd != d:
                continue  # go straight through a crossing
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (x0 <= nxt[0] <= x1 and y0 <= nxt[1] <= y1) or nxt in blocked:
                continue
            step = 1.0
            others = foreign(nxt)
            if others:
                axis = _axis(nd)
                if any(axes != {H if axis == V else V} for axes in others.values()):
                    continue  # would overlap, or hit a bend/junction of another net
                if nxt in goal:
                    continue
                step += CROSSING_COST
            if d != NO_DIR and nd != d:
                step += BEND_COST
                if cell == start and start_dir != NO_DIR:
                    step += PIN_EXIT_COST
            if nxt in goal and nxt in pin_dirs:
                ox, oy = pin_dirs[nxt]
                if (dx, dy) != (-ox, -oy):
                    step += PIN_ENTRY_COST
            new_cost = cost + step
            state = (nxt, nd)
            if new_cost < best.get(state, float("inf")):
                best[state] = new_cost
                parent[state] = (cell, d)
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nxt, nd))
    return None


def _record(usage, net: str, path: list[Cell]) -> None:
    for i, cell in enumerate(path):
        axes = usage.setdefault(cell, {}).setdefault(net, set())
        for j in (i - 1, i + 1):
            if 0 <= j < len(path):
                axes.add(H if path[j][1] == cell[1] else V)


def route(diagram: Diagram, placement: Placement) -> Routing:
    """Route every non-ground net whose pins are placed.

    Short nets go first (fewest options). If nets fail, a second pass routes the
    failed ones first and the better result is kept.
    """
    pins = pin_cells(diagram, placement)
    bodies = body_cells(diagram, placement)
    outward = {
        (ci, pi): diagram.components[ci].pin(pi, placement[ci][2])[1]
        for ci, pi in pins
    }
    nets = sorted(
        (n for n in diagram.nets if not diagram.is_ground(n) and n not in diagram.named_nets),
        key=lambda n: (len(diagram.nets[n]), n),
    )
    if diagram.bounded:
        area = (0, 0, diagram.w - 1, diagram.h - 1)
    else:
        # Unbounded scene: route inside the parts' extent plus a margin.
        x0, y0, x1, y1 = extent(diagram, placement)
        area = (x0 - 4, y0 - 4, x1 + 4, y1 + 4)
    result = _route_in_order(diagram, nets, pins, bodies, area, outward)
    if result.unrouted:
        failed = set(result.unrouted)
        retry = _route_in_order(diagram, [n for n in nets if n in failed] + [n for n in nets if n not in failed], pins, bodies, area, outward)
        if (len(retry.unrouted), retry.length) < (len(result.unrouted), result.length):
            result = retry
    return result


def _route_in_order(diagram: Diagram, nets: list[str], pins, bodies, area, outward=None) -> Routing:
    result = Routing()
    for net in nets:
        # IC pins with their own supply symbol are not wired.
        members = [m for m in diagram.nets[net] if not diagram.is_power_pin(*m)]
        cells = [pins[m] for m in members if m in pins]
        result.pins[net] = set(cells)
        if len(cells) < 2:
            continue
        foreign_pins = {pins[m] for m in pins if m not in members}
        blocked = bodies | foreign_pins
        pin_dirs = {pins[m]: outward[m] for m in members if m in pins} if outward else {}
        tree = {cells[0]}
        found: list[list[Cell]] = []
        ok = True
        for target in cells[1:]:
            if target in tree:
                continue
            path = _search(target, tree, blocked, net, result.usage, area, pin_dirs)
            if path is None:
                ok = False
                break
            found.append(path)
            tree.update(path)
            _record(result.usage, net, path)
        if not ok:
            result.unrouted.append(net)
        if found:
            result.paths[net] = found
        result.usage.setdefault(cells[0], {}).setdefault(net, set())
    return result
