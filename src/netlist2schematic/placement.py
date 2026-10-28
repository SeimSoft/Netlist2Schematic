"""Placement state on the grid: anchors, rotations, occupancy and pin cells.

The grid size belongs to the diagram (``diagram.w`` x ``diagram.h``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .diagram import Diagram

# ci -> (anchor x, anchor y, rotation)
Placement = dict[int, tuple[int, int, int]]


def in_bounds(diagram: Diagram, cell: tuple[int, int]) -> bool:
    if not diagram.bounded:
        return True
    return 0 <= cell[0] < diagram.w and 0 <= cell[1] < diagram.h


def relative_cells(diagram: Diagram, ci: int, r: int):
    """Rotated (blocked, pins) offsets of a component. Blocked = body, ground symbols, labels."""
    return diagram.components[ci].relative_cells(r)


def cells_of(diagram: Diagram, ci: int, x: int, y: int, r: int):
    """Blocked and pin cells of component ``ci`` placed at ``(x, y)`` with rotation ``r``."""
    body, pins = relative_cells(diagram, ci, r)
    return [(x + dx, y + dy) for dx, dy in body], [(x + dx, y + dy) for dx, dy in pins]


def occupancy(diagram: Diagram, placement: Placement) -> np.ndarray:
    """Boolean grid marking blocked and pin cells of all placed components."""
    occ = np.zeros((diagram.w, diagram.h), dtype=bool)
    for ci, (x, y, r) in placement.items():
        body, pins = cells_of(diagram, ci, x, y, r)
        for cell in body + pins:
            if in_bounds(diagram, cell):
                occ[cell] = True
    return occ


def exits_of(diagram: Diagram, ci: int, x: int, y: int, r: int) -> list[tuple[int, int]]:
    return [(x + dx, y + dy) for dx, dy in diagram.components[ci].exit_cells(r)]


def exit_map(diagram: Diagram, placement: Placement) -> np.ndarray:
    """Boolean grid of pin exit cells: no other part may sit there (wires may)."""
    grid = np.zeros((diagram.w, diagram.h), dtype=bool)
    for ci, (x, y, r) in placement.items():
        for cell in exits_of(diagram, ci, x, y, r):
            if in_bounds(diagram, cell):
                grid[cell] = True
    return grid


def pin_cells(diagram: Diagram, placement: Placement) -> dict[tuple[int, int], tuple[int, int]]:
    """(component, pin) -> grid cell for every placed component."""
    result = {}
    for ci, (x, y, r) in placement.items():
        _, pins = cells_of(diagram, ci, x, y, r)
        for pi, cell in enumerate(pins):
            result[(ci, pi)] = cell
    return result


def body_cells(diagram: Diagram, placement: Placement) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for ci, (x, y, r) in placement.items():
        body, _ = cells_of(diagram, ci, x, y, r)
        cells.update(body)
    return cells


def fits(diagram: Diagram, placement: Placement, ci: int, x: int, y: int, r: int, ignore: set[int] | None = None) -> bool:
    """Whether component ``ci`` can sit at ``(x, y, r)`` without touching the others.

    Components in ``ignore`` are treated as absent (used when moving a group).
    """
    skip = {ci} | (ignore or set())
    others = {k: v for k, v in placement.items() if k not in skip}
    return group_fits(diagram, others, {ci: (x, y, r)})


def _solid_and_exits(diagram: Diagram, placement: Placement) -> tuple[set, set]:
    solid: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    for ci, (x, y, r) in placement.items():
        body, pins = cells_of(diagram, ci, x, y, r)
        solid.update(body)
        solid.update(pins)
        exits.update(exits_of(diagram, ci, x, y, r))
    return solid, exits


def group_conflicts(diagram: Diagram, placement: Placement, moved: Placement) -> set[int | None]:
    """Parts that block ``moved`` at its new positions (None: grid edge or disallowed orientation).

    Solid cells (body, labels, ground and supply symbols, pins) may not overlap anything
    solid or any pin exit; pin exits may not be covered by anything solid.
    Works without grid limits when ``diagram.bounded`` is False.
    """
    owner: dict[tuple[int, int], int] = {}
    exit_owner: dict[tuple[int, int], int] = {}
    for ci, (x, y, r) in placement.items():
        if ci in moved:
            continue
        body, pins = cells_of(diagram, ci, x, y, r)
        for c in body + pins:
            owner[c] = ci
        for c in exits_of(diagram, ci, x, y, r):
            exit_owner.setdefault(c, ci)
    conflicts: set[int | None] = set()
    new_solid: dict[tuple[int, int], int] = {}
    new_exits: list[tuple[tuple[int, int], int]] = []
    for ci, (x, y, r) in moved.items():
        if r not in diagram.components[ci].rotations:
            conflicts.add(None)
            continue
        body, pins = cells_of(diagram, ci, x, y, r)
        for c in body + pins:
            if not in_bounds(diagram, c):
                conflicts.add(None)
            if c in owner:
                conflicts.add(owner[c])
            if c in exit_owner:
                conflicts.add(exit_owner[c])
            if c in new_solid and new_solid[c] != ci:
                conflicts.add(new_solid[c])
            new_solid[c] = ci
        new_exits += [(c, ci) for c in exits_of(diagram, ci, x, y, r)]
    for c, ci in new_exits:
        if not in_bounds(diagram, c):
            conflicts.add(None)
        if c in owner:
            conflicts.add(owner[c])
        if c in new_solid and new_solid[c] != ci:
            conflicts.add(new_solid[c])
    return conflicts


def group_fits(diagram: Diagram, placement: Placement, moved: Placement) -> bool:
    """Whether all components in ``moved`` fit at their new positions together."""
    return not group_conflicts(diagram, placement, moved)


def nearest_fit(diagram: Diagram, placement: Placement, moved: Placement, radius: int = 3) -> Placement | None:
    """``moved`` shifted by the smallest offset (up to ``radius`` cells) that fits, or None."""
    for r in range(radius + 1):
        ring = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if max(abs(dx), abs(dy)) == r]
        for dx, dy in sorted(ring, key=lambda o: (abs(o[0]) + abs(o[1]), o)):
            shifted = {ci: (x + dx, y + dy, o) for ci, (x, y, o) in moved.items()}
            if group_fits(diagram, placement, shifted):
                return shifted
    return None


def repair(diagram: Diagram, placement: Placement, radius: int = 4) -> Placement:
    """Placement with overlapping parts moved to the nearest free spot.

    Drawings saved under older symbol rules (e.g. before supply symbols sat on the
    pins) may overlap now; later parts are moved first, the rest stays as drawn.
    Parts that cannot be moved within ``radius`` keep their position.
    """
    fixed = dict(placement)
    for ci in sorted(fixed, reverse=True):
        others = {k: v for k, v in fixed.items() if k != ci}
        if group_fits(diagram, others, {ci: fixed[ci]}):
            continue
        moved = nearest_fit(diagram, others, {ci: fixed[ci]}, radius)
        if moved is not None:
            fixed.update(moved)
    return fixed


def extent(diagram: Diagram, placement: Placement) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) over all solid, pin and exit cells."""
    solid, exits = _solid_and_exits(diagram, placement)
    cells = solid | exits
    if not cells:
        return 0, 0, 0, 0
    return min(c[0] for c in cells), min(c[1] for c in cells), max(c[0] for c in cells), max(c[1] for c in cells)


def fit_to_grid(diagram: Diagram, placement: Placement) -> tuple[Diagram, Placement]:
    """Bounded diagram plus the placement shifted into it (grid grows if needed).

    Manual editing happens on an unbounded scene; training and the policy need a
    finite grid. A placement that already lies inside the grid stays where it is
    (training and proposals then share coordinates); otherwise it is shifted so it
    starts one cell from the origin, and the grid grows to hold it.
    """
    x0, y0, x1, y1 = extent(diagram, placement)
    dx = 1 - x0 if x0 < 1 else 0
    dy = 1 - y0 if y0 < 1 else 0
    w = max(diagram.w, x1 + dx + 2)
    h = max(diagram.h, y1 + dy + 2)
    fitted = replace(diagram, w=w, h=h, bounded=True)
    return fitted, {ci: (x + dx, y + dy, r) for ci, (x, y, r) in placement.items()}
