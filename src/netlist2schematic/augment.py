"""Training data augmentation from approved placements.

Each approved placement yields extra training examples:

- connected sub-circuits: a random seed part plus neighbours grown breadth-first
  (random order) up to N parts, N random in [2, n-1]. The sub-diagram keeps the
  original symbols, net names and grid, so the human positions stay valid; the
  policy then learns the placement order and revisions for smaller situations.
- shifts: the whole placement moved inside the grid, so the policy does not tie
  a good arrangement to absolute coordinates.

Mirroring the whole circuit is deliberately not used: it would reverse the
left-to-right signal flow the score asks for.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .diagram import Diagram
from .placement import Placement, extent


def sub_diagram(diagram: Diagram, keep: set[int]) -> tuple[Diagram, dict[int, int]]:
    """Diagram with only the parts in ``keep`` (same grid); returns it and the old -> new index map."""
    order = sorted(keep)
    remap = {old: new for new, old in enumerate(order)}
    nets: dict[str, list[tuple[int, int]]] = {}
    for net, pins in diagram.nets.items():
        kept = [(remap[c], p) for c, p in pins if c in remap]
        if kept:
            nets[net] = kept
    return replace(diagram, components=[diagram.components[i] for i in order], nets=nets), remap


def connected_subset(diagram: Diagram, rng: np.random.Generator, size: int) -> set[int]:
    """``size`` parts grown from a random seed through connections (random breadth-first order)."""
    n = len(diagram.components)
    seed = int(rng.integers(n))
    chosen = [seed]
    frontier = [seed]
    while frontier and len(chosen) < size:
        current = frontier.pop(0)
        neighbours = [o for o in diagram.neighbours(current) if o not in chosen]
        rng.shuffle(neighbours)
        for o in neighbours:
            if len(chosen) >= size:
                break
            chosen.append(o)
            frontier.append(o)
    return set(chosen)


def shifted(diagram: Diagram, placement: Placement, rng: np.random.Generator) -> Placement | None:
    """The placement moved by a random offset that keeps it inside the grid (None if no room)."""
    x0, y0, x1, y1 = extent(diagram, placement)
    lo_x, hi_x = 1 - x0, diagram.w - 2 - x1
    lo_y, hi_y = 1 - y0, diagram.h - 2 - y1
    if lo_x > hi_x or lo_y > hi_y or (lo_x == hi_x == 0 and lo_y == hi_y == 0):
        return None
    dx = int(rng.integers(lo_x, hi_x + 1))
    dy = int(rng.integers(lo_y, hi_y + 1))
    if dx == 0 and dy == 0:
        return None
    return {ci: (x + dx, y + dy, o) for ci, (x, y, o) in placement.items()}


def variants(
    diagram: Diagram,
    placement: Placement,
    rng: np.random.Generator,
    subsets: int = 8,
    shifts: int = 2,
) -> list[tuple[Diagram, Placement]]:
    """Augmented (diagram, placement) pairs; the original is not included."""
    out: list[tuple[Diagram, Placement]] = []
    n = len(diagram.components)
    seen: set[frozenset[int]] = set()
    if n >= 3:
        # Seeds without signal neighbours or repeated subsets are retried (bounded).
        for _ in range(3 * subsets):
            if len(seen) >= subsets:
                break
            keep = connected_subset(diagram, rng, int(rng.integers(2, n)))
            if len(keep) < 2 or frozenset(keep) in seen:
                continue
            seen.add(frozenset(keep))
            sub, remap = sub_diagram(diagram, keep)
            out.append((sub, {remap[ci]: placement[ci] for ci in keep}))
    for _ in range(shifts):
        moved = shifted(diagram, placement, rng)
        if moved is not None:
            out.append((diagram, moved))
    return out
