"""Readability score of a placement with its routing.

The reward is a weighted sum of penalties; higher is better (0 is the ideal).
Weights are deliberately simple and live in one place so they can be tuned
against manual corrections. Scene y grows downwards.
"""

from __future__ import annotations

from .diagram import Diagram, _rot
from .placement import Placement, body_cells
from .routing import Routing, route

WEIGHTS = {
    "crossing": 3.0,  # perpendicular wire crossings
    "length": 0.05,  # total wire length in grid cells
    "bend": 0.3,  # direction changes
    "unrouted": 15.0,  # nets that could not be connected
    "flow": 2.0,  # signal ports in the wrong left-to-right order
    "ground_dir": 1.0,  # ground symbols pointing up (sideways pins get a stub and count half)
    "ground_low": 0.5,  # ground pins above the lowest ground pin (rows)
    "supply_dir": 1.0,  # supply flags not pointing up
    "supply_high": 0.5,  # supply flags below the highest one (rows)
    "area": 0.01,  # bounding box area of all bodies
}


def _ground_pins(diagram: Diagram, placement: Placement):
    """(cell y, rotated direction) of every ground pin."""
    out = []
    for ci, (x, y, r) in placement.items():
        comp = diagram.components[ci]
        for pi in comp.ground_pins:
            px, py = _rot(comp.symbol.pins[pi], r)
            out.append((y + py, _rot(comp.symbol.pin_dirs[pi], r)))
    return out


def evaluate(diagram: Diagram, placement: Placement, routing: Routing | None = None) -> tuple[float, dict[str, float]]:
    """Return (reward, details) for a complete placement."""
    if routing is None:
        routing = route(diagram, placement)

    details: dict[str, float] = {
        "crossing": float(routing.crossings),
        "length": float(routing.length),
        "bend": float(routing.bends),
        "unrouted": float(len(routing.unrouted)),
    }

    # Signal flow: port components in header order from left to right.
    xs = [placement[ci][0] for p in diagram.signal_ports if (ci := diagram.port_component(p)) is not None and ci in placement]
    details["flow"] = float(sum(1 for a, b in zip(xs, xs[1:]) if a > b))

    ground = _ground_pins(diagram, placement)
    details["ground_dir"] = float(sum(1.0 if d == (0, -1) else 0.5 if d[1] == 0 else 0.0 for _, d in ground))
    lowest = max((y for y, _ in ground), default=0)
    details["ground_low"] = float(sum(lowest - y for y, _ in ground))

    supply = [(ci, placement[ci]) for ci, c in enumerate(diagram.components) if c.power and ci in placement]
    # Flag points up when its pin (below the bar) faces down.
    details["supply_dir"] = float(sum(1 for ci, (_, _, r) in supply if _rot(diagram.components[ci].symbol.pin_dirs[0], r) != (0, 1)))
    highest = min((pos[1] for _, pos in supply), default=0)
    details["supply_high"] = float(sum(pos[1] - highest for _, pos in supply))

    cells = body_cells(diagram, placement)
    if cells:
        cx = [c[0] for c in cells]
        cy = [c[1] for c in cells]
        details["area"] = float((max(cx) - min(cx) + 1) * (max(cy) - min(cy) + 1))
    else:
        details["area"] = 0.0

    reward = -sum(WEIGHTS[k] * v for k, v in details.items())
    return reward, details
