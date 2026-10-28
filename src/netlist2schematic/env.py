"""Sequential placement environment with action masking.

Placement order (fixed per diagram; the next part is the first unplaced one in it):
1. Start with an IC (most pins first); without ICs a transformer; without
   transformers a voltage/current source; otherwise the most connected part.
2. From there breadth-first through the wired connections: first all children of
   the start part (in its pin order), then their children, and so on. Ground,
   supply and named nets are no connections here (they are drawn with symbols or
   labels, not wires), so ICs connected to an IC by name are not its children and
   start their own tree; an IC or transformer reached by a wire is a child like any
   other part.
3. When the tree is exhausted, the next start part is taken as in 1 (preferring
   one linked to what is already ordered), and so on.

Revisions: after every new part, the policy re-decides the position of up to
``MAX_REVISITS`` already placed parts that are directly connected to it (most
connections first). For a revision the part is lifted off the sheet and all its
positions are candidates again; staying where it is is always valid (and marked
by a feature), so earlier decisions can be corrected or kept.

For the current part every grid anchor and orientation (4 rotations, each
optionally mirrored) is a candidate. Candidates that overlap bodies, labels,
ground symbols, pins or pin exits, or leave the grid, are masked out, so every
reachable placement is valid by construction. The reward is only known after the
last part is placed (see ``metrics.evaluate``).

Per decision the policy gets two inputs (computed with numpy here):
- ``features``: one vector per candidate (anchor, orientation), see F_* below,
  including routing estimates (added wire length, crossings with already placed
  connections);
- ``graph_inputs``: the netlist as a graph (parts = nodes, shared signal nets =
  edges) with per-part state (placed, current, position) for the GNN in ``policy``,
  plus the candidate coordinates and the placed neighbours of the current part
  (position, orientation, node) for the pairwise relation scorer.
"""

from __future__ import annotations

import copy

import numpy as np

from .diagram import Diagram, _rot
from .graphics import ORIENTATIONS
from .metrics import evaluate
from .placement import Placement, exit_map, occupancy, pin_cells, relative_cells
from .routing import Routing, route
from .symbols import KIND_IC, KIND_PORT, KIND_SOURCE, NUM_KINDS

PAD = 12  # minimum padding; valid_mask pads by the largest symbol offset of the current part
MAX_REVISITS_PER_PART = 2  # a part is re-decided at most this often (prevents thrashing)
MAX_RELATIONS = 6  # placed neighbours the relation scorer compares each candidate with
SCALE = 20.0  # cells; fixed length scale so features do not depend on the grid size

# Feature layout (index: meaning)
F_HAS_NB, F_DX, F_DY, F_DIST, F_X, F_Y = 0, 1, 2, 3, 4, 5
F_ROT = 6  # 6..9 one-hot quarter turns
F_MIRROR = 10
F_ALIGN_ROW, F_ALIGN_COL, F_CONGESTION = 11, 12, 13
F_KIND = 14  # 14..18 one-hot symbol kind
F_GROUND_DOWN, F_GROUND_LOW, F_SUPPLY_UP, F_SUPPLY_HIGH = 19, 20, 21, 22
F_FIRST_PORT, F_LAST_PORT, F_PLACED_FRAC, F_CENTRE, F_PIN_FACING = 23, 24, 25, 26, 27
F_REVISIT, F_STAY = 28, 29  # decision re-places a placed part / candidate is its current position
F_WIRE, F_CROSS = 30, 31  # added wire length to placed partners / crossings with placed connections
N_FEATURES = 32

# Graph node features: element category one-hot, static properties, state.
NODE_CATEGORIES = ["R", "C", "L", "D", "V", "I", "CTRL", "Q", "M", "S", "IC", "K", "P"]
_CATEGORY = {"B": "CTRL", "E": "CTRL", "G": "CTRL", "F": "CTRL", "H": "CTRL", "W": "S", "X": "IC", "N": "IC"}
N_NODE_FEATURES = len(NODE_CATEGORIES) + 6 + 5
MAX_REVISITS = 3


def next_component(diagram: Diagram, placed: set[int], order: list[int] | None = None) -> int:
    """The part to place next, given the set of placed parts (see module docstring)."""
    for ci in order if order is not None else placement_order(diagram):
        if ci not in placed:
            return ci
    raise ValueError("all parts are placed")


def _wired_children(diagram: Diagram) -> list[list[tuple[int, int]]]:
    """Per part: (own pin index, other part) for every wired connection, in pin order."""
    wired: list[list[tuple[int, int]]] = [[] for _ in diagram.components]
    for net, pins in diagram.nets.items():
        if diagram.is_ground(net) or net in diagram.power_nets or net in diagram.named_nets:
            continue
        for ci, pi in pins:
            wired[ci] += [(pi, other) for other, _ in pins if other != ci]
    return [sorted(set(links)) for links in wired]


def _start_rank(comp) -> int:
    """Which parts anchor a schematic: IC 3, transformer 2, voltage/current source 1, rest 0."""
    if comp.symbol.kind == KIND_IC:
        return 3
    if comp.kind == "K":
        return 2
    if comp.symbol.kind == KIND_SOURCE:
        return 1
    return 0


def _graph(diagram: Diagram) -> tuple[np.ndarray, np.ndarray]:
    """Static node features and the row-normalised adjacency of the netlist graph.

    Two parts are linked for every signal net they share, with weight 1/(k-1) for
    a net with k parts, so large nets (rails) do not dominate.
    """
    comps = diagram.components
    n = len(comps)
    nodes = np.zeros((n, len(NODE_CATEGORIES) + 6), dtype=np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for net, pins in diagram.nets.items():
        if diagram.is_ground(net):
            continue
        members = sorted({c for c, _ in pins})
        if len(members) < 2:
            continue
        weight = 1.0 / (len(members) - 1)
        for a in members:
            for b in members:
                if a != b:
                    adj[a, b] += weight
    for ci, comp in enumerate(comps):
        category = "P" if comp.kind == "P" else _CATEGORY.get(comp.kind, comp.kind)
        if category in NODE_CATEGORIES:
            nodes[ci, NODE_CATEGORIES.index(category)] = 1.0
        k = len(NODE_CATEGORIES)
        nodes[ci, k] = len(comp.symbol.pins) / 10.0
        nodes[ci, k + 1] = (adj[ci] > 0).sum() / 10.0
        nodes[ci, k + 2] = len(comp.ground_pins) / 4.0
        nodes[ci, k + 3] = 1.0 if comp.power else 0.0
        nodes[ci, k + 4] = 1.0 if comp.named_pins else 0.0
        nodes[ci, k + 5] = 1.0 if comp.power_pins else 0.0
    rows = adj.sum(axis=1, keepdims=True)
    adj = np.divide(adj, rows, out=np.zeros_like(adj), where=rows > 0)
    return nodes, adj


def placement_order(diagram: Diagram, placed: set[int] | None = None) -> list[int]:
    """Order in which the parts are placed (see module docstring); without ``placed`` parts."""
    comps = diagram.components
    n = len(comps)
    wired = _wired_children(diagram)
    links = [len({o for _, o in w}) for w in wired]
    rank = [_start_rank(c) for c in comps]
    order: list[int] = []
    seen: set[int] = set()

    def linked(ci: int) -> int:
        return sum(1 for o in diagram.neighbours(ci) if o in seen)

    while len(order) < n:
        root = max(
            (ci for ci in range(n) if ci not in seen),
            key=lambda ci: (rank[ci], linked(ci) > 0, len(comps[ci].symbol.pins) if rank[ci] else 0, links[ci], -ci),
        )
        queue = [root]
        seen.add(root)
        while queue:
            ci = queue.pop(0)
            order.append(ci)
            for _, child in wired[ci]:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
    done = set(placed or ())
    return [ci for ci in order if ci not in done]


class PlacementEnv:
    def __init__(self, diagram: Diagram, revisits: bool = True, regions: dict[str, tuple[int, int, int, int]] | None = None):
        """``regions``: group id -> (x0, y0, x1, y1) in grid cells (``groups.floorplan``); every
        part of a group must lie inside its region (relaxed step by step if it no longer fits)."""
        if not diagram.bounded:
            raise ValueError("the environment needs a bounded diagram (see placement.fit_to_grid)")
        self.diagram = diagram
        self.revisits = revisits
        self.regions = dict(regions or {})
        self.w = diagram.w
        self.h = diagram.h
        self._neighbours = [diagram.neighbours(ci) for ci in range(len(diagram.components))]
        self._order = placement_order(diagram)
        self._static_nodes, self._adjacency = _graph(diagram)
        # Candidate table: index -> (x, y, orientation); index = (x * h + y) * 8 + o.
        self.anchors = np.array([(x, y, o) for x in range(self.w) for y in range(self.h) for o in range(ORIENTATIONS)])
        self.reset()

    @property
    def num_candidates(self) -> int:
        return len(self.anchors)

    def reset(self, prefix: Placement | None = None) -> None:
        """Start empty, or from parts that are already placed (e.g. by hand).

        Parts of the prefix are frozen: revisions never move them, so manual
        corrections survive when the future is regenerated.
        """
        self.placement: Placement = dict(prefix or {})
        self.frozen: set[int] = set(self.placement)
        self.revisit_count: dict[int, int] = {}
        self.anchor: int | None = None  # first placed part: never revisited
        self._current: int | None = None
        self._pending: list[int] = []  # placed parts to re-decide next

    @property
    def done(self) -> bool:
        return len(self.placement) >= len(self.diagram.components) and not self._pending

    @property
    def revisiting(self) -> bool:
        """The current decision re-places a part that is already on the sheet."""
        return bool(self._pending)

    @property
    def current(self) -> int:
        if self._pending:
            return self._pending[0]
        if self._current is None:
            self._current = next_component(self.diagram, set(self.placement), self._order)
        return self._current

    def _base(self) -> Placement:
        """The other parts: the placement without the part being decided."""
        if self._pending:
            return {k: v for k, v in self.placement.items() if k != self._pending[0]}
        return self.placement

    def _revisit_targets(self, ci: int) -> list[int]:
        placed = [
            (c, o)
            for o, c in self._neighbours[ci].items()
            if o in self.placement
            and o != ci
            and o not in self.frozen
            and o != self.anchor
            and self.revisit_count.get(o, 0) < MAX_REVISITS_PER_PART
        ]
        placed.sort(key=lambda t: (-t[0], t[1]))
        return [o for _, o in placed[:MAX_REVISITS]]

    def clone(self) -> "PlacementEnv":
        """Independent copy of the current state (static data is shared)."""
        other = copy.copy(self)
        other.placement = dict(self.placement)
        other.frozen = set(self.frozen)
        other.revisit_count = dict(self.revisit_count)
        other._pending = list(self._pending)
        return other

    def index_of(self, x: int, y: int, o: int) -> int:
        return (x * self.h + y) * ORIENTATIONS + o

    def valid_mask(self, occ: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask over all candidates for the current component.

        Solid cells may not hit anything solid or a pin exit of a placed part;
        the new part's pin exits may not be covered by anything solid.
        """
        base = self._base()
        if occ is None:
            occ = occupancy(self.diagram, base)
        ci = self.current
        comp = self.diagram.components[ci]
        # Pad with blocked cells so anything outside the grid counts as blocked.
        cells = [c for o in comp.rotations for part in relative_cells(self.diagram, ci, o) for c in part]
        cells += [c for o in comp.rotations for c in comp.exit_cells(o)]
        pad = max([PAD] + [abs(v) + 1 for c in cells for v in c])
        solid = np.pad(occ, pad, constant_values=True)
        hard = np.pad(occ | exit_map(self.diagram, base), pad, constant_values=True)
        valid = np.zeros((self.w, self.h, ORIENTATIONS), dtype=bool)
        xs, ys = np.meshgrid(np.arange(self.w), np.arange(self.h), indexing="ij")
        xs, ys = xs.reshape(-1)[:, None], ys.reshape(-1)[:, None]
        for o in comp.rotations:
            body, pins = relative_cells(self.diagram, ci, o)
            own = np.array(body + pins)
            blocked = hard[xs + own[None, :, 0] + pad, ys + own[None, :, 1] + pad].any(axis=1)
            exits = comp.exit_cells(o)
            if exits:
                ex = np.array(exits)
                blocked |= solid[xs + ex[None, :, 0] + pad, ys + ex[None, :, 1] + pad].any(axis=1)
            valid[:, :, o] = (~blocked).reshape(self.w, self.h)
        region = self.regions.get(comp.group)
        if region is not None:
            valid = self._in_region(ci, valid, region)
        return valid.reshape(-1)

    REGION_RELAX = (0, 3, 8)  # cells the region grows by when the part does not fit (then: anywhere)

    def _in_region(self, ci: int, valid: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        """``valid`` limited to anchors that keep every reserved cell of the part inside the
        region; grown step by step (finally unrestricted) if no such candidate is left."""
        comp = self.diagram.components[ci]
        x0, y0, x1, y1 = region
        for grow in self.REGION_RELAX:
            inside = np.zeros_like(valid)
            for o in comp.rotations:
                body, pins = relative_cells(self.diagram, ci, o)
                cells = body + pins + comp.exit_cells(o)
                ax0 = max(0, x0 - grow - min(c[0] for c in cells))
                ax1 = min(self.w - 1, x1 + grow - max(c[0] for c in cells))
                ay0 = max(0, y0 - grow - min(c[1] for c in cells))
                ay1 = min(self.h - 1, y1 + grow - max(c[1] for c in cells))
                if ax0 <= ax1 and ay0 <= ay1:
                    inside[ax0 : ax1 + 1, ay0 : ay1 + 1, o] = valid[ax0 : ax1 + 1, ay0 : ay1 + 1, o]
            if inside.any():
                return inside
        return valid

    def features(self, occ: np.ndarray) -> np.ndarray:
        """(num_candidates, N_FEATURES) features for the current component."""
        d = self.diagram
        ci = self.current
        comp = d.components[ci]
        w, h = self.w, self.h
        xs = self.anchors[:, 0].astype(float)
        ys = self.anchors[:, 1].astype(float)
        os_ = self.anchors[:, 2]

        # All features are translation invariant and independent of the grid size:
        # positions are measured from the centroid of the placed parts (from the grid
        # centre for the very first part), lengths in units of SCALE cells.
        base = self._base()
        if base:
            pcx = float(np.mean([p[0] for p in base.values()]))
            pcy = float(np.mean([p[1] for p in base.values()]))
        else:
            pcx, pcy = w / 2, h / 2
        rel_x = (xs - pcx) / SCALE
        rel_y = (ys - pcy) / SCALE

        # Placed neighbours: connection-weighted centroid of their anchors.
        nbs = self._neighbours[ci]
        total = sum(nbs.values())
        placed = {o: c for o, c in nbs.items() if o in self.placement}
        placed_total = sum(placed.values())
        has = 1.0 if placed_total else 0.0
        if placed_total:
            cx = sum(self.placement[o][0] * c for o, c in placed.items()) / placed_total
            cy = sum(self.placement[o][1] * c for o, c in placed.items()) / placed_total
        else:
            cx = cy = 0.0
        dx = (xs - cx) * has
        dy = (ys - cy) * has

        # Local congestion: occupied fraction in a 5x5 window.
        padded = np.pad(occ.astype(float), 2)
        cong = np.zeros((w, h))
        for ox in range(5):
            for oy in range(5):
                cong += padded[ox : ox + w, oy : oy + h]
        cong /= 25.0

        f = np.zeros((self.num_candidates, N_FEATURES))
        f[:, F_HAS_NB] = has
        f[:, F_DX] = np.clip(dx / SCALE, -3, 3)
        f[:, F_DY] = np.clip(dy / SCALE, -3, 3)
        f[:, F_DIST] = np.clip((np.abs(dx) + np.abs(dy)) / SCALE, 0, 6)
        f[:, F_X] = np.tanh(rel_x)
        f[:, F_Y] = np.tanh(rel_y)
        f[np.arange(self.num_candidates), F_ROT + os_ % 4] = 1.0
        f[:, F_MIRROR] = (os_ >= 4).astype(float)
        f[:, F_ALIGN_ROW] = (dy == 0) * has
        f[:, F_ALIGN_COL] = (dx == 0) * has
        f[:, F_CONGESTION] = cong[xs.astype(int), ys.astype(int)]
        f[:, F_KIND + comp.symbol.kind] = 1.0

        # Ground pins should face down and sit low; supply flags point up and sit high.
        sym = comp.symbol
        if comp.ground_pins:
            # 1 = symbol directly below the pin, 0.5 = sideways stub, 0 = pointing up.
            score = {(0, 1): 1.0, (0, -1): 0.0}
            down = np.array([np.mean([score.get(_rot(sym.pin_dirs[p], o), 0.5) for p in comp.ground_pins]) for o in range(ORIENTATIONS)])
            f[:, F_GROUND_DOWN] = down[os_]
            f[:, F_GROUND_LOW] = np.tanh(rel_y)  # below the rest of the circuit
        if comp.power:
            up = np.array([float(_rot(sym.pin_dirs[0], o) == (0, 1)) for o in range(ORIENTATIONS)])
            f[:, F_SUPPLY_UP] = up[os_]
            f[:, F_SUPPLY_HIGH] = np.tanh(-rel_y)
        elif comp.power_pins:
            # Supply symbol directly at the pin: 1 = pin (and symbol) up, 0.5 = sideways, 0 = down.
            score = {(0, -1): 1.0, (0, 1): 0.0}
            up = np.array([np.mean([score.get(_rot(sym.pin_dirs[p], o), 0.5) for p, _ in comp.power_pins]) for o in range(ORIENTATIONS)])
            f[:, F_SUPPLY_UP] = up[os_]

        ports = d.signal_ports
        first = bool(ports) and (comp.name == f"port:{ports[0]}" or any(c == ci for c, _ in d.nets.get(ports[0], [])))
        last = len(ports) > 1 and (comp.name == f"port:{ports[-1]}" or any(c == ci for c, _ in d.nets.get(ports[-1], [])))
        if comp.symbol.kind == KIND_PORT and comp.power:
            first = last = False  # supply flags go up, not left/right
        # Saturating, so a convention never pushes a part arbitrarily far.
        f[:, F_FIRST_PORT] = float(first) * np.tanh(rel_x)
        f[:, F_LAST_PORT] = float(last) * np.tanh(-rel_x)
        f[:, F_PLACED_FRAC] = (placed_total / total) if total else 0.0
        f[:, F_CENTRE] = np.clip(np.abs(rel_x) + np.abs(rel_y), 0, 6)
        f[:, F_PIN_FACING] = self._pin_facing(ci, xs, ys, os_)
        f[:, F_WIRE], f[:, F_CROSS] = self._routing_estimates(ci, xs, ys, os_)
        if self.revisiting:
            f[:, F_REVISIT] = 1.0
            f[self.index_of(*self.placement[ci]), F_STAY] = 1.0
        return f

    def _pin_facing(self, ci: int, xs: np.ndarray, ys: np.ndarray, os_: np.ndarray) -> np.ndarray:
        """Mean cosine between each pin's outward direction and its placed net partners."""
        d = self.diagram
        sym = d.components[ci].symbol
        placed_pins = pin_cells(d, self._base())
        targets = []
        for pi, node in enumerate(sym.pin_nodes):
            if not node or d.is_ground(node) or node in d.named_nets:
                continue
            cells = [placed_pins[m] for m in d.nets.get(node, []) if m[0] != ci and m in placed_pins]
            if cells:
                targets.append((pi, np.mean([c[0] for c in cells]), np.mean([c[1] for c in cells])))
        out = np.zeros(len(xs))
        if not targets:
            return out
        for o in range(ORIENTATIONS):
            sel = os_ == o
            acc = np.zeros(sel.sum())
            for pi, tx, ty in targets:
                ox, oy = _rot(sym.pins[pi], o)
                dx, dy = _rot(sym.pin_dirs[pi], o)
                vx = tx - (xs[sel] + ox)
                vy = ty - (ys[sel] + oy)
                acc += (dx * vx + dy * vy) / (np.hypot(vx, vy) + 1e-6)
            out[sel] = acc / len(targets)
        return out

    def _partner_pins(self, ci: int) -> list[tuple[int, str, np.ndarray]]:
        """(pin, net, placed partner pin cells) for every wired pin of ``ci``."""
        d = self.diagram
        placed_pins = pin_cells(d, self._base())
        out = []
        for pi, node in enumerate(d.components[ci].symbol.pin_nodes):
            if not node or d.is_ground(node) or node in d.named_nets or d.is_power_pin(ci, pi):
                continue
            cells = [placed_pins[m] for m in d.nets.get(node, []) if m[0] != ci and m in placed_pins and not d.is_power_pin(*m)]
            if cells:
                out.append((pi, node, np.array(cells, dtype=float)))
        return out

    def _routing_estimates(self, ci: int, xs: np.ndarray, ys: np.ndarray, os_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Added wire length and estimated crossings for every candidate.

        Wire: per pin the Manhattan distance to the nearest placed partner pin
        (orientation-aware), summed, in units of SCALE cells.
        Crossings: straight "rat lines" from the candidate anchor to each partner
        net's centroid, counted against the rat lines of the other placed nets
        (star from each net's first pin). Orientation-independent; divided by 5.
        """
        d = self.diagram
        n = len(xs)
        targets = self._partner_pins(ci)
        wire = np.zeros(n)
        cross = np.zeros(n)
        if not targets:
            return wire, cross
        sym = d.components[ci].symbol
        for o in range(ORIENTATIONS):
            sel = os_ == o
            for pi, _, cells in targets:
                ox, oy = _rot(sym.pins[pi], o)
                px = xs[sel][:, None] + ox
                py = ys[sel][:, None] + oy
                wire[sel] += (np.abs(px - cells[None, :, 0]) + np.abs(py - cells[None, :, 1])).min(axis=1)
        wire /= SCALE

        # Rat lines of the placed parts, per net (excluding nets of the current part).
        own_nets = {net for _, net, _ in targets}
        placed_pins = pin_cells(d, self._base())
        segments = []
        for net, members in d.nets.items():
            if net in own_nets or d.is_ground(net) or net in d.named_nets:
                continue
            cells = [placed_pins[m] for m in members if m in placed_pins and not d.is_power_pin(*m)]
            for c in cells[1:]:
                segments.append((*cells[0], *c))
        if segments:
            seg = np.array(segments, dtype=float)
            e1, e2 = seg[:, :2], seg[:, 2:]
            ax = np.arange(self.w, dtype=float).repeat(self.h)
            ay = np.tile(np.arange(self.h, dtype=float), self.w)
            p1 = np.stack([ax, ay], axis=1)  # one per anchor cell
            counts = np.zeros(len(p1))
            ed = e2 - e1

            def crossz(a, b):
                return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

            for _, _, cells in targets:
                p2 = cells.mean(axis=0)
                d1 = crossz(ed[None, :, :], p1[:, None, :] - e1[None, :, :])
                d2 = crossz(ed, p2[None, :] - e1)[None, :]
                pd = p2[None, :] - p1
                d3 = crossz(pd[:, None, :], e1[None, :, :] - p1[:, None, :])
                d4 = crossz(pd[:, None, :], e2[None, :, :] - p1[:, None, :])
                counts += ((d1 * d2 < 0) & (d3 * d4 < 0)).sum(axis=1)
            # Candidate index = (x * h + y) * 8 + o: repeat each anchor cell for all orientations.
            cross = counts.repeat(ORIENTATIONS) / 5.0
        return wire, cross

    def graph_inputs(self) -> tuple:
        """(node features, row-normalised adjacency, current part, candidate (x, y, o),
        placed neighbours (x, y, o, node)) for the policy."""
        n = len(self.diagram.components)
        state = np.zeros((n, 5), dtype=np.float32)
        base = self._base()
        if base:
            pcx = np.mean([p[0] for p in base.values()])
            pcy = np.mean([p[1] for p in base.values()])
        for ci, (x, y, o) in base.items():
            state[ci, 0] = 1.0  # placed
            # Position relative to the placed parts' centroid: translation invariant.
            state[ci, 3] = np.tanh((x - pcx) / SCALE)
            state[ci, 4] = np.tanh((y - pcy) / SCALE)
        state[self.current, 1] = 1.0
        state[self.current, 2] = 1.0 if self.revisiting else 0.0
        # Placed neighbours of the current part, most connected first.
        links = sorted(((c, o) for o, c in self._neighbours[self.current].items() if o in base), key=lambda t: (-t[0], t[1]))
        neighbours = np.array([(*base[o], o) for _, o in links[:MAX_RELATIONS]], dtype=np.int32).reshape(-1, 4)
        nodes = np.concatenate([self._static_nodes, state], axis=1)
        return nodes, self._adjacency, self.current, self.anchors, neighbours

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """Features and validity mask for the current step."""
        occ = occupancy(self.diagram, self._base())
        return self.features(occ), self.valid_mask(occ)

    def step(self, index: int) -> None:
        x, y, o = (int(v) for v in self.anchors[index])
        ci = self.current
        if self._pending:
            self._pending.pop(0)
            self.placement[ci] = (x, y, o)
            self.revisit_count[ci] = self.revisit_count.get(ci, 0) + 1
        else:
            if not self.placement:
                self.anchor = ci
            self.placement[ci] = (x, y, o)
            self._pending = self._revisit_targets(ci) if self.revisits else []
        self._current = None

    def demo_steps(
        self,
        named: dict[str, tuple[int, int, int]],
        rng: np.random.Generator | None = None,
        perturb: float = 0.0,
    ) -> list:
        """Replay a complete placement given by component name and return the training steps
        (candidate features, valid mask, target index, graph inputs).

        With ``perturb`` > 0, that share of the revisions is replayed from a displaced
        position (random shift of up to 3 cells, sometimes another orientation) while
        the target stays the human position, so the policy learns to correct a badly
        placed part instead of always keeping it.
        """
        self.reset()
        steps = []
        while not self.done:
            ci = self.current
            name = self.diagram.components[ci].name
            if name not in named:
                raise ValueError(f"no placement for component {name!r}")
            x, y, o = named[name]
            if not (0 <= x < self.w and 0 <= y < self.h) or not 0 <= o < ORIENTATIONS:
                raise ValueError(f"placement of {name!r} at {(x, y, o)} is outside the grid")
            index = self.index_of(x, y, o)
            features, valid = self.observe()
            if not valid[index]:
                raise ValueError(f"placement of {name!r} at {(x, y, o)} is not valid")
            if self.revisiting and rng is not None and rng.random() < perturb:
                displaced = self._displaced(ci, valid, rng)
                if displaced is not None:
                    # Only the "current position" changes for a revision (the part is lifted off).
                    features[:, F_STAY] = 0.0
                    features[displaced, F_STAY] = 1.0
            steps.append((features, valid, index, self.graph_inputs()))
            self.step(index)
        return steps

    def _displaced(self, ci: int, valid: np.ndarray, rng: np.random.Generator) -> int | None:
        """A valid candidate near the current position of ``ci`` (shift, maybe reorient)."""
        x, y, o = self.placement[ci]
        allowed = self.diagram.components[ci].rotations
        for _ in range(10):
            dx, dy = (int(v) for v in rng.integers(-3, 4, size=2))
            no = int(rng.choice(allowed)) if rng.random() < 0.3 else o
            nx, ny = x + dx, y + dy
            if (dx, dy, no) == (0, 0, o) or not (0 <= nx < self.w and 0 <= ny < self.h):
                continue
            index = self.index_of(nx, ny, no)
            if valid[index]:
                return index
        return None

    def named_placement(self) -> dict[str, tuple[int, int, int]]:
        return {self.diagram.components[ci].name: pos for ci, pos in self.placement.items()}

    def evaluate(self) -> tuple[float, dict[str, float], Routing]:
        routing = route(self.diagram, self.placement)
        reward, details = evaluate(self.diagram, self.placement, routing)
        return reward, details, routing


assert F_KIND + NUM_KINDS == F_GROUND_DOWN
