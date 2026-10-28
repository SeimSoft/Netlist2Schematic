"""Diagram model: components and nets of one subcircuit.

Besides the netlist elements, every non-ground port of the subcircuit becomes a
placeable port component (a power flag for supply-like names, a signal label
otherwise). Ground nets are never routed: each ground pin gets its own ground
symbol, which also reserves the cell in front of the pin.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from . import symbols as S
from .graphics import Prim, Text, rotate_point
from .netlist import Element, Group, Netlist, Part, Subckt, ground_name, is_ground, is_ground_rail, is_rail, signal_direction

# Supply nets (pins get supply symbols, no wires): v3v3, v12, v5, vin, vdd*, vcc*, vbat*, hv*, +*.
# Other v-names (vout, vref, vfb, vp, ...) are signals.
_POWER = re.compile(r"^(v\d|vin$|hv|\+|.*vcc|.*vdd|.*vbat)", re.I)
MAX_LABEL = 22
LINE_GAP = 0.45

# subckt name (lower) -> port name (lower) -> package pin number
PinMaps = dict[str, dict[str, int]]


def is_power(net: str) -> bool:
    """Supply net: upper-case rail by convention, or a legacy supply name; never a suffixed signal."""
    if signal_direction(net) is not None:
        return False
    return is_rail(net) or bool(_POWER.match(net))


def pin_side(name: str) -> str | None:
    """Side of an IC / block pin from its port name: supplies top, grounds bottom, inputs left, outputs right."""
    if is_ground_rail(name):
        return "bottom"
    if is_rail(name):
        return "top"
    return {"in": "left", "out": "right"}.get(signal_direction(name) or "")


@dataclass(frozen=True)
class Component:
    name: str
    kind: str  # element letter, "P" for ports, "K" for transformers
    symbol: S.Symbol
    label: tuple[str, ...] = ()
    title: str = ""  # full description for tooltips
    ground_pins: frozenset[int] = frozenset()
    power: bool = False
    rotations: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)  # allowed orientations (4-7 mirrored)
    named_pins: tuple[tuple[int, str], ...] = ()  # pins connected by name: (pin index, net)
    power_pins: tuple[tuple[int, str], ...] = ()  # pins with their own supply symbol
    attached: tuple[tuple[int, str, tuple[str, ...], str], ...] = ()  # (pin, R/C/L, label, ground or supply net)
    members: tuple[str, ...] = ()  # netlist elements drawn as this one symbol (transformer, bridge)
    subckt: str = ""  # design subcircuit this block instantiates (opens in its own tab)
    group: str = ""  # id of the functional group (``*@groups`` header) this part belongs to
    _cache: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def pin(self, pi: int, rotation: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Rotated (offset, outward direction) of pin ``pi``."""
        return _rot(self.symbol.pins[pi], rotation), _rot(self.symbol.pin_dirs[pi], rotation)

    def ground_layout(self, rotation: int) -> list[tuple[int, str]]:
        """How each ground symbol is drawn: (pin, style).

        "down": pin faces down, compact symbol right at the pin. "stub": sideways pin,
        one-cell stub outwards with the symbol hanging below its end (inside that
        cell, so a pin right below stays connectable). "out": the symbol continues
        in the pin direction (upward pins, or no room for the stub).
        """
        key = ("ground", rotation)
        if key not in self._cache:
            own = {_rot(o, rotation) for o in self.symbol.pins}
            own.update(self.exit_cells(rotation))
            own.update(self._net_label_cells(rotation))
            own.update(self._power_cells(rotation))
            own.update(self._attached_cells(rotation))
            layout = []
            for pi in sorted(self.ground_pins):
                (px, py), (dx, dy) = self.pin(pi, rotation)
                if dy == 1:
                    style = "down"
                elif dy == 0 and (px + dx, py) not in own:
                    style = "stub"
                    own.add((px + dx, py))
                else:
                    style = "out"
                layout.append((pi, style))
            self._cache[key] = layout
        return self._cache[key]

    def glyph_cells(self, rotation: int) -> list[tuple[int, int]]:
        """Cells reserved for ground symbols beyond the pin cell (only the stub cell)."""
        cells = []
        for pi, style in self.ground_layout(rotation):
            if style == "stub":
                (px, py), (dx, _) = self.pin(pi, rotation)
                cells.append((px + dx, py))
        return cells

    def power_prims(self, rotation: int) -> list[Prim]:
        prims: list[Prim] = []
        for pi, net in self.power_pins:
            (px, py), (dx, dy) = self.pin(pi, rotation)
            prims += S.power_glyph(px, py, dx, dy, net)
        return prims

    def attached_prims(self, rotation: int) -> list[Prim]:
        """Two-terminal parts hanging directly on IC pins, each ending in a ground or supply symbol."""
        prims: list[Prim] = []
        for pi, kind, label, end in self.attached:
            (px, py), (dx, dy) = self.pin(pi, rotation)
            prims += S.passive_glyph(kind, px, py, dx, dy, label)
            reach = S.PASSIVE_LEAD + S.PASSIVE_BODY
            ex, ey = px + reach * dx, py + reach * dy
            if is_ground(end):
                prims += S.ground_glyph(ex, ey, dx, dy, ground_name(end))
            else:
                prims += S.power_glyph(ex, ey, dx, dy, end)
        return prims

    def _attached_cells(self, rotation: int) -> list[tuple[int, int]]:
        cells = []
        for pi, _, _, end in self.attached:
            (px, py), (dx, dy) = self.pin(pi, rotation)
            supply = None if is_ground(end) else end
            reach = S.passive_reach(supply, horizontal=dy == 0)
            for k in range(1, math.ceil(reach + 0.5) + 1):
                cells.append((px + k * dx, py + k * dy))
        return cells

    def _power_cells(self, rotation: int) -> list[tuple[int, int]]:
        cells = []
        for pi, net in self.power_pins:
            (px, py), (dx, dy) = self.pin(pi, rotation)
            reach = S.power_glyph_length(net, horizontal=dy == 0)
            for k in range(1, math.ceil(reach + 0.5) + 1):
                cells.append((px + k * dx, py + k * dy))
        return cells

    def net_label_prims(self, rotation: int) -> list[Prim]:
        prims: list[Prim] = []
        for pi, net in self.named_pins:
            (px, py), (dx, dy) = self.pin(pi, rotation)
            prims += S.net_label(px, py, dx, dy, net)
        return prims

    def _net_label_cells(self, rotation: int) -> list[tuple[int, int]]:
        cells = []
        for pi, net in self.named_pins:
            (px, py), (dx, dy) = self.pin(pi, rotation)
            for k in range(1, math.ceil(0.15 + S.net_label_length(net) + 0.5)):
                cells.append((px + k * dx, py + k * dy))
        return cells

    def label_prims(self, rotation: int) -> list[Prim]:
        """Name/value text placed next to the rotated body."""
        lines = [t for t in self.label if t]
        if not lines or self.symbol.kind == S.KIND_PORT:
            return []
        body = [_rot(c, rotation) for c in self.symbol.body]
        x0 = min(c[0] for c in body) - 0.5
        x1 = max(c[0] for c in body) + 0.5
        y0 = min(c[1] for c in body) - 0.5
        y1 = max(c[1] for c in body) + 0.5
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        n = len(lines)
        size = S.LABEL_SIZE
        if self.kind == "D" and len(self.symbol.pins) == 4:
            # Bridge rectifier: pins point in all four directions, the corner between
            # two of them (top right, beside the upper right diode) stays free.
            return [
                Text(1.1, -1.65 + LINE_GAP * i, t, anchor="start", size=size, title=self.title) for i, t in enumerate(lines)
            ]
        if self.symbol.kind == S.KIND_IC or self.kind == "K":
            # Pins on both sides: name above, value below; beyond the supply / ground symbols
            # of top / bottom pins.
            pin_ys = [_rot(p, rotation)[1] for p in self.symbol.pins]
            if pin_ys and min(pin_ys) < y0:
                y0 = min(pin_ys) - 1.1
            if pin_ys and max(pin_ys) > y1:
                y1 = max(pin_ys) + 0.9
            out = [Text(cx, y0 - 0.35, lines[0], size=size, title=self.title)]
            if n > 1:
                out.append(Text(cx, y1 + 0.35, lines[1], size=size * 0.9, title=self.title))
            return out
        if (x1 - x0) > (y1 - y0):
            # Horizontal part: label above the body.
            return [Text(cx, y0 - 0.2 - LINE_GAP * (n - 1 - i), t, size=size, title=self.title) for i, t in enumerate(lines)]
        # Vertical or square part: label to the right, unless a pin leaves to the right and
        # none to the left (e.g. a transistor mirrored so its base points right): then left.
        dirs = {_rot(d, rotation) for d in self.symbol.pin_dirs}
        if (1, 0) in dirs and (-1, 0) not in dirs:
            return [
                Text(x0 - 0.2, cy + (i - (n - 1) / 2) * LINE_GAP, t, anchor="end", size=size, title=self.title)
                for i, t in enumerate(lines)
            ]
        return [
            Text(x1 + 0.2, cy + (i - (n - 1) / 2) * LINE_GAP, t, anchor="start", size=size, title=self.title)
            for i, t in enumerate(lines)
        ]

    def exit_cells(self, rotation: int) -> list[tuple[int, int]]:
        """Cell in front of every non-ground pin. Kept free of other parts so wires can leave."""
        key = ("exits", rotation)
        if key not in self._cache:
            exits = []
            named = (
                {pi for pi, _ in self.named_pins}
                | {pi for pi, _ in self.power_pins}
                | {pi for pi, *_ in self.attached}
            )
            for pi in range(len(self.symbol.pins)):
                if pi in self.ground_pins or pi in named:
                    continue
                (px, py), (dx, dy) = self.pin(pi, rotation)
                exits.append((px + dx, py + dy))
            self._cache[key] = exits
        return self._cache[key]

    def relative_cells(self, rotation: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """(blocked, pins) offsets: body, ground symbols and net-label stubs are blocked; text labels are not."""
        key = ("cells", rotation)
        if key not in self._cache:
            blocked = {_rot(o, rotation) for o in self.symbol.body}
            blocked.update(self.glyph_cells(rotation))
            blocked.update(self._net_label_cells(rotation))
            blocked.update(self._power_cells(rotation))
            blocked.update(self._attached_cells(rotation))
            pins = [_rot(o, rotation) for o in self.symbol.pins]
            blocked.difference_update(pins)
            self._cache[key] = (sorted(blocked), pins)
        return self._cache[key]


def _rot(p, r):
    x, y = rotate_point(p[0], p[1], r)
    return int(round(x)), int(round(y))


_PREFIX = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}


def pretty_value(value: str) -> str:
    """Engineering notation for plain numbers: 1.02e+06 -> 1.02M, 2.2e-09 -> 2.2n."""
    try:
        number = float(value)
    except ValueError:
        return value
    if number == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(number)) / 3) * 3)
    exp = max(-15, min(9, exp))
    mantissa = number / 10**exp
    text = f"{mantissa:.3g}"
    return f"{text}{_PREFIX[exp]}"


@dataclass
class Diagram:
    circuit: str
    ports: tuple[str, ...]
    components: list[Component]
    # net name -> list of (component index, pin index)
    nets: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    w: int = 20
    h: int = 14
    named_nets: frozenset[str] = frozenset()  # nets drawn as labels instead of wires
    power_nets: frozenset[str] = frozenset()  # supply ports; IC pins on them get supply symbols
    bounded: bool = True  # False while editing: no grid limits (the scene is unbounded)
    groups: tuple[Group, ...] = ()  # functional groups in header order (+ REST for ungrouped parts)
    description: str = ""  # the subcircuit's description from its header
    parts: dict[str, tuple[Part, ...]] = field(default_factory=dict)  # instance -> board parts (``*@parts``)
    pad_nets: dict[str, dict[str, str]] = field(default_factory=dict)  # part ref -> pad -> net

    def parts_of(self, ci: int) -> list[Part]:
        """Board parts that realise component ``ci`` (its element and the elements it combines)."""
        comp = self.components[ci]
        found: dict[str, Part] = {}
        for name in (comp.name, *comp.members):
            for part in self.parts.get(name, ()):
                found.setdefault(part.ref, part)
        return list(found.values())

    def group(self, gid: str) -> Group | None:
        return next((g for g in self.groups if g.id == gid), None)

    def group_members(self, gid: str) -> list[int]:
        return [ci for ci, comp in enumerate(self.components) if comp.group == gid]

    @property
    def signal_ports(self) -> list[str]:
        return [p for p in self.ports if not is_ground(p)]

    def is_ground(self, net: str) -> bool:
        return is_ground(net)

    def component_index(self, name: str) -> int:
        for i, comp in enumerate(self.components):
            if comp.name == name:
                return i
        raise KeyError(name)

    def component_of(self, element: str) -> int | None:
        """Component that draws netlist element ``element``: itself, the transformer or bridge
        it belongs to, or the IC a two-terminal part is attached to."""
        key = element.lower()
        for i, comp in enumerate(self.components):
            if comp.kind == "P":
                continue
            names = {comp.name.lower(), *(m.lower() for m in comp.members), *(a[2][0].lower() for a in comp.attached if a[2])}
            if key in names:
                return i
        return None

    def is_power_pin(self, ci: int, pi: int) -> bool:
        return any(p == pi for p, _ in self.components[ci].power_pins)

    def port_component(self, net: str) -> int | None:
        try:
            return self.component_index(f"port:{net}")
        except KeyError:
            return None

    def neighbours(self, ci: int) -> dict[int, int]:
        """Other components sharing a non-ground net with ``ci``, with shared net counts."""
        counts: dict[int, int] = {}
        for net, pins in self.nets.items():
            if is_ground(net):
                continue
            members = {c for c, _ in pins}
            if ci not in members:
                continue
            for other in members - {ci}:
                counts[other] = counts.get(other, 0) + 1
        return counts


PASSIVE_KINDS = ("R", "C", "L")


def _attach_two_terminals(components: list[Component], power_nets: frozenset[str]) -> list[Component]:
    """Move R, C, L parts that are the only connection of an IC pin onto that pin.

    The part must go to ground or a supply net. The IC pin's net must have exactly these two
    members, otherwise the part stays a normal component and gets routed.
    """
    members: dict[str, list[tuple[int, int]]] = {}
    for ci, comp in enumerate(components):
        for pi, node in enumerate(comp.symbol.pin_nodes):
            if node:
                members.setdefault(node, []).append((ci, pi))
    drop: set[int] = set()
    attach: dict[int, list[tuple[int, str, tuple[str, ...], str]]] = {}
    for net, pins in members.items():
        if len(pins) != 2 or is_ground(net) or net in power_nets:
            continue
        (a, ap), (b, bp) = pins
        if components[a].symbol.kind == S.KIND_IC and components[b].kind in PASSIVE_KINDS:
            ic, ip, two, tp = a, ap, b, bp
        elif components[b].symbol.kind == S.KIND_IC and components[a].kind in PASSIVE_KINDS:
            ic, ip, two, tp = b, bp, a, ap
        else:
            continue
        nodes = components[two].symbol.pin_nodes
        if two in drop or len(nodes) != 2:
            continue
        other = nodes[1 - tp]
        if not (is_ground(other) or other in power_nets):
            continue
        drop.add(two)
        attach.setdefault(ic, []).append((ip, components[two].kind, components[two].label, other))
    if not attach:
        return components
    result = []
    for ci, comp in enumerate(components):
        if ci in drop:
            continue
        if ci in attach:
            comp = replace(comp, attached=tuple(attach[ci]), _cache={})
        result.append(comp)
    return result


def _short(text: str) -> str:
    return text if len(text) <= MAX_LABEL else text[: MAX_LABEL - 1] + "…"


def own_symbol(ports: tuple[str, ...] | list[str]) -> S.Symbol | None:
    """The circuit's own block symbol, as a parent sheet draws it: one pin per port, placed
    by the naming convention (inputs left, outputs right, supplies top, grounds bottom)."""
    names = list(ports)
    if not names:
        return None
    sides = [pin_side(n) for n in names]
    if any(sides):
        return S.directional_ic(names, names, sides, block=True)
    return S.ic(names, names, block=True)


def _ic_pins(element: Element, child: Subckt | None, pin_map: dict[str, int] | None):
    """Package-ordered (nodes, names) for an IC box."""
    nodes = list(element.nodes)
    names = list(child.ports) if child is not None and len(child.ports) == len(nodes) else [""] * len(nodes)
    if not pin_map:
        return nodes, names
    count = max(max(pin_map.values()), len(nodes))
    out_nodes = [""] * count
    out_names = [""] * count
    leftovers = []
    for node, name in zip(nodes, names):
        number = pin_map.get(name.lower())
        if number and not out_nodes[number - 1]:
            out_nodes[number - 1], out_names[number - 1] = node, name
        else:
            leftovers.append((node, name))
    free = [i for i in range(count) if not out_nodes[i]]
    for (node, name), i in zip(leftovers, free):
        out_nodes[i], out_names[i] = node, name
    for port, number in pin_map.items():
        if not out_names[number - 1]:
            out_names[number - 1] = "NC" if port.lower().startswith("nc") else ""
    return out_nodes, out_names


def _symbol_for(
    element: Element, netlist: Netlist, pin_maps: PinMaps, blocks: frozenset[str] = frozenset()
) -> tuple[S.Symbol, tuple[str, ...]]:
    k = element.kind
    nodes = element.nodes
    value = element.value
    mtype = netlist.model_types.get((element.model or "").lower(), "")
    label = (element.name, _short(pretty_value(value) if value else element.model or ""))
    if k == "R":
        return S.resistor(nodes), label
    if k == "C":
        return S.capacitor(nodes), label
    if k == "L":
        return S.inductor(nodes), label
    if k == "D":
        return S.diode(nodes), (element.name, element.model or "")
    if k in ("V", "I"):
        return S.source(nodes, voltage=(k == "V"), controlled=False), label
    if k == "B":
        return S.source(nodes, voltage=value.upper().startswith("V"), controlled=True), label
    if k in ("F", "H"):
        return S.source(nodes, voltage=(k == "H"), controlled=True), label
    if k in ("E", "G"):
        if len(nodes) >= 4:
            return S.controlled_four(nodes, voltage=(k == "E")), label
        return S.source(nodes, voltage=(k == "E"), controlled=True), label
    if k == "Q" and len(nodes) >= 3:
        return S.bjt(nodes, pnp=(mtype == "PNP")), (element.name, element.model or "")
    if k == "M" and len(nodes) >= 3:
        return S.mosfet(nodes, pmos=(mtype == "PMOS")), (element.name, element.model or "")
    if k in ("S", "W"):
        return S.switch(nodes), (element.name, element.model or "")
    # Subcircuit instances, Python devices and anything unknown: IC box.
    child = netlist.find(element.model) if k == "X" and element.model else None
    pin_map = pin_maps.get((element.model or "").lower())
    ic_nodes, ic_names = _ic_pins(element, child, pin_map)
    block = child is not None and child.name.lower() in blocks
    sides = [pin_side(name) for name in ic_names]
    if any(sides):
        symbol = S.directional_ic(ic_nodes, ic_names, sides, block=block)
    else:
        symbol = S.ic(ic_nodes, ic_names, block=block)
    return symbol, (element.name, element.model or _short(value))


def _windings(inductors: list[Element]) -> tuple[list[S.Winding], list[list[str]]]:
    """Group coupled inductors into windings: inductors sharing a node form one coil
    with taps (e.g. L_p1 d1-ct + L_p2 ct-d2 = centre-tapped d1-ct-d2). Returns the
    windings (first referenced first) and the inductor names per winding."""
    groups: list[list[Element]] = []
    for ind in inductors:
        joined = [g for g in groups if any(set(ind.nodes) & set(o.nodes) for o in g)]
        merged = [ind] + [o for g in joined for o in g]
        groups = [g for g in groups if g not in joined] + [merged]
    groups.sort(key=lambda g: min(inductors.index(i) for i in g))

    windings, names = [], []
    for group in groups:
        chain = _chain(group)
        if chain is None:
            # Not a simple series chain: draw each inductor as its own winding.
            for ind in sorted(group, key=inductors.index):
                windings.append(S.Winding(tuple(ind.nodes[:2]), ((0, True),)))
                names.append([ind.name])
            continue
        nodes, order = chain
        dots = tuple((seg, ind.nodes[0] == nodes[seg]) for seg, ind in enumerate(order))
        windings.append(S.Winding(tuple(nodes), dots))
        names.append([ind.name for ind in order])
    return windings, names


def _chain(group: list[Element]) -> tuple[list[str], list[Element]] | None:
    """Order inductors of a winding along the coil: (nodes, inductor per segment)."""
    if len(group) == 1:
        return list(group[0].nodes[:2]), group
    degree: dict[str, int] = {}
    for ind in group:
        for node in ind.nodes[:2]:
            degree[node] = degree.get(node, 0) + 1
    ends = sorted(n for n, d in degree.items() if d == 1)
    if len(ends) != 2 or any(d > 2 for d in degree.values()):
        return None
    # Start at the end that belongs to the first-listed inductor's first node if possible.
    first = group[0].nodes[0]
    node = first if first in ends else ends[0]
    nodes, order, left = [node], [], list(group)
    while left:
        ind = next((i for i in left if node in i.nodes[:2]), None)
        if ind is None:
            return None
        left.remove(ind)
        node = ind.nodes[1] if ind.nodes[0] == node else ind.nodes[0]
        nodes.append(node)
        order.append(ind)
    return nodes, order


def _transformers(elements: list[Element]) -> list[tuple[str, list[str], tuple[list[S.Winding], list[list[str]]]]]:
    """(name, inductor names, windings) per set of inductors coupled by K elements."""
    by_name = {e.name: e for e in elements}
    sets: list[tuple[list[str], list[str]]] = []  # (K names, inductor names)
    for k in elements:
        if k.kind != "K":
            continue
        refs = [r for r in k.refs if r in by_name]
        if len(refs) < 2:
            continue
        joined = [s for s in sets if set(s[1]) & set(refs)]
        ks = [k.name] + [n for s in joined for n in s[0]]
        inds = list(dict.fromkeys([i for s in joined for i in s[1]] + refs))
        sets = [s for s in sets if s not in joined] + [(ks, inds)]
    result = []
    for ks, inds in sets:
        result.append(("_".join(sorted(ks)), inds, _windings([by_name[i] for i in inds])))
    return result


def _bridges(elements: list[Element]) -> list[tuple[list[Element], tuple[str, str, str, str]]]:
    """Full bridge rectifiers: (diodes, (ac1, plus, ac2, minus)).

    Four diodes form a bridge when two of them have their cathodes on "plus", two
    their anodes on "minus", and both pairs share the same two AC nodes. Diodes that
    sit across a transistor or switch (flyback diodes) never count.
    """
    switches = [set(e.nodes) for e in elements if e.kind in ("Q", "M", "S", "W")]
    # A diode across a switch (an H-bridge's flyback diode) is drawn on its own, so four
    # of them around two half bridges are not a rectifier.
    diodes = [
        e for e in elements
        if e.kind == "D" and len(e.nodes) >= 2 and not any({e.nodes[0], e.nodes[1]} <= sw for sw in switches)
    ]
    by_pair = {(e.nodes[0], e.nodes[1]): e for e in diodes}  # (anode, cathode)
    used: set[str] = set()
    found = []
    for plus in dict.fromkeys(e.nodes[1] for e in diodes):
        into = [e for e in diodes if e.nodes[1] == plus and e.name not in used]
        for i, d1 in enumerate(into):
            for d2 in into[i + 1 :]:
                ac1, ac2 = d1.nodes[0], d2.nodes[0]
                if ac1 == ac2 or plus in (ac1, ac2) or d1.name in used or d2.name in used:
                    continue
                # Lower diodes: anode on a common "minus", cathodes on the AC nodes.
                for minus in dict.fromkeys(e.nodes[0] for e in diodes):
                    d3, d4 = by_pair.get((minus, ac1)), by_pair.get((minus, ac2))
                    if minus in (plus, ac1, ac2) or d3 is None or d4 is None or {d3.name, d4.name} & used:
                        continue
                    found.append(([d1, d2, d3, d4], (ac1, plus, ac2, minus)))
                    used.update(d.name for d in (d1, d2, d3, d4))
                    break
    return found


def grid_size(components: list[Component]) -> tuple[int, int]:
    """Grid large enough for comfortable placement (3:2 aspect)."""
    area = 0
    for comp in components:
        blocked, pins = comp.relative_cells(0)
        cells = blocked + pins
        bw = max(c[0] for c in cells) - min(c[0] for c in cells) + 1
        bh = max(c[1] for c in cells) - min(c[1] for c in cells) + 1
        area += (bw + 3) * (bh + 3)
    w = max(16, math.ceil(math.sqrt(area * 2.0 * 1.5)))
    h = max(12, math.ceil(w / 1.5))
    return w, h


def build_diagram(
    netlist: Netlist, circuit: str, pin_maps: PinMaps | None = None, blocks: frozenset[str] | set[str] = frozenset()
) -> Diagram:
    """Diagram of one subcircuit. ``blocks``: names of subcircuits that belong to the design
    (drawn as plain rectangles that open in their own tab); other subcircuits are ICs."""
    blocks = frozenset(b.lower() for b in blocks)
    pin_maps = {k.lower(): {p.lower(): n for p, n in v.items()} for k, v in (pin_maps or {}).items()}
    subckt: Subckt = netlist.get(circuit)
    elements = {e.name: e for e in subckt.elements}

    # Coupled inductors become one transformer per coupled set (see _windings).
    transformers = _transformers(subckt.elements)
    coupled = {name for _, inductors, _ in transformers for name in inductors}

    components: list[Component] = []

    bridges = _bridges(subckt.elements)
    in_bridge = {d.name for diodes, _ in bridges for d in diodes}

    def add(name, kind, symbol, label, title="", members=()):
        ground = frozenset(i for i, n in enumerate(symbol.pin_nodes) if n and is_ground(n))
        # Transformers stay upright (optionally mirrored): windings side by side. Everything
        # else, IC boxes and subcircuit blocks included, may be turned and mirrored (their pin
        # names turn along and stay readable).
        rotations: tuple[int, ...] = (0, 4) if kind == "K" else (0, 1, 2, 3, 4, 5, 6, 7)
        components.append(Component(name, kind, symbol, label, title or " ".join(label), ground, rotations=rotations, members=tuple(members)))

    for e in subckt.elements:
        if e.kind == "K" or e.name in coupled or e.name in in_bridge:
            continue
        symbol, label = _symbol_for(e, netlist, pin_maps, blocks)
        add(e.name, e.kind, symbol, label, f"{e.name} {e.model or ''} {e.value}".strip())
        child = netlist.find(e.model) if e.kind == "X" and e.model else None
        if child is not None and child.name.lower() in blocks:
            components[-1] = replace(components[-1], subckt=child.name)
    for name, inductors, windings in transformers:
        ks = [e for e in subckt.elements if e.kind == "K" and set(e.refs) & set(inductors)]
        coupling = ", ".join(k.value for k in ks if k.value)
        sides = " : ".join("/".join(w_names) for w_names in windings[1])
        title = " ".join(f"{e.name}={pretty_value(e.value)}" for e in subckt.elements if e.name in inductors)
        add(name, "K", S.transformer(windings[0]), (name, f"{sides} k={coupling}" if coupling else sides), f"{name}: {title}",
            list(inductors) + [k.name for k in ks])
    for diodes, nodes in bridges:
        names = [d.name for d in diodes]
        models = "/".join(dict.fromkeys(d.model or "" for d in diodes))
        name = "/".join(names)
        add(name, "D", S.bridge_rectifier(*nodes), (_short(name), models), f"Brückengleichrichter {name} {models}", names)

    # Supply ports: pins on them get their own supply symbol (like ground), no wire or label.
    # Internal upper-case rails (e.g. VDD_GATE_HS of a gate supply) are drawn the same way when
    # they connect three or more pins or only ICs; a two-pin rail with a passive (IC pin + its
    # decoupling cap) stays a wire or an attached part.
    pin_count: dict[str, int] = {}
    passive_on: set[str] = set()
    for comp in components:
        for node in comp.symbol.pin_nodes:
            if node and node not in subckt.ports:
                pin_count[node] = pin_count.get(node, 0) + 1
                if comp.kind in PASSIVE_KINDS:
                    passive_on.add(node)
    power_nets = frozenset(p for p in subckt.ports if not is_ground(p) and is_power(p)) | frozenset(
        n for n, count in pin_count.items()
        if (count >= 3 or n not in passive_on) and is_rail(n) and not is_ground(n)
    )
    for ci, comp in enumerate(components):
        pins = tuple((pi, node) for pi, node in enumerate(comp.symbol.pin_nodes) if node in power_nets)
        if pins:
            components[ci] = replace(comp, power_pins=pins, _cache={})
    components = _attach_two_terminals(components, power_nets)

    def wired_members(net: str) -> int:
        return sum(
            1
            for comp in components
            for pi, node in enumerate(comp.symbol.pin_nodes)
            if node == net and all(p != pi for p, _ in comp.power_pins)
        )

    # Nets between ICs / blocks only (two or more, nothing else on the net) are connected by
    # name: every pin gets a label, no wire. As soon as a plain part (R, C, D, ...) is on the
    # net too, it is drawn as a wire like any other net.
    ic_count: dict[str, set[int]] = {}
    with_parts: set[str] = set()
    for ci, comp in enumerate(components):
        for node in comp.symbol.pin_nodes:
            if comp.symbol.kind == S.KIND_IC:
                ic_count.setdefault(node, set()).add(ci)
            elif node:
                with_parts.add(node)
    named = {
        net
        for net, ics in ic_count.items()
        if net and not is_ground(net) and net not in power_nets and len(ics) >= 2 and net not in with_parts
    }

    for port in subckt.ports:
        if is_ground(port):
            continue
        power = port in power_nets
        if power and wired_members(port) == 0:
            continue  # only IC pins on this supply: their own symbols name it
        if not power and (port in named or wired_members(port) == 1):
            # A single pin (or a net labelled at every pin anyway): the port name is drawn
            # as a label right at the pin instead of a separate label part with a wire.
            named.add(port)
            continue
        symbol = S.power_port(port) if power else S.signal_port(port)
        # Signal labels only make sense horizontally (pin right or left); mirroring adds nothing.
        # Supply flags always upright, like ground symbols point down. Inputs enter from the left
        # (label left of its pin), outputs leave to the right; unmarked signals may face either way.
        direction = signal_direction(port)
        rotations = (0,) if power or direction == "in" else (2,) if direction == "out" else (0, 2)
        components.append(Component(f"port:{port}", "P", symbol, (), port, frozenset(), power, rotations=rotations))

    nets: dict[str, list[tuple[int, int]]] = {}
    for ci, comp in enumerate(components):
        for pi, node in enumerate(comp.symbol.pin_nodes):
            if node:
                nets.setdefault(node, []).append((ci, pi))
    named = frozenset(named)
    for ci, comp in enumerate(components):
        if comp.kind == "P":
            continue
        pins = tuple((pi, node) for pi, node in enumerate(comp.symbol.pin_nodes) if node in named)
        if pins:
            components[ci] = replace(comp, named_pins=pins, _cache={})

    components, groups = _assign_groups(components, subckt.groups, nets, named)
    pad_nets = _pad_nets(subckt, netlist)
    w, h = grid_size(components)
    return Diagram(
        circuit=subckt.name,
        ports=subckt.ports,
        components=components,
        nets=nets,
        w=w,
        h=h,
        named_nets=named,
        power_nets=power_nets,
        groups=groups,
        description=subckt.description,
        parts=dict(subckt.parts),
        pad_nets=pad_nets,
    )


# Terminal names of the primitive elements (as in the LVS maps), in node order.
TERMINALS = {"R": "pn", "C": "pn", "L": "pn", "V": "pn", "I": "pn", "B": "pn", "F": "pn", "H": "pn", "W": "pn",
             "D": ("a", "k"), "S": ("p", "n", "cp", "cn"), "E": ("p", "n", "cp", "cn"), "G": ("p", "n", "cp", "cn"),
             "Q": ("c", "b", "e"), "M": ("d", "g", "s", "b")}


def _terminals(element: Element, netlist: Netlist) -> dict[str, str]:
    """Terminal name (lower case) -> net of one element."""
    if element.kind == "X" and element.model:
        child = netlist.find(element.model)
        names = list(child.ports) if child is not None else []
    elif element.kind == "N":
        names = [str(i + 1) for i in range(len(element.nodes))]
    else:
        names = list(TERMINALS.get(element.kind, ()))
    return {n.lower(): node for n, node in zip(names, element.nodes)}


def _pad_nets(subckt: Subckt, netlist: Netlist) -> dict[str, dict[str, str]]:
    """Net at every pad of every board part, from its pad -> terminal map (two-terminal parts
    without a map: pad 1 / 2 = first / second node). Terminals of a part realising several
    instances are written ``<instance>.<terminal>``; "a+b" lists several terminals on one pad."""
    elements = {e.name.lower(): e for e in subckt.elements}
    out: dict[str, dict[str, str]] = {}
    for instance, parts in subckt.parts.items():
        element = elements.get(instance.lower())
        for part in parts:
            pads = out.setdefault(part.ref, {})
            if not part.pins:
                if element is not None and len(element.nodes) == 2:
                    pads.update({"1": element.nodes[0], "2": element.nodes[1]})
                continue
            for pad, terms in part.pins:
                for term in terms.split("+"):
                    if "." in term:
                        name, term = term.split(".", 1)
                        owner = elements.get(name.lower())
                    else:
                        owner = element
                    if owner is not None and owner.name.lower() == instance.lower():
                        net = _terminals(owner, netlist).get(term.lower())
                        if net:
                            pads.setdefault(pad, net)
    return out


REST = "rest"  # implicit group of the parts no header group claims


def _assign_groups(
    components: list[Component], groups: tuple[Group, ...], nets: dict[str, list[tuple[int, int]]], named: frozenset[str]
) -> tuple[list[Component], tuple[Group, ...]]:
    """Group of every component: the first header group listing it (by element name, any
    element it combines, or ``port:<net>``). Ports not listed join the group they are wired
    to most; everything else goes to the implicit REST group (only if there are groups)."""
    if not groups:
        return components, ()
    result = list(components)
    for ci, comp in enumerate(components):
        names = [comp.name, *comp.members] if comp.kind != "P" else [comp.name]
        gid = next((g.id for g in groups if any(g.matches(n) for n in names)), "")
        result[ci] = replace(comp, group=gid, _cache={})
    for ci, comp in enumerate(result):
        if comp.kind != "P" or comp.group:
            continue
        votes: dict[str, int] = {}
        for net, pins in nets.items():
            if net in named or all(c != ci for c, _ in pins):
                continue
            for other, _ in pins:
                if other != ci and result[other].group:
                    votes[result[other].group] = votes.get(result[other].group, 0) + 1
        if votes:
            result[ci] = replace(comp, group=max(votes, key=lambda g: (votes[g], -[x.id for x in groups].index(g))), _cache={})
    used = {c.group for c in result if c.group}
    out_groups = tuple(g for g in groups if g.id in used)
    if any(not c.group for c in result):
        result = [c if c.group else replace(c, group=REST, _cache={}) for c in result]
        out_groups += (Group(REST, "Sonstiges", (), "Bauteile ohne Gruppe"),)
    return result, out_groups
