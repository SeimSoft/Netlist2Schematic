"""Parser for rendered Xyce netlists.

Only the structure needed for drawing is extracted: subcircuits, their ports,
the element lines inside them (with nodes, model/subcircuit name and value) and
the device type of every ``.MODEL`` (NPN, PNP, NMOS, ...). Parameter tokens
(``key=value``) are dropped, continuation lines are joined and braced
expressions such as ``I={I(V_sense)+8m}`` stay one token.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

import yaml

_GROUND = frozenset({"0", "agnd", "pgnd", "dgnd"})
# Naming convention: nets starting with these are drawn with ground symbols (GND, VSS,
# negative rails HVN and VEE), never with supply flags.
_GROUND_PREFIXES = ("gnd", "vss", "hvn", "vee")
# The main ground; every other ground net shows its name at the symbol.
_MAIN_GROUND = frozenset({"0", "gnd"})

# Port / net naming convention of the project:
#   <name>_i / _o     digital input / output      <name>_ai / _ao   analog input / output
#   <name>_ni / _no   active-low digital in / out <name>_io / _aio  bidirectional
#   UPPER CASE        supply rail (VDD3V3, GND_PE, VCCI)
SIGNAL_SUFFIX = re.compile(r"_(ni|no|a?i|a?o|a?io)$", re.I)
_RAIL_WORDS = ("GND", "VDD", "VCC", "VSS", "VEE")


def signal_direction(name: str) -> str | None:
    """"in", "out" or "inout" from the port suffix, None without one."""
    m = SIGNAL_SUFFIX.search(name)
    if m is None:
        return None
    kind = m.group(1).lower()
    kind = kind[1:] if kind.startswith(("a", "n")) and kind not in ("io",) else kind
    return {"i": "in", "o": "out", "io": "inout"}[kind]


def active_low(name: str) -> bool:
    """Active-low digital port (``_ni`` / ``_no``)."""
    m = SIGNAL_SUFFIX.search(name)
    return bool(m) and m.group(1).lower() in ("ni", "no")


def is_rail(name: str) -> bool:
    """Supply rail by convention: upper case, no signal suffix, a V-name or a GND/VDD/VCC/VSS/VEE word."""
    if not name or name != name.upper() or not any(c.isalpha() for c in name) or SIGNAL_SUFFIX.search(name):
        return False
    return name.startswith(("V", "+")) or any(w in name for w in _RAIL_WORDS)


def is_ground_rail(name: str) -> bool:
    return is_rail(name) and ("GND" in name or name.startswith(("VSS", "VEE")))


def is_ground(net: str) -> bool:
    name = net.lower()
    return name in _GROUND or name.startswith(_GROUND_PREFIXES) or is_ground_rail(net)


def ground_name(net: str) -> str:
    """Name written at a ground symbol ("" for the main ground)."""
    return "" if net.lower() in _MAIN_GROUND else net


# Elements whose node count is fixed by the element letter.
_FIXED_NODES = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "B": 2, "E": 4, "G": 4, "F": 2, "H": 2, "Q": 3, "S": 4, "W": 2}


@dataclass(frozen=True)
class Element:
    """One netlist element line inside a subcircuit."""

    name: str
    nodes: tuple[str, ...]
    model: str | None = None
    value: str = ""
    refs: tuple[str, ...] = ()  # coupled inductors of a K element

    @property
    def kind(self) -> str:
        return self.name[0].upper()


@dataclass(frozen=True)
class Group:
    """A functional block of a subcircuit (from its ``*@groups`` header)."""

    id: str
    title: str
    members: tuple[str, ...]  # element names or fnmatch patterns ("C_d*"); "port:<net>" for ports
    description: str = ""
    color: str = ""

    def matches(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name.lower(), m.lower()) for m in self.members)


@dataclass(frozen=True)
class Part:
    """A board part realising a netlist instance (``*@parts`` header, written from bom.yaml)."""

    ref: str
    footprint: str = ""  # KiCad "Library:Footprint"
    value: str = ""
    description: str = ""
    rating: str = ""
    datasheet: str = ""
    pins: tuple[tuple[str, str], ...] = ()  # (pad, terminal); a terminal list is joined with "+"


@dataclass
class Subckt:
    name: str
    ports: tuple[str, ...]
    elements: list[Element] = field(default_factory=list)
    groups: tuple[Group, ...] = ()
    description: str = ""  # the subcircuit's own description (``*@description`` in the header)
    parts: dict[str, tuple[Part, ...]] = field(default_factory=dict)  # instance -> board parts


@dataclass
class Netlist:
    subckts: dict[str, Subckt]
    model_types: dict[str, str] = field(default_factory=dict)  # lower-case model name -> NPN/PNP/...

    def get(self, name: str) -> Subckt:
        key = name.lower()
        for subckt in self.subckts.values():
            if subckt.name.lower() == key:
                return subckt
        raise KeyError(f"no subcircuit named {name!r}")

    def find(self, name: str) -> Subckt | None:
        try:
            return self.get(name)
        except KeyError:
            return None

    @property
    def names(self) -> list[str]:
        return list(self.subckts)


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+") and lines:
            lines[-1] += " " + stripped[1:].strip()
        else:
            lines.append(stripped)
    return lines


def _tokenize(line: str) -> list[str]:
    """Split on whitespace, keeping ``{...}`` and ``(...)`` groups together."""
    tokens: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in line:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch.isspace() and depth == 0:
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _model_types(lines: list[str]) -> dict[str, str]:
    types = {}
    for line in lines:
        tokens = _tokenize(line)
        if tokens and tokens[0].lower() == ".model" and len(tokens) > 2:
            kind = tokens[2].split("(")[0].upper()
            types[tokens[1].lower()] = kind
    return types


def _is_param(token: str) -> bool:
    """``key=value`` parameter. Braced expressions such as ``I={...}`` are not parameters."""
    return "=" in token and "{" not in token


def _parse_element(tokens: list[str], known_models: dict[str, str]) -> Element:
    name = tokens[0]
    kind = name[0].upper()
    args = [t for t in tokens[1:] if not _is_param(t)]

    if kind == "X":
        # X<name> node... subckt_name [params]
        return Element(name, tuple(args[:-1]), model=args[-1], value="")

    if kind == "K":
        # K<name> L1 L2 [L3...] coupling
        refs = tuple(a for a in args if a[0].upper() == "L")
        value = " ".join(a for a in args if a[0].upper() != "L")
        return Element(name, (), model=None, value=value, refs=refs)

    if kind in ("D", "Q", "M", "N"):
        # Model name is the argument that refers to a known .MODEL (or the last one).
        idx = next((i for i, a in enumerate(args) if a.lower() in known_models), len(args) - 1)
        nodes = list(args[:idx])
        if kind == "N":
            # Python devices pad unused terminals with 0.
            while len(nodes) > 1 and nodes[-1] == "0":
                nodes.pop()
        return Element(name, tuple(nodes), model=args[idx], value="")

    count = _FIXED_NODES.get(kind)
    if count is None:
        # Unknown element letter: assume two nodes, keep the rest as value.
        count = 2
    if kind == "S" and len(args) >= 5:
        return Element(name, tuple(args[:4]), model=args[4], value="")
    # Nodes end early at expressions such as VALUE={...} or POLY(1).
    nodes: list[str] = []
    for a in args:
        if len(nodes) == count or any(c in a for c in "{(="):
            break
        nodes.append(a)
    value = " ".join(args[len(nodes):])
    return Element(name, tuple(nodes), model=None, value=value)


GROUPS_START, GROUPS_END = "*@groups", "*@end"
PARTS_START = "*@parts"


def header_blocks(text: str, start: str) -> dict[str, object]:
    """YAML of every ``start`` … ``*@end`` comment block per subcircuit (lower-case name)."""
    out: dict[str, object] = {}
    current: str | None = None
    block: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if block is not None:
            if low.startswith(GROUPS_END):
                if current is not None:
                    out[current] = yaml.safe_load("\n".join(block))
                block = None
            else:
                block.append(line[1:] if line.startswith("*") else raw)
            continue
        if low.startswith(".subckt"):
            parts = line.split()
            current = parts[1].lower() if len(parts) > 1 else None
        elif low.startswith(".ends"):
            current = None
        elif low.startswith(start) and current is not None:
            block = []
    return out


def parse_parts(text: str) -> dict[str, dict[str, tuple[Part, ...]]]:
    """``*@parts`` headers: subcircuit (lower case) -> instance -> parts."""
    out: dict[str, dict[str, tuple[Part, ...]]] = {}
    for subckt, data in header_blocks(text, PARTS_START).items():
        parts: dict[str, tuple[Part, ...]] = {}
        for instance, entries in (data or {}).items():
            parts[str(instance)] = tuple(
                Part(
                    ref=str(e.get("ref", instance)),
                    footprint=str(e.get("footprint", "")),
                    value=str(e.get("value", "")),
                    description=str(e.get("description", "")),
                    rating=str(e.get("rating", "")),
                    datasheet=str(e.get("datasheet", "")),
                    pins=tuple(
                        (str(pad), "+".join(map(str, term)) if isinstance(term, list) else str(term))
                        for pad, term in (e.get("pins") or {}).items()
                    ),
                )
                for e in entries or []
            )
        out[subckt] = parts
    return out


def parse_groups(text: str) -> dict[str, tuple[str, tuple[Group, ...]]]:
    """Machine-readable headers per subcircuit (lower-case name -> (description, groups)).

    The header is YAML in comment lines between ``*@groups`` and ``*@end`` inside the
    ``.subckt`` (Xyce ignores it)::

        .subckt bridge_leg VDD12V GND ...
        *@groups
        * description: Half bridge with isolated gate driver
        * groups:
        *   - id: gate_driver
        *     title: Gate driver
        *     members: [X_U1, R_dis, R_dt, C_v*]
        *     description: UCC21520, dead time set by R_dt
        *     color: "#f59e0b"          # optional
        *@end

    Members are element names or patterns; ``port:<net>`` puts a port label in a group.
    Members missing from the netlist (e.g. only in the other model variant) are ignored.
    """
    out: dict[str, tuple[str, tuple[Group, ...]]] = {}
    current: str | None = None
    block: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if block is not None:
            if low.startswith(GROUPS_END):
                data = yaml.safe_load("\n".join(block)) or {}
                groups = tuple(
                    Group(
                        id=str(g["id"]),
                        title=str(g.get("title", g["id"])),
                        members=tuple(str(m) for m in g.get("members", []) or []),
                        description=" ".join(str(g.get("description", "")).split()),
                        color=str(g.get("color", "")),
                    )
                    for g in data.get("groups", []) or []
                )
                if current is not None:
                    out[current] = (" ".join(str(data.get("description", "")).split()), groups)
                block = None
            else:
                block.append(raw.strip()[1:] if raw.strip().startswith("*") else raw)
            continue
        if low.startswith(".subckt"):
            parts = line.split()
            current = parts[1].lower() if len(parts) > 1 else None
        elif low.startswith(".ends"):
            current = None
        elif low.startswith(GROUPS_START) and current is not None:
            block = []
    return out


def parse_netlist(text: str) -> Netlist:
    lines = _logical_lines(text)
    model_types = _model_types(lines)
    subckts: dict[str, Subckt] = {}
    current: Subckt | None = None

    for line in lines:
        tokens = _tokenize(line)
        head = tokens[0].lower()
        if head == ".subckt":
            current = Subckt(name=tokens[1], ports=tuple(t for t in tokens[2:] if not _is_param(t)))
            subckts[current.name] = current
        elif head == ".ends":
            current = None
        elif head.startswith("."):
            continue
        elif current is not None:
            current.elements.append(_parse_element(tokens, model_types))

    for name, parts in parse_parts(text).items():
        for subckt in subckts.values():
            if subckt.name.lower() == name:
                subckt.parts = parts
    for name, (description, groups) in parse_groups(text).items():
        for subckt in subckts.values():
            if subckt.name.lower() == name:
                subckt.groups, subckt.description = groups, description
    return Netlist(subckts=subckts, model_types=model_types)
