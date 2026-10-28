"""High-level entry points for reports and scripts.

    from netlist2schematic.api import svg_for
    svg = svg_for("src/project/ldo_5v/sizing.py")            # top subcircuit
    svg = svg_for("src/project/ldo_5v/sizing.py", "LM2940_50")

The approved placement from ``schematic.json`` is used when it still matches
the netlist; otherwise the policy proposes one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from . import paths, trainer
from .diagram import Diagram, build_diagram
from .netlist import Netlist, parse_netlist
from .placement import Placement
from .render import schematic_svg
from .sources import Source, load_saved_placement, load_source


def top_subckts(netlist: Netlist) -> list[str]:
    """Subcircuits that no other subcircuit instantiates come first."""
    used = {e.model.lower() for s in netlist.subckts.values() for e in s.elements if e.kind == "X" and e.model}
    names = netlist.names
    return sorted(names, key=lambda n: (n.lower() in used, names.index(n)))


@dataclass
class HierarchyNode:
    """One element in the cell hierarchy: ``parent`` is the subcircuit it sits in,
    ``subckt`` the subcircuit it instantiates (X elements defined in the netlist)."""

    name: str
    parent: str
    kind: str
    model: str = ""
    subckt: str = ""
    children: list["HierarchyNode"] = field(default_factory=list)


def hierarchy(netlist: Netlist, top: str) -> HierarchyNode:
    """Cell hierarchy below ``top``: every element of every (nested) subcircuit."""

    def expand(name: str, seen: tuple[str, ...]) -> list[HierarchyNode]:
        subckt = netlist.get(name)
        nodes = []
        for e in subckt.elements:
            child = netlist.find(e.model) if e.kind == "X" and e.model else None
            node = HierarchyNode(e.name, subckt.name, e.kind, e.model or e.value, child.name if child else "")
            if child is not None and child.name.lower() not in seen:
                node.children = expand(child.name, seen + (child.name.lower(),))
            nodes.append(node)
        return nodes

    root = netlist.get(top)
    return HierarchyNode(root.name, "", "X", root.name, root.name, expand(root.name, (root.name.lower(),)))


def default_circuit(source: Source, netlist: Netlist) -> str:
    if source.circuit and netlist.find(source.circuit) is not None:
        return netlist.find(source.circuit).name
    return top_subckts(netlist)[0]


def saved_placement(diagram: Diagram, source: Source) -> Placement | None:
    """Approved placement for this subcircuit, if it is still valid for the current netlist."""
    named = load_saved_placement(source.circuit_dir, diagram.circuit)
    if named is None:
        return None
    try:
        env, shifted = trainer.fitted_env(diagram, named)
        env.demo_steps(shifted)
    except (KeyError, ValueError):
        return None
    # The fitted (repaired, migrated) positions: valid under the current symbol rules.
    return {diagram.component_index(n): pos for n, pos in shifted.items()}


def placement_for(diagram: Diagram, source: Source, model_path: Path | None = None) -> tuple[Placement, str]:
    """(placement, origin) with origin "saved" or "suggested"."""
    saved = saved_placement(diagram, source)
    if saved is not None:
        return saved, "saved"
    suggestion = trainer.suggest(diagram, model_path if model_path is not None else paths.model_path())
    return {diagram.component_index(n): pos for n, pos in suggestion.placement.items()}, "suggested"


def svg_for(path: str | Path, circuit: str | None = None, model_path: Path | None = None) -> str:
    source = load_source(Path(path))
    netlist = parse_netlist(source.netlist)
    diagram = build_diagram(netlist, circuit or default_circuit(source, netlist), source.pin_maps, source.subcircuits)
    placement, _ = placement_for(diagram, source, model_path)
    return schematic_svg(replace(diagram, bounded=False), placement, source=str(source.path))
