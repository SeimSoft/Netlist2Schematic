"""Netlist sources: rendered netlists, sizing.py modules and jinja2 templates.

A ``sizing.py`` is imported (fresh on every load) and its ``sizing()`` is
called; if it requires the specs, the sibling ``specs.yaml`` is passed. A
``.jinja2`` template is rendered through its sibling ``sizing.py``.

IC pin numbers come from the circuit's ``bom.yaml``: entries with ``subckt``
and ``pins`` keys map subcircuit ports to package pins, e.g.::

    - ref: U1
      subckt: LTC7801
      pins: {VIN: 1, RUN: 2, OVLO: 3, GND: 25}

Children listed under ``children:`` are searched too. Approved placements are
stored next to the circuit in ``schematic.json`` (one entry per subcircuit) so
they are versioned with the design and reused by the reports.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .diagram import PinMaps

SCHEMATIC_FILE = "schematic.json"


@dataclass
class Source:
    path: Path
    netlist: str
    circuit: str | None = None  # preferred subcircuit (the circuit folder name)
    pin_maps: PinMaps = field(default_factory=dict)
    circuit_dir: Path | None = None  # where schematic.json lives
    subcircuits: frozenset[str] = frozenset()  # design subcircuits (lower case): blocks, not ICs


def _src_root(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    return None


def _import_sizing(path: Path):
    root = _src_root(path)
    if root is not None:
        # Sizing modules import both "project.x" and "src.project.x".
        for entry in (str(root.parent), str(root)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        module_name = ".".join(path.relative_to(root).with_suffix("").parts)
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)
    spec = importlib.util.spec_from_file_location(f"n2s_sizing_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sizing(path: Path) -> str:
    module = _import_sizing(path)
    fn = getattr(module, "sizing", None)
    if fn is None:
        raise ValueError(f"{path} has no sizing() function")
    params = inspect.signature(fn).parameters
    first = next(iter(params.values()), None)
    if first is not None and first.default is inspect.Parameter.empty:
        specs = yaml.safe_load((path.parent / "specs.yaml").read_text(encoding="utf-8"))
        return fn(specs)
    return fn()


def pin_maps_for(circuit_dir: Path, seen: set[Path] | None = None) -> PinMaps:
    """Pin maps from bom.yaml of a circuit and, recursively, of its children."""
    seen = seen or set()
    if circuit_dir in seen:
        return {}
    seen.add(circuit_dir)
    bom_path = circuit_dir / "bom.yaml"
    if not bom_path.exists():
        return {}
    data = yaml.safe_load(bom_path.read_text(encoding="utf-8")) or {}
    maps: PinMaps = {}
    for child in data.get("children", []) or []:
        maps.update(pin_maps_for(circuit_dir.parent / child["circuit"], seen))
    for entry in data.get("bom", []) or []:
        if isinstance(entry, dict) and entry.get("subckt") and entry.get("pins"):
            maps[str(entry["subckt"]).lower()] = {str(k).lower(): int(v) for k, v in entry["pins"].items()}
    return maps


def subcircuits_for(circuit_dir: Path) -> frozenset[str]:
    """Subcircuits that belong to the design: circuits with their own project folder next
    to this one, plus the children listed in bom.yaml (recursively). Everything else
    (vendor models such as AMC3330) is an IC."""
    names = {d.name.lower() for d in circuit_dir.parent.iterdir() if d.is_dir() and any(d.glob("netlist.cir*"))}

    def children(folder: Path, seen: set[Path]) -> None:
        bom_path = folder / "bom.yaml"
        if folder in seen or not bom_path.exists():
            return
        seen.add(folder)
        data = yaml.safe_load(bom_path.read_text(encoding="utf-8")) or {}
        for child in data.get("children", []) or []:
            names.add(str(child["circuit"]).lower())
            children(folder.parent / child["circuit"], seen)

    children(circuit_dir, set())
    return frozenset(names)


def load_source(path: Path) -> Source:
    path = Path(path).resolve()
    if path.suffix == ".py":
        netlist = run_sizing(path)
        circuit_dir = path.parent
    elif path.name.endswith(".jinja2"):
        sizing = path.parent / "sizing.py"
        if not sizing.exists():
            raise ValueError(f"{path.name} needs a sibling sizing.py to be rendered")
        netlist = run_sizing(sizing)
        circuit_dir = path.parent
    else:
        netlist = path.read_text()
        circuit_dir = path.parent if (path.parent / "bom.yaml").exists() else None
    circuit = circuit_dir.name if circuit_dir is not None else None
    pin_maps = pin_maps_for(circuit_dir) if circuit_dir is not None else {}
    blocks = subcircuits_for(circuit_dir) if circuit_dir is not None else frozenset()
    return Source(path=path, netlist=netlist, circuit=circuit, pin_maps=pin_maps, circuit_dir=circuit_dir, subcircuits=blocks)


def load_saved_placement(circuit_dir: Path | None, circuit: str) -> dict[str, tuple[int, int, int]] | None:
    if circuit_dir is None:
        return None
    path = circuit_dir / SCHEMATIC_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    entry = data.get(circuit)
    if not entry:
        return None
    return {name: tuple(pos) for name, pos in entry.items()}


def save_placement(circuit_dir: Path, circuit: str, placement: dict[str, tuple[int, int, int]]) -> Path:
    path = circuit_dir / SCHEMATIC_FILE
    data = json.loads(path.read_text()) if path.exists() else {}
    data[circuit] = {name: list(pos) for name, pos in sorted(placement.items())}
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path
