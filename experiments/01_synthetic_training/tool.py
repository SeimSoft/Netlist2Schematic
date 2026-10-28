"""Synthetic training data: hand-drawn placements of textbook circuits.

Usage (from netlist2schematic/):
    uv run python experiments/01_synthetic_training/tool.py check  [names...]   # validate + target PNGs
    uv run python experiments/01_synthetic_training/tool.py before [names...]   # proposals of the current model
    uv run python experiments/01_synthetic_training/tool.py add    [names...]   # store as training examples
    uv run python experiments/01_synthetic_training/tool.py after  [names...]   # proposals after retraining
    uv run python experiments/01_synthetic_training/tool.py report <batch> [names...]

Images go to reports/synthetic_training/<batch>/ (target, before, after); metrics to
results.json there. Examples are stored with source "synthetic" so they can be told
apart from the hand-made ones in the training data dialog.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # netlist2schematic/
sys.path.insert(0, str(HERE))

from circuits import CIRCUITS  # noqa: E402

from netlist2schematic import paths, trainer  # noqa: E402
from netlist2schematic.diagram import build_diagram  # noqa: E402
from netlist2schematic.graphics import to_svg  # noqa: E402
from netlist2schematic.netlist import parse_netlist  # noqa: E402
from netlist2schematic.placement import group_conflicts  # noqa: E402
from netlist2schematic.render import schematic_graphics  # noqa: E402
from netlist2schematic.training_db import TrainingDB  # noqa: E402

SOURCE = "synthetic"


def out_dir(batch: str) -> Path:
    d = ROOT / "reports" / "synthetic_training" / batch
    d.mkdir(parents=True, exist_ok=True)
    return d


def diagram(name: str):
    netlist, _ = CIRCUITS[name]
    nl = parse_netlist(netlist)
    return build_diagram(nl, nl.names[0])


def by_index(d, named):
    return {d.component_index(n): pos for n, pos in named.items()}


def problems(name: str) -> list[str]:
    """Missing or unknown parts and overlaps of the hand placement."""
    d = replace(diagram(name), bounded=False)
    _, named = CIRCUITS[name]
    names = {c.name for c in d.components}
    out = [f"missing: {n}" for n in sorted(names - set(named))]
    out += [f"unknown: {n}" for n in sorted(set(named) - names)]
    if out:
        return out
    pl = by_index(d, named)
    for ci, pos in pl.items():
        others = {k: v for k, v in pl.items() if k != ci}
        blockers = group_conflicts(d, others, {ci: pos})
        if blockers:
            out.append(f"{d.components[ci].name} overlaps {sorted(d.components[b].name if b is not None else 'edge/orientation' for b in blockers)}")
    return out


def png(name: str, named: dict, path: Path) -> None:
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    QGuiApplication.instance() or QGuiApplication([])
    d = replace(diagram(name), bounded=False)
    svg = to_svg(schematic_graphics(d, by_index(d, named)), scale=28)
    path.with_suffix(".svg").write_text(svg)
    r = QSvgRenderer(QByteArray(svg.encode()))
    size = r.defaultSize()
    img = QImage(size.width(), size.height(), QImage.Format.Format_ARGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    r.render(p)
    p.end()
    img.save(str(path))


def agreement(name: str, result: dict, tol: int = 0) -> float:
    """Share of connected part pairs whose relative offset (and orientations) match the target."""
    d = diagram(name)
    _, named = CIRCUITS[name]
    hits = pairs = 0
    for ci in range(len(d.components)):
        for cj in d.neighbours(ci):
            if cj <= ci:
                continue
            a, b = d.components[ci].name, d.components[cj].name
            pairs += 1
            hx, hy = named[b][0] - named[a][0], named[b][1] - named[a][1]
            rx, ry = result[b][0] - result[a][0], result[b][1] - result[a][1]
            same = named[a][2] == result[a][2] and named[b][2] == result[b][2]
            hits += max(abs(hx - rx), abs(hy - ry)) <= tol and same
    return hits / max(pairs, 1)


def proposal(name: str, beam: int | None = None) -> dict:
    """Proposal of the model in $MODEL (default: the current weights), beam search width $BEAM (default 4)."""
    model = Path(os.environ.get("MODEL", paths.model_path()))
    return trainer.suggest(diagram(name), model, beam=beam or int(os.environ.get("BEAM", "4"))).placement


def load_results(batch: str) -> dict:
    f = out_dir(batch) / "results.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_results(batch: str, results: dict) -> None:
    (out_dir(batch) / "results.json").write_text(json.dumps(results, indent=1))


def main(argv: list[str]) -> None:
    cmd = argv[0]
    batch = os.environ.get("BATCH", "01_small")
    names = argv[1:] or list(CIRCUITS)
    results = load_results(batch)
    for name in names:
        r = results.setdefault(name, {})
        if cmd == "check":
            issues = problems(name)
            print(f"{name:18s} {'OK' if not issues else '; '.join(issues)}")
            if not issues:
                png(name, CIRCUITS[name][1], out_dir(batch) / f"{name}_target.png")
        elif cmd in ("before", "after"):
            res = proposal(name)
            png(name, res, out_dir(batch) / f"{name}_{cmd}.png")
            r[cmd] = {"exact": agreement(name, res), "near": agreement(name, res, 1)}
            greedy = proposal(name, beam=1)
            r[f"{cmd}_greedy"] = {"exact": agreement(name, greedy), "near": agreement(name, greedy, 1)}
            print(f"{name:18s} {cmd}: pairs exact {r[cmd]['exact']:.0%}, within 1 cell {r[cmd]['near']:.0%}"
                  f"  (greedy {r[f'{cmd}_greedy']['exact']:.0%})")
        elif cmd == "pins":
            d = replace(diagram(name), bounded=False)
            for n, (x, y, o) in CIRCUITS[name][1].items():
                comp = d.components[d.component_index(n)]
                pins = [f"{comp.symbol.pin_nodes[i] or '-'}@({x + px},{y + py})" for i, ((px, py), _) in enumerate(comp.pin(i, o) for i in range(len(comp.symbol.pins)))]
                print(f"  {n:8s} {' '.join(pins)}")
        elif cmd == "add":
            netlist, named = CIRCUITS[name]
            db = TrainingDB(paths.db_path())
            ex_id = db.add(parse_netlist(netlist).names[0], netlist, named, source=SOURCE)
            r["example_id"] = ex_id
            print(f"{name:18s} stored as example {ex_id}")
    save_results(batch, results)


if __name__ == "__main__":
    main(sys.argv[1:])
