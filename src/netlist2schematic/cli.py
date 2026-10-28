"""Command line entry point: ``n2s ui | train | suggest | stats``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths, trainer
from .training_db import TrainingDB


def _cmd_ui(args: argparse.Namespace) -> int:
    from .ui import run_ui

    return run_ui(Path(args.netlist) if args.netlist else None)


def _cmd_train(args: argparse.Namespace) -> int:
    db = TrainingDB(paths.db_path())
    if db.count() == 0:
        print("Keine Trainingsbeispiele vorhanden.", file=sys.stderr)
        return 1
    cfg = trainer.TrainConfig(bc_epochs=args.epochs, rl_episodes=args.rl_episodes)

    def progress(stage: str, done: int, total: int, info: str) -> None:
        print(f"{stage}: {done}/{total} {info}")

    history = trainer.train(db, paths.model_path(), cfg, progress)
    print(f"gespeichert: {paths.model_path()} (BC-Schritte {len(history['bc_loss'])}, RL-Episoden {len(history['rl_reward'])})")
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    from .api import default_circuit
    from .diagram import build_diagram
    from .netlist import parse_netlist
    from .sources import load_source

    source = load_source(Path(args.netlist))
    netlist = parse_netlist(source.netlist)
    diagram = build_diagram(netlist, args.circuit or default_circuit(source, netlist), source.pin_maps, source.subcircuits)
    result = trainer.suggest(diagram, paths.model_path())
    for name, (x, y, r) in sorted(result.placement.items()):
        print(f"{name}\t{x}\t{y}\t{r}")
    print(f"score {result.reward:.2f}", result.details, file=sys.stderr)
    return 0


def _cmd_stats(_: argparse.Namespace) -> int:
    db = TrainingDB(paths.db_path())
    print(f"datenbank: {db.path}")
    print(f"beispiele: {db.count()}")
    print(f"modell: {paths.model_path()} ({'vorhanden' if paths.model_path().exists() else 'fehlt'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="n2s", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ui = sub.add_parser("ui", help="Qt-Oberfläche starten")
    p_ui.add_argument("netlist", nargs="?", help="optional: gerenderte Netzliste öffnen")
    p_ui.set_defaults(func=_cmd_ui)

    p_train = sub.add_parser("train", help="Modell auf der Trainingsdatenbank trainieren")
    p_train.add_argument("--epochs", type=int, default=200, help="Behavior-Cloning-Epochen")
    p_train.add_argument("--rl-episodes", type=int, default=0, help="REINFORCE-Episoden nach dem Cloning")
    p_train.set_defaults(func=_cmd_train)

    p_sug = sub.add_parser("suggest", help="Platzierungsvorschlag für eine Netzliste ausgeben")
    p_sug.add_argument("netlist")
    p_sug.add_argument("circuit", nargs="?")
    p_sug.set_defaults(func=_cmd_suggest)

    p_stats = sub.add_parser("stats", help="Datenbank und Modell anzeigen")
    p_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
