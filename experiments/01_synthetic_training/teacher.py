"""Teacher-forced accuracy: each decision with the hand placement of all earlier parts given."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuits import CIRCUITS  # noqa: E402

from netlist2schematic import paths, policy, trainer  # noqa: E402
from netlist2schematic.diagram import build_diagram  # noqa: E402
from netlist2schematic.netlist import parse_netlist  # noqa: E402

params = policy.load_or_init(paths.model_path())
for name in sys.argv[1:] or CIRCUITS:
    netlist, named = CIRCUITS[name]
    nl = parse_netlist(netlist)
    env, fitted = trainer.fitted_env(build_diagram(nl, nl.names[0]), named)
    env.reset()
    wrong = []
    total = exact = 0
    while not env.done:
        comp = env.diagram.components[env.current].name
        rev = env.revisiting
        f, v = env.observe()
        target = env.index_of(*fitted[comp])
        if not rev:
            guess = policy.greedy_action(params, f, v, env.graph_inputs())
            total += 1
            if guess == target:
                exact += 1
            else:
                gx, gy, go = env.anchors[guess]
                tx, ty, to = fitted[comp]
                wrong.append(f"{comp}: got ({gx - tx:+d},{gy - ty:+d}) o{go} want o{to}")
        env.step(target)
    print(f"{name:18s} teacher-forced {exact}/{total}  " + "; ".join(wrong))
