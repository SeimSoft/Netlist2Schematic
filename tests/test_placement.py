from dataclasses import replace

import numpy as np
import pytest
from circuits import HUMAN, RC, TWO_ICS

from netlist2schematic.diagram import build_diagram
from netlist2schematic.env import PlacementEnv, next_component, placement_order
from netlist2schematic.netlist import parse_netlist
from netlist2schematic.placement import fit_to_grid, group_fits
from netlist2schematic.render import schematic_svg
from netlist2schematic.routing import route
from netlist2schematic.symbols import KIND_IC


def rc():
    return build_diagram(parse_netlist(RC), "rc")


def by_index(d, named):
    return {d.component_index(n): pos for n, pos in named.items()}


def test_mask_agrees_with_group_fits():
    d = rc()
    env = PlacementEnv(d, revisits=False)
    first = env.current
    env.step(env.index_of(10, 8, 0))
    valid = env.valid_mask()
    for index in range(0, env.num_candidates, 7):
        x, y, o = (int(v) for v in env.anchors[index])
        expected = group_fits(d, env.placement, {env.current: (x, y, o)})
        assert valid[index] == expected, (x, y, o)
    assert first in env.placement


def test_demo_round_trip_and_rejections():
    d = rc()
    env = PlacementEnv(d)
    steps = env.demo_steps(HUMAN)
    assert len(steps) >= 4  # placements plus revisions (which keep the human's positions)
    assert env.named_placement() == HUMAN
    with pytest.raises(ValueError):
        env.demo_steps({**HUMAN, "C1": HUMAN["R1"]})
    with pytest.raises(ValueError):
        env.demo_steps({"R1": HUMAN["R1"]})


def test_pin_exit_stays_free():
    d = rc()
    r1 = d.component_index("R1")
    placement = {r1: (10, 8, 0)}
    exit_cell = d.components[r1].exit_cells(0)[0]
    # Port label whose body would cover R1's pin exit.
    vin = d.component_index("port:vout")
    body, _ = d.components[vin].relative_cells(0)
    bx, by = body[0]
    target = (10 + exit_cell[0] - bx, 8 + exit_cell[1] - by, 0)
    assert not group_fits(d, placement, {vin: target})


def test_unbounded_editing_and_fit_to_grid():
    d = replace(rc(), bounded=False)
    placement = by_index(d, {n: (x - 100, y - 50, o) for n, (x, y, o) in HUMAN.items()})
    assert group_fits(d, {}, placement)
    fitted, shifted = fit_to_grid(d, placement)
    assert fitted.bounded
    PlacementEnv(fitted).demo_steps({fitted.components[ci].name: pos for ci, pos in shifted.items()})
    # A placement inside the grid is not moved.
    inside = by_index(d, HUMAN)
    assert fit_to_grid(d, inside)[1] == inside


def test_placement_order_ic_first_then_connected():
    d = build_diagram(parse_netlist(TWO_ICS), "top")
    order = placement_order(d)
    assert d.components[order[0]].symbol.kind == KIND_IC
    placed = {order[0]}
    for ci in order[1:]:
        if any(o in placed for o in d.neighbours(ci)):
            pass
        else:
            # Only allowed when nothing unplaced is connected.
            assert not any(any(o in placed for o in d.neighbours(c)) for c in range(len(d.components)) if c not in placed)
        placed.add(ci)
    assert next_component(d, set()) == order[0]


def test_route_connects_rc_without_crossings():
    d = rc()
    routing = route(d, by_index(d, HUMAN))
    assert routing.unrouted == []
    assert routing.crossings == 0
    assert "gnd" not in routing.paths  # ground uses symbols


def test_named_nets_are_not_routed_and_get_labels():
    d = build_diagram(parse_netlist(TWO_ICS), "top")
    env = PlacementEnv(d)
    while not env.done:
        features, valid = env.observe()
        env.step(int(np.flatnonzero(valid)[0]))
    routing = route(d, env.placement)
    assert "mid" not in routing.paths
    svg = schematic_svg(d, env.placement)
    assert svg.count(">mid</text>") == 2  # one label per pin on the named net (X1, X2)


def test_wires_never_overlap_other_nets():
    d = rc()
    routing = route(d, by_index(d, HUMAN))
    for cell, nets in routing.usage.items():
        if len(nets) > 1:
            axes = [a for a in nets.values()]
            assert all(len(a) == 1 for a in axes) and axes[0] != axes[1]


def test_conflicts_name_the_blocking_part_and_nearest_fit_resolves():
    from netlist2schematic.placement import group_conflicts, nearest_fit

    d = rc()
    placement = by_index(d, HUMAN)
    r1, c1 = d.component_index("R1"), d.component_index("C1")
    on_top = {c1: placement[r1]}
    assert r1 in group_conflicts(d, placement, on_top)
    shifted = nearest_fit(d, placement, {c1: (placement[c1][0] - 2, placement[c1][1], placement[c1][2])}, radius=3)
    assert shifted is not None and group_fits(d, placement, shifted)


def test_start_order_ic_then_transformer_then_source():
    with_trafo = """
.subckt t vin gnd out
V1 vin gnd 5
R1 vin a 1k
L1 a gnd 1u
L2 out gnd 1u
K1 L1 L2 0.99
.ends
"""
    d = build_diagram(parse_netlist(with_trafo), "t")
    assert d.components[next_component(d, set())].name == "K1"
    only_source = """
.subckt s vin gnd
R1 vin a 1k
C1 a gnd 1u
I1 a gnd 1m
.ends
"""
    d = build_diagram(parse_netlist(only_source), "s")
    assert d.components[next_component(d, set())].name == "I1"
    # With an IC present, the IC comes first.
    d = build_diagram(parse_netlist(TWO_ICS), "top")
    assert d.components[next_component(d, set())].symbol.kind == KIND_IC


def test_wires_leave_pins_straight():
    # R1 horizontal, C1 vertical below-right: the wire leaves R1's right pin to the
    # right before turning, and enters C1's top pin from above.
    text = """
.subckt s a b gnd
R1 a b 1k
C1 b gnd 1u
.ends
"""
    d = build_diagram(parse_netlist(text), "s")
    r1, c1 = d.component_index("R1"), d.component_index("C1")
    placement = {r1: (5, 5, 0), c1: (12, 10, 1)}
    routing = route(d, placement)
    path = routing.paths["b"][0]
    r1_pin = (7, 5)  # right pin of R1
    c1_pin = (12, 8)  # top pin of C1 (vertical)
    ends = {path[0], path[-1]}
    assert ends == {r1_pin, c1_pin}
    seq = path if path[0] == r1_pin else path[::-1]
    assert seq[1] == (8, 5)  # straight out of R1 to the right
    assert seq[-2] == (12, 7)  # into C1 from above


def test_placement_order_is_breadth_first_per_ic():
    # X1's children (in pin order) first, then grandchildren, and so on. X2 shares "mid"
    # with X1 (connected by name, no wire): it starts its own tree afterwards. X3 hangs
    # on a wire behind R_a: a grandchild like any other part. v3v3 is a supply (no wire).
    text = """
.subckt top gnd v3v3
X1 n1 n2 mid v3v3 CHIP
X2 mid n3 gnd v3v3 CHIP
X3 n1a gnd gnd v3v3 CHIP
R_a n1 n1a 1k
R_aa n1a gnd 1k
R_b n2 n2b 1k
R_bb n2b gnd 1k
R_c n3 n3c 1k
C_c n3c gnd 1n
C_dec v3v3 gnd 100n
.ends
.subckt CHIP P1 P2 P3 P4
R9 P1 P2 1
.ends
"""
    d = build_diagram(parse_netlist(text), "top")
    names = [d.components[ci].name for ci in placement_order(d)]
    assert names[:6] == ["X1", "R_a", "R_b", "X3", "R_aa", "R_bb"]
    assert names[6:9] == ["X2", "R_c", "C_c"]
    assert names[-1] == "C_dec"  # only on supply and ground: nothing wires it to a tree
    # A decoupling cap on the supply gets a supply symbol at its pin, no flag.
    c = d.components[d.component_index("C_dec")]
    assert c.power_pins == ((0, "v3v3"),) and "port:v3v3" not in names
    env = PlacementEnv(replace(d, bounded=True))
    assert d.components[env.current].name == "X1"
