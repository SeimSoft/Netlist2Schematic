from circuits import TWO_ICS

from netlist2schematic.diagram import build_diagram, pretty_value
from netlist2schematic.netlist import parse_netlist
from netlist2schematic.symbols import KIND_ACTIVE, KIND_IC, KIND_PASSIVE, KIND_PORT, KIND_SOURCE

SAMPLE = """
* comment line
.param vdd=5
.subckt demo vin vout gnd
R1 vin vout 1k
C1 vout gnd
+ 10u IC=0
Q1 vout vb gnd QP
.MODEL QP PNP (BF=100)
N_DUT vin vout gnd 0 0 0 IdealModel
.model IdealModel npy (MODULE="m" CLASS="C")
B_in vin gnd I={I(V_sense)+8m}
E1 vb 0 VALUE={V(vin)*0.5}
V1 vin gnd PULSE(0 5 0 1n 1n 1u 2u)
L1 vin a 1u
L2 b gnd 1u
K1 L1 L2 0.99
X_REG vin gnd vout LM1117
.ends demo
.subckt LM1117 IN GND OUT
R9 IN OUT 1
.ends
"""


def demo():
    return parse_netlist(SAMPLE)


def element(name):
    return next(e for e in demo().get("demo").elements if e.name == name)


def test_subckts_ports_and_model_types():
    nl = demo()
    assert nl.names == ["demo", "LM1117"]
    assert nl.get("demo").ports == ("vin", "vout", "gnd")
    assert nl.model_types["qp"] == "PNP"


def test_continuation_and_value():
    assert element("C1").nodes == ("vout", "gnd")
    assert element("C1").value == "10u"


def test_npy_device_drops_padding_nodes():
    dut = element("N_DUT")
    assert dut.model == "IdealModel"
    assert dut.nodes == ("vin", "vout", "gnd")


def test_expressions_stay_one_token():
    assert element("B_in").value == "I={I(V_sense)+8m}"
    assert element("E1").nodes == ("vb", "0")
    assert element("V1").value.startswith("PULSE(")


def test_coupling_refs():
    assert element("K1").refs == ("L1", "L2")


def test_diagram_symbols():
    d = build_diagram(demo(), "demo")
    kinds = {c.name: c.symbol.kind for c in d.components}
    assert kinds["R1"] == KIND_PASSIVE
    assert kinds["Q1"] == KIND_ACTIVE
    assert kinds["B_in"] == KIND_SOURCE
    assert kinds["X_REG"] == KIND_IC
    assert kinds["N_DUT"] == KIND_IC
    assert kinds["K1"] == KIND_PASSIVE  # transformer replaces L1/L2
    assert "L1" not in kinds and "L2" not in kinds
    # vin is a supply: every pin on it gets its own supply symbol, no flag.
    # vout connects two ICs (N_DUT, X_REG) but also R1, C1, Q1: a wire with a port label.
    assert "port:vin" not in kinds and kinds["port:vout"] == KIND_PORT and "vout" not in d.named_nets
    r1 = d.components[d.component_index("R1")]
    assert r1.power_pins == ((0, "vin"),)
    assert "port:gnd" not in kinds


def test_ground_pins_and_ic_orientation():
    d = build_diagram(demo(), "demo")
    c1 = d.components[d.component_index("C1")]
    assert c1.ground_pins == frozenset({1})
    dut = d.components[d.component_index("N_DUT")]
    assert dut.rotations == tuple(range(8))  # IC boxes may be turned and mirrored
    ic = d.components[d.component_index("X_REG")]
    assert ic.rotations == tuple(range(8))  # also with the pins grouped by function
    gnd = ic.symbol.pin_nodes.index("gnd")
    assert ic.symbol.pin_dirs[gnd] == (0, 1)  # ground rail at the bottom


CONVENTION = """
.subckt top VDD3V3 GND sense_ai ctrl_o
X1 sense_ai ctrl_o VDD3V3 GND VSS_ISO mode_io AMP
R1 VDD_GATE GND 1k
R6 VDD_GATE GND 1k
C6 VDD_GATE GND 1n
R2 sense_ai sense_div 10k
R4 sense_div GND 10k
R3 ctrl_o ctrl_div 10k
R5 ctrl_div GND 10k
.ends
.subckt AMP in_ai out_o VDD GND VSS mode_io
R9 in_ai out_o 1
.ends
"""


def test_naming_convention_places_pins_by_function():
    from netlist2schematic.diagram import is_power, pin_side
    from netlist2schematic.netlist import is_ground, signal_direction

    assert [signal_direction(n) for n in ("a_i", "a_o", "a_ai", "a_ao", "a_io", "VDD")] == ["in", "out", "in", "out", "inout", None]
    assert signal_direction("reset_ni") == "in" and signal_direction("fault_no") == "out"
    assert is_power("VDD3V3") and is_power("VCCI") and not is_power("vdd_sense_ai")
    assert is_ground("GND_PE") and is_ground("VSS_ISO") and is_ground("DCDC_HGND") and not is_ground("VDD12V")
    assert [pin_side(n) for n in ("in_ai", "out_o", "VDD", "GND", "mode_io", "OUT")] == ["left", "right", "top", "bottom", None, None]

    d = build_diagram(parse_netlist(CONVENTION), "top")
    amp = d.components[d.component_index("X1")].symbol
    dirs = dict(zip(("in", "out", "vdd", "gnd", "vss", "mode"), amp.pin_dirs))
    assert dirs["in"] == (-1, 0) and dirs["out"] == (1, 0)
    assert dirs["vdd"] == (0, -1) and dirs["gnd"] == (0, 1) and dirs["vss"] == (0, 1)
    assert "VDD_GATE" in d.power_nets  # internal upper-case rail on 3+ pins: supply symbols, no wire
    # signal ports: inputs enter from the left, outputs leave to the right
    assert d.components[d.component_index("port:sense_ai")].rotations == (0,)
    assert d.components[d.component_index("port:ctrl_o")].rotations == (2,)


def test_ic_pins_follow_child_ports_and_pin_map():
    d = build_diagram(demo(), "demo")
    ic = d.components[d.component_index("X_REG")].symbol
    assert ic.pin_nodes == ("vin", "gnd", "vout")
    mapped = build_diagram(demo(), "demo", {"LM1117": {"OUT": 2, "IN": 3, "GND": 1, "TAB": 4}})
    ic = mapped.components[mapped.component_index("X_REG")].symbol
    assert ic.pin_nodes == ("gnd", "vout", "vin", "")  # package order, pin 4 unconnected
    assert len(ic.pins) == 4


def test_nets_between_ics_are_connected_by_name():
    d = build_diagram(parse_netlist(TWO_ICS), "top")
    assert d.named_nets == frozenset({"mid", "vout"})  # vout: port on a single pin
    x1 = d.components[d.component_index("X1")]
    assert "mid" in dict(x1.named_pins).values()


def test_net_with_ics_and_a_plain_part_is_a_wire():
    # Two ICs and a resistor on one net: drawn as a wire (net), not with labels.
    text = """
.subckt top vin vout gnd
X1 vin mid gnd CHIP
X2 mid vout gnd CHIP
R1 mid gnd 10k
.ends
.subckt CHIP IN OUT GND
R9 IN OUT 1
.ends
"""
    d = build_diagram(parse_netlist(text), "top")
    assert "mid" not in d.named_nets
    assert all("mid" not in dict(c.named_pins).values() for c in d.components)


def test_pretty_value():
    assert pretty_value("1.02e+06") == "1.02M"
    assert pretty_value("2.2e-09") == "2.2n"
    assert pretty_value("33000") == "33k"
    assert pretty_value("10u") == "10u"


CENTER_TAP = """
.subckt pp vin gnd hv
L_p1 d1 ct 2.5u
L_p2 ct d2 2.5u
L_s s1 s2 1.4m
K_tr L_p1 L_p2 L_s 0.995
R1 vin ct 1
S_1 d1 gnd g1 gnd SW
S_2 d2 gnd g2 gnd SW
D1 s1 hv DX
D2 s2 hv DX
.ends
"""


def test_center_tapped_transformer():
    d = build_diagram(parse_netlist(CENTER_TAP), "pp")
    names = [c.name for c in d.components]
    assert "K_tr" in names
    assert not {"L_p1", "L_p2", "L_s"} & set(names)  # all three windings are in the transformer
    tr = d.components[d.component_index("K_tr")]
    assert tr.symbol.pin_nodes == ("d1", "ct", "d2", "s1", "s2")
    pins = dict(zip(tr.symbol.pin_nodes, tr.symbol.pins))
    assert pins["d1"][0] == pins["ct"][0] == pins["d2"][0] < 0  # primary with tap on the left
    assert pins["d1"][1] < pins["ct"][1] < pins["d2"][1]
    assert pins["s1"][0] > 0 and pins["s2"][0] > 0
    assert "L_p1/L_p2 : L_s" in tr.label[1]
    assert tr.rotations == (0, 4)
    # The centre tap is a normal net: wired to R1.
    assert {c for c, _ in d.nets["ct"]} == {d.component_index("K_tr"), d.component_index("R1")}


def test_ground_naming_convention():
    from netlist2schematic.netlist import ground_name, is_ground
    from netlist2schematic.render import component_graphics

    for net in ("0", "gnd", "GND_iso", "vss", "VSS", "hvn", "HVN", "vee", "vee2", "agnd"):
        assert is_ground(net), net
    for net in ("vdd", "hvp", "vin", "sgnd_sense", "evee"):
        assert not is_ground(net), net
    assert ground_name("0") == ground_name("GND") == ""
    assert ground_name("hvn") == "hvn"

    # hvn starts like a supply ("hv") but gets ground symbols at every pin, no flag, no wire.
    text = """
.subckt top hvp hvn gnd
R1 hvp hvn 1k
C1 hvn gnd 1n
E1 a hvn hvp hvn 1
.ends
"""
    d = build_diagram(parse_netlist(text), "top")
    assert "port:hvn" not in [c.name for c in d.components]
    assert "hvn" not in d.power_nets
    for name in ("R1", "C1", "E1"):
        comp = d.components[d.component_index(name)]
        hvn_pins = {pi for pi, n in enumerate(comp.symbol.pin_nodes) if n == "hvn"}
        assert hvn_pins <= comp.ground_pins
        texts = [p.text for p in component_graphics(comp, 0) if hasattr(p, "text")]
        assert "hvn" in texts  # other grounds than the main one are named
    c1 = d.components[d.component_index("C1")]
    assert [p.text for p in component_graphics(c1, 0) if hasattr(p, "text")].count("hvn") == 1


BRIDGE = """
.subckt br l n plus minus
D1 l plus DM
D2 n plus DM
D3 minus l DM
D4 minus n DM
D5 x plus DM
R1 x minus 1k
.ends
.model DM D
"""


def test_four_diodes_become_one_bridge_rectifier():
    d = build_diagram(parse_netlist(BRIDGE), "br")
    names = [c.name for c in d.components]
    assert "D1/D2/D3/D4" in names and "D5" in names  # D5 is no part of the bridge
    assert not {"D1", "D2", "D3", "D4"} & set(names)
    br = d.components[d.component_index("D1/D2/D3/D4")]
    assert br.symbol.pin_nodes == ("l", "plus", "n", "minus")  # ~ left, + top, ~ right, - bottom
    assert br.symbol.pin_dirs == ((-1, 0), (0, -1), (1, 0), (0, 1))
    assert br.members == ("D1", "D2", "D3", "D4") and br.label[1] == "DM"
    # Diodes the other way round (or only three of them) are no bridge.
    three = BRIDGE.replace("D4 minus n DM\n", "")
    assert "D1/D2/D3/D4" not in [c.name for c in build_diagram(parse_netlist(three), "br").components]


def test_old_drawing_with_single_diodes_places_the_bridge_at_their_centre():
    from netlist2schematic.trainer import fitted_env

    d = build_diagram(parse_netlist(BRIDGE), "br")
    old = {"D1": (10, 10, 0), "D2": (16, 10, 0), "D3": (10, 16, 0), "D4": (16, 16, 0), "D5": (24, 13, 1)}
    env, named = fitted_env(d, old)
    assert "D1/D2/D3/D4" in named and "D1" not in named
    x, y, _ = named["D1/D2/D3/D4"]
    assert abs(x - (13 + named["D5"][0] - 24)) <= 4 and abs(y - (13 + named["D5"][1] - 13)) <= 4


def test_port_on_a_single_pin_is_labelled_at_the_pin():
    text = """
.subckt s a b gnd
R1 a m 1k
R2 m b 1k
C1 b gnd 1n
.ends
"""
    d = build_diagram(parse_netlist(text), "s")
    names = [c.name for c in d.components]
    assert "port:a" not in names and "a" in d.named_nets  # only R1 on a: label at its pin
    assert dict(d.components[d.component_index("R1")].named_pins) == {0: "a"}
    assert "port:b" in names and "b" not in d.named_nets  # R2 and C1 on b: label part with wires


def test_flyback_diodes_of_an_h_bridge_are_no_rectifier():
    text = """
.subckt hb vcc
Q1 ma b1 vcc PNP
Q2 ma b2 0 NPN
Q3 mb b3 vcc PNP
Q4 mb b4 0 NPN
D1 ma vcc DF
D2 0 ma DF
D3 mb vcc DF
D4 0 mb DF
.model PNP PNP
.model NPN NPN
.model DF D
.ends
"""
    names = [c.name for c in build_diagram(parse_netlist(text), "hb").components]
    assert {"D1", "D2", "D3", "D4"} <= set(names)


def test_ic_has_pin1_dot_block_has_none():
    from netlist2schematic.graphics import Circle
    from netlist2schematic.symbols import directional_ic, ic

    def dots(sym):
        return [p for p in sym.graphics if isinstance(p, Circle) and p.fill]

    for make in (lambda b: ic(["a", "b", "c"], ["A", "B", "C"], block=b),
                 lambda b: directional_ic(["a", "b"], ["in_i", "out_o"], ["left", "right"], block=b)):
        chip, block = make(False), make(True)
        assert len(dots(chip)) == 1 and dots(block) == []
        dot = dots(chip)[0]
        xs = [x for x, _ in chip.body]
        ys = [y for _, y in chip.body]
        assert dot.x < min(xs) and dot.y < min(ys)  # top left corner, inside the outline


def test_pin_names_turn_with_the_ic_and_stay_readable():
    from netlist2schematic.graphics import Text, transform
    from netlist2schematic.symbols import ic

    sym = ic(["a", "b"], ["IN", "OUT"])
    names = [p for p in sym.graphics if isinstance(p, Text) and p.text in ("IN", "OUT")]
    assert all(p.follow for p in names)
    for o in range(8):
        for p in names:
            t = transform(p, o, 0, 0)
            assert t.angle in (0.0, -90.0)  # never upside down or top-to-bottom
            assert t.angle == (-90.0 if o % 2 else 0.0)  # vertical whenever the box is turned
    # Value / name labels outside the box stay horizontal.
    labels = [p for p in sym.graphics if isinstance(p, Text) and not p.follow]
    assert all(transform(p, 1, 0, 0).angle == 0.0 for p in labels)
