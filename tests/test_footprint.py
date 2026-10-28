import pytest

from netlist2schematic import footprint as FP
from netlist2schematic.diagram import build_diagram
from netlist2schematic.netlist import parse_netlist

WITH_PARTS = """
.subckt top VDD GND in_ai out_ao
R1 in_ai out_ao 1k
X_U1 in_ai out_ao VDD GND AMP
*@parts
* R1:
* - {ref: R1, footprint: 'Resistor_SMD:R_0603_1608Metric', value: 1k, description: series resistor}
* X_U1:
* - ref: U1
*   footprint: Package_SO:SOIC-8_3.9x4.9mm_P1.27mm
*   value: OPA1
*   pins: {'1': out_o, '2': in_ai, '4': GND, '8': VDD}
*@end
.ends
.subckt AMP in_ai out_o VDD GND
R9 in_ai out_o 1
.ends
"""


def test_parts_header_and_pad_nets():
    nl = parse_netlist(WITH_PARTS)
    parts = nl.get("top").parts
    assert parts["R1"][0].footprint == "Resistor_SMD:R_0603_1608Metric" and parts["X_U1"][0].ref == "U1"
    d = build_diagram(nl, "top")
    assert [p.ref for p in d.parts_of(d.component_index("X_U1"))] == ["U1"]
    assert d.pad_nets["U1"] == {"1": "out_ao", "2": "in_ai", "4": "GND", "8": "VDD"}
    assert d.pad_nets["R1"] == {"1": "in_ai", "2": "out_ao"}  # two-terminal part without pin map


needs_kicad = pytest.mark.skipif(FP.footprint_dir() is None, reason="KiCad footprint library not installed")


@needs_kicad
def test_footprint_graphics_have_pads_with_numbers_and_nets():
    fp = FP.load("Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    assert len(fp.pads) == 8 and fp.model.endswith(".step")
    texts = [p.text for p in FP.graphics(fp, {"1": "out_ao"}) if hasattr(p, "text")]
    assert {"1", "8", "out_ao"} <= set(texts)
    board = FP.board_text(fp, "U1")
    assert '"Package_SO:SOIC-8' in board and "(at 0 0)" in board and '"U1"' in board and "Edge.Cuts" in board


@needs_kicad
@pytest.mark.skipif(FP.kicad_cli() is None, reason="kicad-cli not installed")
def test_render_3d_is_cached(tmp_path):
    fp = FP.load("Resistor_SMD:R_0603_1608Metric")
    png = FP.render_3d(fp, tmp_path, "Iso", "R1", size=(200, 150))
    assert png.exists() and png.stat().st_size > 1000
    assert FP.render_3d(fp, tmp_path, "Iso", "R1", size=(200, 150)) == png
