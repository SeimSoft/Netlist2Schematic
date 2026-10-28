from netlist2schematic import groups as G
from netlist2schematic import trainer
from netlist2schematic.diagram import REST, build_diagram
from netlist2schematic.netlist import parse_netlist
from netlist2schematic.placement import relative_cells
from netlist2schematic.render import group_boxes, schematic_svg

GROUPED = """
.subckt top VDD GND in_ai out_ao
*@groups
* description: Two stage test circuit
* groups:
*   - id: stage1
*     title: Stufe 1
*     members: [R1, C1]
*     description: >
*       first RC
*       stage
*   - id: stage2
*     title: Stufe 2
*     members: [R2, C2, port:out_ao]
*     color: "#123456"
*@end
R1 in_ai mid 1k
C1 mid GND 1n
R2 mid out_ao 1k
C2 out_ao GND 1n
R3 out_ao load 1k
R4 load GND 1k
.ends
"""


def diagram():
    return build_diagram(parse_netlist(GROUPED), "top")


def test_header_is_parsed_with_description_and_patterns():
    s = parse_netlist(GROUPED).get("top")
    assert s.description == "Two stage test circuit"
    assert [g.id for g in s.groups] == ["stage1", "stage2"]
    assert s.groups[0].description == "first RC stage" and s.groups[1].color == "#123456"
    assert s.groups[1].matches("port:out_ao") and not s.groups[0].matches("R2")


def test_components_get_their_group_and_rest_collects_the_others():
    d = diagram()
    group = {c.name: c.group for c in d.components}
    assert group["R1"] == group["C1"] == "stage1" and group["R2"] == group["C2"] == "stage2"
    assert group["R3"] == group["R4"] == REST
    assert [g.id for g in d.groups] == ["stage1", "stage2", REST]
    assert group["port:out_ao"] == "stage2"  # port listed as port:<net>


def test_floorplan_regions_do_not_overlap_and_follow_the_signal_flow():
    d = diagram()
    regions = G.floorplan(d)
    assert list(regions)[0] == "stage1"  # holds the input port
    rects = list(regions.values())
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def test_suggestion_keeps_parts_in_their_group_region():
    d = diagram()
    suggestion = trainer.suggest(d, None)
    assert set(suggestion.regions) == {"stage1", "stage2", REST}
    inside = 0
    for name, (x, y, o) in suggestion.placement.items():
        ci = d.component_index(name)
        comp = d.components[ci]
        body, pins = relative_cells(d, ci, o)
        cells = [(x + cx, y + cy) for cx, cy in body + pins]
        x0, y0, x1, y1 = suggestion.regions[comp.group]
        inside += all(x0 - 3 <= cx <= x1 + 3 and y0 - 3 <= cy <= y1 + 3 for cx, cy in cells)
    assert inside == len(suggestion.placement)


def test_group_boxes_in_svg():
    d = diagram()
    suggestion = trainer.suggest(d, None)
    placement = {d.component_index(n): p for n, p in suggestion.placement.items()}
    boxes = group_boxes(d, placement)
    assert {b.gid for b in boxes} == {"stage1", "stage2", REST}
    assert next(b for b in boxes if b.gid == "stage2").color == "#123456"
    assert "Stufe 1" in schematic_svg(d, placement) and 'fill-opacity="0.1"' in schematic_svg(d, placement)
    planned = group_boxes(d, {}, suggestion.regions)
    assert all(b.planned for b in planned) and len(planned) == 3


def test_floorplan_is_landscape_not_a_column():
    # Three groups of similar size: packed into rows so the plan is wider than tall.
    blocks = []
    for k in range(3):
        blocks.append(f"*   - id: g{k}\n*     title: G{k}\n*     members: [R{k}a, R{k}b, R{k}c, C{k}]")
    elements = []
    for k in range(3):
        elements += [f"R{k}a n{k} m{k} 1k", f"R{k}b m{k} p{k} 1k", f"R{k}c p{k} q{k} 1k", f"C{k} q{k} 0 1n"]
    text = ".subckt big n0 n1 n2\n*@groups\n* groups:\n" + "\n".join(blocks) + "\n*@end\n" + "\n".join(elements) + "\n.ends\n"
    regions = G.floorplan(build_diagram(parse_netlist(text), "big"))
    x0, y0, x1, y1 = G.extent(regions)
    assert x1 - x0 > y1 - y0
