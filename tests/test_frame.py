import pytest
from circuits import RC

from netlist2schematic.diagram import own_symbol
from netlist2schematic.frame import COLUMN, ROW, frame_graphics, sheet_frame, _row_label
from netlist2schematic.graphics import Line, Text, bbox


def texts(prims):
    return [p.text for p in prims if isinstance(p, Text)]


def test_row_labels_run_like_a_spreadsheet():
    assert [_row_label(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


def test_frame_grows_with_the_drawing_and_snaps_to_grid():
    small = frame_graphics((0, 0, 20, 10), own_symbol(("a",)), "c", "x.cir", "2026-01-01")
    large = frame_graphics((0, 0, 200, 100), own_symbol(("a",)), "c", "x.cir", "2026-01-01")
    sx0, sy0, sx1, sy1 = bbox(small)
    lx0, ly0, lx1, ly1 = bbox(large)
    assert (lx1 - lx0) > (sx1 - sx0) and (ly1 - ly0) > (sy1 - sy0)
    # Whole grid columns and rows: the outer border sits on cell edges (integer - 0.5).
    horizontal = {round(p.y1 + 0.5, 6) for p in large if isinstance(p, Line) and p.y1 == p.y2}
    vertical = {round(p.x1 + 0.5, 6) for p in large if isinstance(p, Line) and p.x1 == p.x2}
    assert all(round(v % ROW, 6) == 0 for v in horizontal if round(v) % ROW == 0)
    assert min(vertical) % COLUMN == 0


def test_drawing_lies_inside_the_frame():
    bounds = (0, 0, 60, 30)
    prims = frame_graphics(bounds, own_symbol(("vin", "gnd")), "demo", "demo.cir", "2026-01-01")
    outer = [p for p in prims if isinstance(p, Line) and p.width == 1.6]
    x_lo = min(min(p.x1, p.x2) for p in outer)
    x_hi = max(max(p.x1, p.x2) for p in outer)
    y_lo = min(min(p.y1, p.y2) for p in outer)
    y_hi = max(max(p.y1, p.y2) for p in outer)
    assert x_lo < bounds[0] - 1 and x_hi > bounds[2] + 1
    assert y_lo < bounds[1] - 1 and y_hi > bounds[3] + 1


def test_grid_labels_and_title_block():
    prims = frame_graphics((0, 0, 60, 30), own_symbol(("vin",)), "demo", "/long/path/to/demo.cir", "2026-10-27")
    labels = texts(prims)
    assert "1" in labels and "2" in labels and "A" in labels and "B" in labels
    assert "demo" in labels and "2026-10-27" in labels
    assert any(t.endswith("demo.cir") for t in labels)


def test_own_symbol_sits_bottom_left_with_pins_on_the_grid():
    symbol = own_symbol(("sig_i", "fb_o", "VDD", "GND"))
    sheet = sheet_frame((0, 0, 60, 30), symbol, "demo", "", "2026-01-01")
    ox, oy = sheet.origin
    assert isinstance(ox, int) and isinstance(oy, int)  # pins land on grid points
    pins = {symbol.pin_nodes[i]: (ox + x, oy + y) for i, (x, y) in enumerate(symbol.pins)}
    assert pins["sig_i"][0] < pins["fb_o"][0]  # input left, output right
    assert pins["VDD"][1] < pins["GND"][1]  # supply on top, ground at the bottom
    outer = [p for p in sheet.prims if isinstance(p, Line) and p.width == 1.6]
    x_lo = min(min(p.x1, p.x2) for p in outer)
    y_hi = max(max(p.y1, p.y2) for p in outer)
    assert all(x_lo < x < x_lo + 25 and y_hi - 20 < y < y_hi for x, y in pins.values())  # bottom left
    assert all(y > 30 for _, y in pins.values())  # below the drawing
    labels = texts(frame_graphics((0, 0, 60, 30), symbol, "demo", "", "2026-01-01"))
    assert {"sig_i", "fb_o", "VDD", "GND", "demo"} <= set(labels)


def test_viewer_frame_surrounds_the_parts(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from netlist2schematic.ui import MainWindow
    from netlist2schematic.training_db import TrainingDB
    from test_ui import wait_built

    app = QApplication.instance() or QApplication([])
    netlist = tmp_path / "rc.cir"
    netlist.write_text(RC)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    frame = w.sheet.frame_item
    assert frame is not None and frame.zValue() < 0
    for item in w.sheet.items.values():
        assert frame.sceneBoundingRect().contains(item.sceneBoundingRect())
    w.close()


def test_clicking_a_pin_of_the_own_symbol_selects_its_net(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from netlist2schematic.training_db import TrainingDB
    from netlist2schematic.ui import CELL, MainWindow
    from test_ui import wait_built

    app = QApplication.instance() or QApplication([])
    netlist = tmp_path / "rc.cir"
    netlist.write_text(RC)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    item = w.sheet.symbol_item
    assert item is not None and list(item.symbol.pin_nodes) == list(w.sheet.diagram.ports)
    pi = list(item.symbol.pin_nodes).index("vout")
    px, py = item.symbol.pins[pi]
    pos = QPointF((px + 0.5) * CELL, (py + 0.5) * CELL)  # the pin itself (local coordinates)
    assert item.pin_at(pos) == pi
    assert item.pin_at(QPointF(-50 * CELL, -50 * CELL)) is None
    # Same as clicking a wire of the net: the net is selected, the dock shows it, and a fly
    # line joins the own symbol's pin to the net on the sheet.
    w.sheet.select_net("vout")
    assert w.sheet._selected_nets() == ["vout"]
    texts_ = [w.properties.item(r, 1).text() for r in range(w.properties.rowCount())]
    assert "vout" in texts_
    assert len(w.sheet.fly_items) == 1
    w.sheet.scene.clearSelection()
    assert w.sheet.fly_items == []
    w.close()


def test_fly_lines_join_the_separate_places_of_a_named_net():
    import numpy as np
    from circuits import TWO_ICS

    from netlist2schematic.diagram import build_diagram
    from netlist2schematic.env import PlacementEnv
    from netlist2schematic.netlist import parse_netlist
    from netlist2schematic.render import fly_lines, net_islands
    from netlist2schematic.routing import route

    d = build_diagram(parse_netlist(TWO_ICS), "top")
    env = PlacementEnv(d)
    while not env.done:
        _, valid = env.observe()
        env.step(int(np.flatnonzero(valid)[0]))
    routing = route(d, env.placement)
    islands = net_islands(d, env.placement, routing, "mid")  # by name: X1 and X2 apart
    assert len(islands) == 2
    lines = fly_lines(islands)
    assert len(lines) == 1  # spanning tree: islands - 1
    joined = {p for line in lines for p in line}
    assert all(any(p in joined for p in island) for island in islands)
    # An extra point (the own symbol's pin) is one more island.
    assert len(fly_lines(net_islands(d, env.placement, routing, "mid", [(-50.0, -50.0)]))) == 2


def _window(tmp_path):
    from PySide6.QtWidgets import QApplication

    from netlist2schematic.training_db import TrainingDB
    from netlist2schematic.ui import MainWindow
    from test_ui import wait_built

    app = QApplication.instance() or QApplication([])
    netlist = tmp_path / "rc.cir"
    netlist.write_text(RC)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    return w


def test_supply_and_ground_pins_of_the_own_symbol_are_selectable(tmp_path):
    pytest.importorskip("PySide6")
    w = _window(tmp_path)
    sheet = w.sheet
    for net in ("vin", "gnd"):  # supply and ground: symbols at the pins, no wire
        assert net not in sheet.net_items
        sheet.select_net(net)
        assert sheet._selected_nets() == [net]
        assert len(sheet.fly_items) == 1  # own symbol pin -> the supply / ground symbol
        texts_ = [w.properties.item(r, 1).text() for r in range(w.properties.rowCount())]
        assert net in texts_
    sheet.select_net("vout", additive=True)  # Shift: both selected, both get fly lines
    assert sheet._selected_nets() == ["vout", "gnd"]
    assert len(sheet.fly_items) == 2
    sheet.view.empty_click()  # click on empty space drops the symbol-only selection too
    sheet.scene.clearSelection()
    assert sheet._selected_nets() == [] and sheet.fly_items == []
    w.close()


def test_selected_footprint_pads_give_fly_lines(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QPainterPath

    from netlist2schematic.ui import PadItem

    w = _window(tmp_path)
    panel, sheet = w.footprint_panel, w.sheet
    pads = [PadItem(QPainterPath(), "1", "vout"), PadItem(QPainterPath(), "2", "gnd"), PadItem(QPainterPath(), "3", "")]
    for pad in pads:
        panel.scene.addItem(pad)
    received = []
    panel.pads_selected.connect(received.append)
    pads[0].setSelected(True)
    assert received[-1] == ["vout"] and sheet.pad_nets == ["vout"]
    assert len(sheet.fly_items) == 1
    pads[1].setSelected(True)  # several pads: fly lines for all their nets
    pads[2].setSelected(True)  # unconnected pad: no net
    assert sorted(sheet.pad_nets) == ["gnd", "vout"] and len(sheet.fly_items) == 2
    panel.scene.clearSelection()
    assert sheet.pad_nets == [] and sheet.fly_items == []
    w.close()


def test_selecting_a_part_shows_all_its_nets_with_fly_lines(tmp_path):
    pytest.importorskip("PySide6")
    w = _window(tmp_path)
    sheet = w.sheet
    c1 = sheet.items[sheet.diagram.component_index("C1")]
    c1.setSelected(True)  # C1: vout (wired) and gnd (ground symbol, no wire)
    assert sheet.related_nets == {"vout", "gnd"}
    # Fly lines for both: gnd to the own symbol's GND pin, vout to the own symbol's vout pin.
    assert len(sheet.fly_items) == 2
    r1 = sheet.items[sheet.diagram.component_index("R1")]
    r1.setSelected(True)  # several parts: the union of their nets (vin is a supply symbol)
    assert sheet.related_nets == {"vout", "gnd", "vin"}
    assert len(sheet.fly_items) == 3
    sheet.scene.clearSelection()
    assert sheet.related_nets == set() and sheet.fly_items == []
    w.close()
