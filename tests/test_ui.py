from pathlib import Path

import pytest
from circuits import HUMAN, RC

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from netlist2schematic.trainer import HistoryStep  # noqa: E402
from netlist2schematic.training_db import TrainingDB  # noqa: E402
from netlist2schematic.ui import CELL, MainWindow, TrainingDataDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    # A plain .cir without bom.yaml: nothing is written next to the netlist.
    netlist = tmp_path / "rc.cir"
    netlist.write_text(RC)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    yield w
    w.close()


def wait_built(app, w):
    """Let the deferred load and the step-by-step proposal run in the event loop."""
    for _ in range(1000):
        app.processEvents()
        if w.sheet.diagram is not None and not w.sheet.building:
            return
    raise AssertionError("proposal did not finish")


def set_placement(window, named):
    window.sheet.set_history([HistoryStep(window.sheet.diagram.component_index(n), pos, None, kind="given") for n, pos in named.items()])


def test_loads_circuit_with_full_history(window):
    assert window.circuit_box.currentText() == "rc"
    assert len(window.sheet.placement) == len(window.sheet.diagram.components)
    assert window.sheet.slider.maximum() == len(window.sheet.history) >= len(window.sheet.diagram.components)
    assert window.sheet.placement == window.sheet.state_at(len(window.sheet.history))


def test_history_slider_shows_parts_step_by_step(window):
    window.sheet.set_history([h for h in window.sheet.history if h.kind == "place"])
    window.sheet.slider.setValue(1)
    visible = [ci for ci, item in window.sheet.items.items() if item.isVisible()]
    assert visible == [window.sheet.history[0].component]
    assert window.sheet.heat.heat is not None
    window.sheet.slider.setValue(window.sheet.slider.maximum())
    assert all(item.isVisible() for item in window.sheet.items.values())


def test_correction_discards_future_and_next_predicts_one_step(window):
    window.sheet.slider.setValue(1)
    first = window.sheet.history[0].component
    item = window.sheet.items[first]
    item.setSelected(True)
    item.setPos(item.pos().x() - 60 * CELL, item.pos().y())
    window.sheet.drop_selection()  # recorded as step 2, later steps discarded, nothing recomputed
    assert len(window.sheet.history) == 2
    correction = window.sheet.history[1]
    assert correction.kind == "manual" and correction.component == first
    assert correction.previous == window.sheet.history[0].position
    assert window.sheet.slider.value() == window.sheet.slider.maximum() == 2
    window.sheet.next_step()  # exactly one new step from the policy
    assert len(window.sheet.history) == 3
    assert window.sheet.history[2].kind in ("place", "move") and window.sheet.history[2].heat is not None
    assert window.sheet.placement[first] == correction.position  # the corrected part stays
    while not window.sheet.rollout.done:
        window.sheet.next_step()
    assert {h.component for h in window.sheet.history} == set(range(len(window.sheet.diagram.components)))
    assert window.sheet.placement == window.sheet.state_at(len(window.sheet.history))


def test_edit_at_end_is_recorded_without_regenerating(window):
    n = len(window.sheet.history)
    future = list(window.sheet.history)
    r1 = window.sheet.diagram.component_index("R1")
    item = window.sheet.items[r1]
    item.setSelected(True)
    item.setPos(item.pos().x() + 40 * CELL, item.pos().y())
    window.sheet.drop_selection()
    assert window.sheet.history[:n] == future
    assert window.sheet.history[n].kind == "manual"
    assert window.sheet.slider.value() == n + 1 == window.sheet.slider.maximum()


def test_illegal_drop_snaps_or_reverts(window):
    set_placement(window, HUMAN)
    c1 = window.sheet.diagram.component_index("C1")
    r1 = window.sheet.diagram.component_index("R1")
    window.sheet.items[c1].setSelected(True)
    window.sheet.items[c1].setPos(window.sheet.items[r1].pos())  # on top of R1
    window.sheet.drop_selection()
    assert window.sheet.placement[c1] != window.sheet.placement[r1]


def test_group_move_and_mirror(window):
    set_placement(window, HUMAN)
    for item in window.sheet.items.values():
        item.setSelected(True)
        item.setPos(item.pos().x() + 200 * CELL, item.pos().y() - 100 * CELL)
    window.sheet.drop_selection()
    r1 = window.sheet.diagram.component_index("R1")
    assert window.sheet.placement[r1][:2] == (HUMAN["R1"][0] + 200, HUMAN["R1"][1] - 100)
    for item in window.sheet.items.values():
        item.setSelected(item.ci == r1)
    window.sheet.mirror_selected()
    assert window.sheet.placement[r1][2] == HUMAN["R1"][2] ^ 4


def test_send_training_data_and_retrain(window, app):
    set_placement(window, HUMAN)
    window.send_training_data()
    assert window.db.examples("rc")[0].placement == HUMAN
    window.retrain()
    window.worker.wait(120_000)
    app.processEvents()
    assert window.progress_label.text().startswith("Training fertig")
    assert Path(window.model).exists()


def test_incomplete_history_is_not_sent(window):
    window.sheet.slider.setValue(1)
    window.send_training_data()
    assert window.db.count() == 0


def test_rotation_shifts_part_when_needed(window):
    set_placement(window, HUMAN)
    r1 = window.sheet.diagram.component_index("R1")
    for item in window.sheet.items.values():
        item.setSelected(item.ci == r1)
    before = window.sheet.placement[r1]
    for _ in range(4):
        window.sheet.rotate_selected()
    assert window.sheet.placement[r1][2] == before[2]  # four quarter turns
    assert all(h.kind in ("given", "manual") for h in window.sheet.history)


def test_delete_weights(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from netlist2schematic import policy

    policy.save(policy.init_params(0), Path(window.model))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.delete_weights()
    assert not Path(window.model).exists()


def test_training_data_dialog_deletes_selected(window):
    for _ in range(3):
        window.db.add("rc", RC, HUMAN)
    dialog = TrainingDataDialog(window.db, window)
    assert dialog.table.rowCount() == 3
    column = TrainingDataDialog.COLUMNS.index("Varianten")
    assert int(dialog.table.item(0, column).text()) > 0
    dialog.table.selectRow(0)
    ids = dialog.selected_ids()
    dialog.delete(ids, confirm=False)
    assert dialog.table.rowCount() == 2
    assert window.db.count() == 2
    assert ids[0] not in [e.id for e in window.db.examples()]


def test_net_highlight_and_properties(window):
    set_placement(window, HUMAN)
    net_item = window.sheet.net_items["vout"]
    net_item.setSelected(True)
    texts = [window.properties.item(r, 1).text() for r in range(window.properties.rowCount())]
    assert "vout" in texts
    window.sheet.highlight_selected_nets()
    assert "vout" in window.sheet.highlights
    window.sheet.clear_highlights()
    assert window.sheet.highlights == {}
    window.sheet.scene.clearSelection()
    r1 = window.sheet.diagram.component_index("R1")
    window.sheet.items[r1].setSelected(True)
    texts = [window.properties.item(r, 1).text() for r in range(window.properties.rowCount())]
    assert "R1" in texts and "Widerstand" in texts


def test_open_training_example_shows_its_history(window):
    example_id = window.db.add("rc", RC, HUMAN)
    dialog = TrainingDataDialog(window.db, window)
    column = TrainingDataDialog.COLUMNS.index("Training")
    assert dialog.table.item(0, column).text() == "ja"
    dialog.open(example_id)
    assert window.sheet.placement == {window.sheet.diagram.component_index(n): pos for n, pos in HUMAN.items()}
    assert all(h.kind == "example" and h.heat is not None for h in window.sheet.history)
    assert window.sheet.slider.value() == 1
    assert window.sheet.source.circuit_dir is None  # nothing is written next to a netlist file


def test_window_shows_before_loading_and_builds_step_by_step(app, tmp_path):
    netlist = tmp_path / "rc.cir"
    netlist.write_text(RC)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    assert w.sheet is None  # nothing loaded in the constructor: the window appears at once
    app.processEvents()  # deferred load: diagram built, proposal started
    assert w.sheet.diagram is not None and w.sheet.building
    seen = []
    while w.sheet.building:
        seen.append(len(w.sheet.history))
        w.sheet._build_step()
    assert seen == sorted(seen) and seen[0] < len(w.sheet.history)  # grows one step at a time
    assert len(w.sheet.placement) == len(w.sheet.diagram.components)
    # Esc / an edit stops a running build; "Rest vorschlagen" continues from the slider.
    w.sheet.slider.setValue(1)
    w.sheet.regenerate_future()
    assert w.sheet.building and len(w.sheet.history) == 1
    w.sheet.stop_build()
    assert not w.sheet.building
    w.sheet.toggle_build()
    w.sheet.finish_build()
    assert len(w.sheet.placement) == len(w.sheet.diagram.components)
    w.close()


TOP_WITH_BLOCK = """
.subckt top in out gnd
X_F in mid gnd filt
X_U mid out gnd CHIP
.ends
.subckt filt a b g
R1 a b 1k
C1 b g 1n
.ends
.subckt CHIP IN OUT GND
R9 IN OUT 1
.ends
"""


def test_subcircuit_block_opens_in_a_tab_on_double_click(app, tmp_path, monkeypatch):
    from netlist2schematic import ui
    from netlist2schematic.sources import Source

    # A design subcircuit ("filt") is a plain block; a vendor model ("CHIP") stays an IC.
    real = ui.load_source
    monkeypatch.setattr(ui, "load_source", lambda path: Source(**{**real(path).__dict__, "subcircuits": frozenset({"filt"})}))
    netlist = tmp_path / "top.cir"
    netlist.write_text(TOP_WITH_BLOCK)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    top = w.sheet
    assert top.circuit == "top" and w.tabs.count() == 1
    block = top.diagram.components[top.diagram.component_index("X_F")]
    chip = top.diagram.components[top.diagram.component_index("X_U")]
    assert block.subckt == "filt" and chip.subckt == ""
    texts = lambda c: [p.text for p in c.symbol.graphics if hasattr(p, "text")]
    assert "1" not in texts(block) and "1" in texts(chip)  # no package pin numbers on blocks
    assert len(block.symbol.graphics) < len(chip.symbol.graphics)  # and no DIP notch

    top.items[top.diagram.component_index("X_F")].mouseDoubleClickEvent(None)
    wait_built(app, w)
    assert w.tabs.count() == 2 and w.sheet.circuit == "filt" and w.tabs.tabText(1) == "filt"
    assert w.circuit_box.currentText() == "filt"
    assert w.netlist_view.textCursor().selectedText().lower().startswith(".subckt filt")
    # Opening it again switches to the existing tab; the first tab keeps its own drawing.
    w.tabs.setCurrentIndex(0)
    w.open_subcircuit(top.source, top.netlist, "filt")
    assert w.tabs.count() == 2 and w.tabs.currentIndex() == 1
    w.tabs.setCurrentIndex(0)
    assert w.sheet is top and w.circuit_box.currentText() == "top"
    w.close_tab(1)
    assert w.tabs.count() == 1
    w.close()


def test_toolbar_has_icon_actions_and_netlist_dock(window):
    assert window.toolbar.toolButtonStyle().name == "ToolButtonIconOnly"
    assert all(not a.icon().isNull() for a in window.actions.values())
    # Every icon button explains itself: name, a description and the shortcut when it has one.
    for key, action in window.actions.items():
        lines = action.toolTip().split("\n")
        assert len(lines) == 2 and lines[0].strip() and len(lines[1]) > 15, key
        assert action.statusTip() == lines[1], key
    assert window.actions["rotate"].toolTip().startswith("Drehen (R)")
    assert window.actions["suggest"].toolTip().startswith("Vorschlag\n")
    assert window.actions["rotate"].shortcut().toString() == "R"
    assert window.netlist_dock.widget() is not None and ".subckt rc" in window.netlist_view.toPlainText()


def _view(app):
    from PySide6.QtWidgets import QGraphicsScene

    from netlist2schematic.ui import SchematicScene, SchematicView

    scene = SchematicScene(None)
    scene.setSceneRect(-2_000_000.0, -2_000_000.0, 4_000_000.0, 4_000_000.0)
    view = SchematicView(scene)
    view.scene_ref = scene  # the view does not own its scene: keep it alive
    view.resize(800, 600)
    view.show()
    app.processEvents()
    return view


def test_zoom_keeps_point_under_pointer(app):
    from PySide6.QtCore import QPointF

    view = _view(app)
    for pos in (QPointF(120, 90), QPointF(640, 480), QPointF(400, 300)):
        anchor = view.mapToScene(pos.toPoint())
        for i in range(120):  # zoom in and out many times: the point must not drift
            view.zoom_at([1.15, 1.15, 0.8, 1.3, 0.5][i % 5], pos)
            QApplication.processEvents()
            scale = view.transform().m11()
            now = view.mapToScene(pos.toPoint())
            # Scroll positions are whole pixels: at most a couple of pixels off, never more.
            assert abs(now.x() - anchor.x()) * scale < 2.0 and abs(now.y() - anchor.y()) * scale < 2.0
    view.close()


def test_mouse_wheel_zooms_without_modifier_and_trackpad_pans(app):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    view = _view(app)
    scale = view.transform().m11()

    def wheel(phase, pixel, angle):
        event = QWheelEvent(
            QPointF(300, 200), QPointF(300, 200), QPoint(*pixel), QPoint(0, angle),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, phase, False,
        )
        view.wheelEvent(event)

    # Mouse wheel on macOS: pixel delta present, but no scroll phase -> zoom.
    wheel(Qt.ScrollPhase.NoScrollPhase, (0, 12), 120)
    assert view.transform().m11() > scale
    zoomed = view.transform().m11()
    # Trackpad two-finger scroll: scroll phase -> pan, scale unchanged.
    centre = view.mapToScene(view.viewport().rect().center())
    wheel(Qt.ScrollPhase.ScrollUpdate, (0, 12), 0)
    assert view.transform().m11() == zoomed
    moved = view.mapToScene(view.viewport().rect().center()) - centre
    assert abs(moved.y()) > 0
    view.close()


def test_hierarchy_dock_selects_device_in_its_parent_sheet(app, tmp_path):
    netlist = tmp_path / "top.cir"
    netlist.write_text(TOP_WITH_BLOCK)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    root = w.hierarchy.topLevelItem(0)
    assert root.text(0) == "top"
    children = {root.child(i).text(0): root.child(i) for i in range(root.childCount())}
    assert children["X_F"].childCount() > 0  # subcircuit instances expand to their elements
    inner = children["X_F"].child(0)
    w.hierarchy.setCurrentItem(inner)  # an element inside "filt": its sheet opens in a new tab
    wait_built(app, w)
    sheet = w.sheet
    assert w.tabs.count() == 2 and sheet.circuit == "filt"
    ci = sheet.diagram.component_of(inner.text(0))
    assert sheet.items[ci].isSelected() and sheet.pending_focus is None
    w.hierarchy.setCurrentItem(children["X_U"])  # back to the top sheet, the IC is selected there
    assert w.sheet.circuit == "top" and w.sheet.items[w.sheet.diagram.component_index("X_U")].isSelected()
    w.close()


def test_groups_are_coloured_blocks_with_description(app, tmp_path):
    from test_groups import GROUPED

    netlist = tmp_path / "grouped.cir"
    netlist.write_text(GROUPED)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    sheet = w.sheet
    assert set(sheet.group_items) == {"stage1", "stage2", "rest"}
    sheet.group_items["stage1"].setSelected(True)
    rows = {w.properties.item(r, 0).text(): w.properties.item(r, 1).text() for r in range(w.properties.rowCount())}
    assert rows["Gruppe"] == "Stufe 1" and rows["Beschreibung"] == "first RC stage"
    sheet.scene.clearSelection()
    sheet.items[sheet.diagram.component_index("R2")].setSelected(True)
    rows = {w.properties.item(r, 0).text(): w.properties.item(r, 1).text() for r in range(w.properties.rowCount())}
    assert rows["Gruppe"] == "Stufe 2"
    w.close()


def test_footprint_dock_shows_selected_part(app, tmp_path):
    from test_footprint import WITH_PARTS

    from netlist2schematic import footprint as FP

    netlist = tmp_path / "parts.cir"
    netlist.write_text(WITH_PARTS)
    w = MainWindow(TrainingDB(tmp_path / "t.sqlite"), tmp_path / "policy.npz", netlist)
    wait_built(app, w)
    sheet = w.sheet
    sheet.items[sheet.diagram.component_index("X_U1")].setSelected(True)
    panel = w.footprint_panel
    assert [p.ref for p in panel.parts] == ["U1"] and "U1" in panel.info.text()
    rows = {w.properties.item(r, 0).text(): w.properties.item(r, 1).text() for r in range(w.properties.rowCount())}
    assert rows["Footprint"] == "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
    if FP.footprint_dir() is not None:
        assert panel.footprint is not None and panel.scene.items()
    panel.wait()
    w.close()
