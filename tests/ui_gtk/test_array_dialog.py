# flake8: noqa: E402
import pytest

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import GLib, Gtk

from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.array import (
    ArrayMode,
    ArrayParams,
    CircularArrayParams,
)
from rayforge.doceditor.editor import DocEditor
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.array_dialog import (
    CircularArrayDialog,
    GridArrayDialog,
)
from rayforge.ui_gtk.canvas2d.elements.outline import OutlineElement
from rayforge.ui_gtk.canvas2d.surface import WorkSurface
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float


def _process_events():
    """Drains pending GTK events so deferred signals (e.g. close-request)
    are delivered."""
    ctx = GLib.main_context_default()
    while ctx.pending():
        ctx.iteration(False)


def _has_preview(surface) -> bool:
    return any(isinstance(c, OutlineElement) for c in surface.root.children)


@pytest.fixture
def array_dialog_env(ui_context_initializer, ui_task_mgr):
    """Builds a real editor + surface with two workpieces selected."""
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    layer = editor.doc.active_layer
    wp1 = WorkPiece(name="wp1")
    wp1.set_size(10, 10)
    wp1.pos = (0, 0)
    wp2 = WorkPiece(name="wp2")
    wp2.set_size(10, 10)
    wp2.pos = (20, 0)
    layer.add_child(wp1)
    layer.add_child(wp2)

    machine = Machine(ui_context_initializer)
    machine.set_axis_extents(200, 200)
    parent = Gtk.Window()
    surface = WorkSurface(editor, parent, machine)
    surface.update_from_doc()

    yield editor, surface, parent, [wp1, wp2]

    editor.cleanup()


@pytest.mark.ui
def test_preview_attached_and_removed_on_destroy(array_dialog_env):
    editor, surface, parent, items = array_dialog_env
    dialog = GridArrayDialog(parent, editor, surface, items)

    # The preview overlay is attached to the canvas root while open.
    assert _has_preview(surface)

    # Default params (2x2 grid) yield 4 instances.
    deltas = editor.array.compute_plan(items, ArrayParams())
    assert len(deltas) == 4

    # Close the dialog via the window close path (Cancel). The window
    # must be realized for close() to emit close-request.
    dialog.present()
    _process_events()
    dialog.close()
    _process_events()
    assert not _has_preview(surface)


@pytest.mark.ui
def test_apply_creates_copies_and_cleans_up(array_dialog_env):
    editor, surface, parent, items = array_dialog_env
    dialog = GridArrayDialog(parent, editor, surface, items)

    before = len(editor.doc.active_layer.get_content_items())
    # Click Apply (as the user would).
    dialog._on_apply_clicked(None)

    # (2x2 - 1) instances * 2 source items = 6 new copies.
    after = len(editor.doc.active_layer.get_content_items())
    assert after == before + 6
    # The preview overlay is removed after the dialog closes.
    assert not _has_preview(surface)
    # The original selection is selected together with the new copies.
    selected_uids = {
        e.data.uid
        for e in surface.root.get_all_children_recursive()
        if e.selected and e.data is not None
    }
    assert all(item.uid in selected_uids for item in items)


@pytest.mark.ui
def test_circular_dialog_preview(array_dialog_env):
    editor, surface, parent, items = array_dialog_env
    dialog = CircularArrayDialog(parent, editor, surface, items)

    # The circular dialog produces a circular plan directly.
    params = dialog._current_params()
    assert params.mode == ArrayMode.CIRCULAR
    assert isinstance(params.circular, CircularArrayParams)

    deltas = editor.array.compute_plan(items, params)
    # Default circular count is 6 -> 6 instances (instance 0 is identity).
    assert len(deltas) == 6

    dialog.present()
    _process_events()
    dialog.close()
    _process_events()
    assert not _has_preview(surface)


@pytest.mark.ui
def test_spacing_mode_translates_distance(array_dialog_env):
    """Switching Displacement<->Gap converts the current distance,
    and the preview updates immediately on the switch.

    Selection bbox is 30 wide x 10 tall (wp1 at 0, wp2 at 20, each 10).
    """
    editor, surface, parent, items = array_dialog_env
    dialog = GridArrayDialog(parent, editor, surface, items)

    # Defaults: Gap mode, 1 mm on both axes.
    assert dialog._spacing_mode_row.get_selected() == 1  # Gap
    assert get_spinrow_float(dialog._col_spacing_row) == 1.0

    # Switch Gap -> Displacement: displacement = gap + unit size.
    dialog._spacing_mode_row.set_selected(0)
    _process_events()
    assert get_spinrow_float(dialog._col_spacing_row) == pytest.approx(
        31.0
    )  # 1 + 30
    assert get_spinrow_float(dialog._row_spacing_row) == pytest.approx(
        11.0
    )  # 1 + 10

    # The preview must have refreshed with the new spacing: pitch is now
    # 31/11, so the (0,1) cell sits at (31, 0).
    params = dialog._current_params()
    deltas = editor.array.compute_plan(items, params)
    assert deltas[1].get_translation() == pytest.approx((31.0, 0.0))

    # Switch back Displacement -> Gap: gap = displacement - unit size.
    dialog._spacing_mode_row.set_selected(1)
    _process_events()
    assert get_spinrow_float(dialog._col_spacing_row) == pytest.approx(1.0)
    assert get_spinrow_float(dialog._row_spacing_row) == pytest.approx(1.0)

    dialog.close()
