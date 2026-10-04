"""Realize a BedMeshView under xvfb to exercise the GL setup path.

The draw itself is driven by GTK's frame clock and is timing-flaky
under Xvfb, so this test realizes the widget synchronously and
verifies shader compilation and buffer upload; the render handler
only reads the already-uploaded buffers.
"""

import pytest
from gi.repository import GLib, Gtk


@pytest.mark.ui
def test_bed_mesh_view_renders(ui_context_initializer):
    from rayforge.ui_gtk.machine.bed_mesh_view import BedMeshView

    view = BedMeshView()
    # Partial grid: the None entry must be skipped without errors.
    view.update_mesh(
        0.0,
        0.0,
        25.0,
        25.0,
        [[0.0, 0.5, None], [1.0, 1.5, 2.0], [1.5, 2.5, 3.0]],
    )
    win = Gtk.Window()
    win.set_child(view)
    win.present()
    win.realize()

    loop = GLib.MainLoop()
    GLib.timeout_add(200, loop.quit)
    loop.run()

    assert view._shader is not None, "view never realized"
    assert view._surface_count > 0
    assert view._grid_count > 0
    assert view._dots_count > 0
    # The pending mesh uploaded exactly once: the 3x3 grid minus the
    # None corner leaves 3 complete quads = 18 vertices.
    assert view._surface_count == 18
    win.destroy()
