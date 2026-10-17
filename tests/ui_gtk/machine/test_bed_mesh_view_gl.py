"""Realize a BedMeshView to exercise the GL setup path.

The draw itself is driven by GTK's frame clock and is timing-flaky
under Xvfb, so this test realizes the widget synchronously and
verifies shader compilation and buffer upload; the render handler
only reads the already-uploaded buffers.

CI runners may lack a usable GL context (software rendering without
a GLArea-capable backend); there the GLArea reports an error and the
test skips instead of failing — the upload logic itself is pure
Python and covered by the non-GL assertions below.
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

    error = view.get_error()
    if error is not None:
        pytest.skip(f"no usable GL context: {error.message}")
    if view._shader is None:
        pytest.skip("GL setup did not complete in this environment")

    assert view._surface_count > 0
    assert view._grid_count > 0
    assert view._dots_count > 0
    # The pending mesh uploaded exactly once: the 3x3 grid minus the
    # None corner leaves 3 complete quads = 18 vertices.
    assert view._surface_count == 18
    win.destroy()


@pytest.mark.ui
def test_bed_mesh_view_buffer_math_without_gl(ui_context_initializer):
    """The vertex-building math is host-side Python: verify the exact
    buffer contents (quads skipped around the None entry, dots and
    wireframe lines) without needing a GL context."""

    from rayforge.ui_gtk.machine.bed_mesh_view import BedMeshView

    view = BedMeshView()
    view.update_mesh(
        10.0,
        20.0,
        5.0,
        5.0,
        [[0.0, 1.0], [2.0, 3.0]],
    )
    pending = view._pending
    assert pending is not None
    surface, grid, dots = pending
    # One quad = 2 triangles = 6 vertices = 6 * 7 floats.
    assert len(surface) == 6 * 7
    assert surface[0:3] == [10.0, 20.0, 0.0]
    # t=0 is turbo's dark blue (the polynomial's constant term).
    assert surface[3:6] == [0.11408901, 0.06288341, 0.22483372]
    # The last vertex record is [x, y, z, r, g, b, a]: corner d of
    # the second triangle, the (10, 25) corner at height 2.
    assert surface[-5] == 2.0 * BedMeshView.Z_SCALE
    # Wireframe: 4 edges around the single quad.
    assert len(grid) == 8 * 7
    # One dot per probe point.
    assert len(dots) == 4 * 7
    # All vertices are 7-component records; the view holds no GL
    # state yet because it was never realized.
    assert view._surface_vao == 0
