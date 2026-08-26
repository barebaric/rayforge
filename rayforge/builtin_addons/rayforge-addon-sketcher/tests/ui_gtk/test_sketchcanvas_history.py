import gi

gi.require_version("Gtk", "4.0")
import pytest  # noqa: E402
from gi.repository import Gtk  # noqa: E402
from sketcher.core import Sketch  # noqa: E402

from rayforge.core.undo.command import Command  # noqa: E402


@pytest.fixture
def canvas():
    Gtk.init()
    window = Gtk.Window()
    from sketcher.ui_gtk.sketchcanvas import SketchCanvas

    return SketchCanvas(parent_window=window)


class NoopCommand(Command):
    def __init__(self):
        super().__init__("noop")

    def execute(self) -> None:
        pass

    def undo(self) -> None:
        pass


@pytest.mark.ui
def test_set_sketch_clears_stale_history(canvas):
    """Switching to a different Sketch instance must discard the old
    history: its commands reference the orphaned model, so undoing
    them would have no visible effect."""
    hm = canvas.sketch_editor.history_manager

    first = Sketch()
    first.add_point(5, 5)
    canvas.set_sketch(first)
    hm.execute(NoopCommand())
    assert len(hm.undo_stack) == 1

    second = Sketch()
    canvas.set_sketch(second)
    assert len(hm.undo_stack) == 0
    assert len(hm.redo_stack) == 0


@pytest.mark.ui
def test_set_sketch_keeps_history_for_same_instance(canvas):
    """Re-setting the same Sketch instance (e.g. view refresh) must
    preserve the undo history."""
    hm = canvas.sketch_editor.history_manager

    sketch = Sketch()
    sketch.add_point(5, 5)
    canvas.set_sketch(sketch)
    hm.execute(NoopCommand())
    assert len(hm.undo_stack) == 1

    canvas.set_sketch(sketch)
    assert len(hm.undo_stack) == 1
